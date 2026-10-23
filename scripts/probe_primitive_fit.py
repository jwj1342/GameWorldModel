"""离散的形状选择，改成对**汇聚几何做拟合**——创新点一的第一个实例。

要解决的问题。我们的输出是程序，里面全是离散选择：选哪个基本体、哪个类、哪根铰链。
梯度传不过去，所以连续的稠密几何没法直接监督它们。MVP 一证明「把渲染结果和原帧比、
拿整帧标量分数当信号」这条走不通（效应/噪声 0.35，方向还会反），它的诊断指向一个
很具体的替代：**别在二维整帧比，要逐物体、在三维里比。**

形状是最干净的一个实例。现在的链路是：检测短语 → `fallbackForClass` 的类名正则 →
基本体。实测类名准确率只有 12.5%，而形状本来是个**几何**问题，不是语义问题。
这里换成：把物体的点跨帧汇到一起，分别拟合 box / sphere / cylinder / cone，按残差选。

口径上必须先防住两件事：

**一，类别不平衡。** 真值里 box 97、sphere 16、cone 9、cylinder 6——**「恒答 box」
就有 0.758**。不跟这个基线比的准确率没有意义（这个坑踩过：八个配置的差全为正，
换成条件概率之后六个是负的）。所以这里一定报「恒答 box」、逐类召回和混淆矩阵，
不只报总准确率。

**二，关联。** 形状对不对，要知道这个预测物体对应哪个真值节点。用位置配对来定
关联、再去评形状，会把两件事绞在一起。所以这个实验跑在**真值掩码**上
（`segmentation_backend: gt`，掩码直接以真值节点 id 为键），几何仍然是真实的 VGGT。
这样量到的是「拟合这一步」本身的上限，关联不进来当混杂。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.perception.masks import resize_mask               # noqa: E402

KINDS = ("box", "sphere", "cylinder", "cone")


def unproject(depth, K, mask, stride=2):
    H, W = depth.shape
    ys, xs = np.where(mask[::stride, ::stride])
    ys, xs = ys * stride, xs * stride
    ok = depth[ys, xs] > 1e-4
    ys, xs = ys[ok], xs[ok]
    if not len(ys):
        return np.zeros((0, 3))
    z = depth[ys, xs]
    return np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1)


CV2THREE = np.diag([1.0, -1.0, -1.0])


def to_world(P, c2w):
    if not len(P):
        return P
    F = np.eye(4); F[:3, :3] = CV2THREE
    M = F @ c2w
    return P @ M[:3, :3].T + M[:3, 3]


def normalise(P: np.ndarray, trim: float = 0.05) -> np.ndarray:
    """移到质心、按均方半径缩成 1；可选先剪掉最外面一圈离群点。

    残差因此是无量纲的，物体之间可比。

    **为什么要剪。** 跨帧并起来的点云在这条链路上是被离群点污染的——`98a48da`
    量过，并云拟合的位置误差 0.642 米，比现算的首帧 0.420 还差。所以直接在并云上
    最小二乘，很可能是在拟合噪声而不是形状。这里按到质心的距离剪掉最远的一小撮，
    并且下面会**同时在逐帧点云上拟合一遍做对照**——如果并云那一列明显更差，
    那就说明形状分类也不该用并云。
    """
    C = P - P.mean(0)
    if 0 < trim < 0.5 and len(C) > 50:
        d = np.linalg.norm(C, axis=1)
        C = C[d <= np.percentile(d, 100 * (1 - trim))]
        C = C - C.mean(0)
    r = float(np.sqrt((C ** 2).sum(1).mean()))
    return C / r if r > 1e-9 else C


def _axis_frame(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (沿 y 的高度, 到 y 轴的水平半径)。世界已经是 y 朝上。"""
    return P[:, 1], np.linalg.norm(P[:, [0, 2]] - P[:, [0, 2]].mean(0), axis=1)


# —— 拟合与评估分开，这样可以做交叉验证 ——
#
# **朴素做法（各自拟合、直接比原始残差）实测是失败的**：输给「恒答 box」26 个点，
# cylinder 0/22、cone 0/16 一个都没答对，混淆矩阵里 cone 那一列全是零。
#
# 原因不是拟合不准，是**不同自由度的模型的原始残差不可比**。盒子有三个半长可调、
# 会「缩包」去贴任何点云；锥面只有一个斜率自由度。谁灵活谁的残差小，所以盒子几乎
# 总赢。这是个**模型选择**问题，不是拟合问题。
#
# 交叉验证天然解决它：在一半点上拟合、在另一半上评。会缩包的模型在没见过的点上
# 泛化更差，于是不需要人为设自由度惩罚。
#
# 每个 fit_* 返回一个闭包 `evaluate(Q) -> 平均距离`，参数只由训练那一半决定。


def fit_box(P):
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)

    def evaluate(Q):
        q = np.abs(Q - c) - h
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0)
        return float(np.mean(np.abs(outside + inside)))
    return evaluate


def fit_sphere(P):
    """代数最小二乘球：|p|^2 = 2p·c + (r^2 - |c|^2)。"""
    A = np.hstack([2 * P, np.ones((len(P), 1))])
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    c, t = sol[:3], sol[3]
    r2 = t + (c ** 2).sum()
    if r2 <= 0:
        return lambda Q: float("inf")
    r = math.sqrt(r2)
    return lambda Q: float(np.mean(np.abs(np.linalg.norm(Q - c, axis=1) - r)))


def circle_fit(xz: np.ndarray) -> tuple[np.ndarray, float]:
    """代数圆拟合（Kåsa）：解 (圆心, 半径)，最小化 |p|² − 2p·c − (r²−|c|²)。

    **为什么必须解圆心，不能拿质心代替。** 第一版两个地方都用 `xz.mean(0)` 当轴心，
    结果 cylinder 召回 0/22、cone 0/16，混淆矩阵里 cone 整列为零。原因是部分可见的
    点云只覆盖物体的一侧，它的质心落在**表面上**而不是轴上；于是「到轴的距离」算出来
    近处的点很小、远处的点很大，恒定半径的柱面模型当然拟合得极差。

    圆拟合对**圆弧**同样良态（只要弧有足够曲率），所以部分可见不是问题——
    偏心才是问题。
    """
    A = np.hstack([2 * xz, np.ones((len(xz), 1))])
    sol, *_ = np.linalg.lstsq(A, (xz ** 2).sum(1), rcond=None)
    c = sol[:2]
    r2 = sol[2] + (c ** 2).sum()
    return c, (math.sqrt(r2) if r2 > 0 else 0.0)


def fit_cylinder(P):
    """轴平行 y（世界已对齐到 y 朝上），轴心和半径由 xz 上的圆拟合解出。"""
    c, R = circle_fit(P[:, [0, 2]])
    if R <= 1e-9:
        return lambda Q: float("inf")
    return lambda Q: float(np.mean(np.abs(np.linalg.norm(Q[:, [0, 2]] - c, axis=1) - R)))


def fit_cone(P):
    """轴平行 y，半径随高度线性收到顶点：R(y) = a·(y_top − y)。

    轴心同样用圆拟合解（取中段的点，那里半径变化小，圆拟合最稳），再在
    (高度, 到轴的距离) 平面上解斜率和顶点高度。
    """
    y = P[:, 1]
    lo, hi = np.percentile(y, 20), np.percentile(y, 60)
    band = P[(y >= lo) & (y <= hi)]
    c, _r = circle_fit(band[:, [0, 2]] if len(band) >= 10 else P[:, [0, 2]])
    rad = np.linalg.norm(P[:, [0, 2]] - c, axis=1)
    # 半径随高度线性：rad ≈ a·(y_top − y)。两个未知量一起解（斜率 a 和顶点 y_top），
    # 写成 rad ≈ -a·y + b，再回推 y_top = b/a。
    A = np.hstack([-y[:, None], np.ones((len(y), 1))])
    sol, *_ = np.linalg.lstsq(A, rad, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if abs(a) < 1e-9:
        return lambda Q: float("inf")
    y_top = b / a

    def evaluate(Q):
        rq = np.linalg.norm(Q[:, [0, 2]] - c, axis=1)
        return float(np.mean(np.abs(rq - np.maximum(a * (y_top - Q[:, 1]), 0.0))))
    return evaluate


FITTERS = {"box": fit_box, "sphere": fit_sphere,
           "cylinder": fit_cylinder, "cone": fit_cone}


def fit_kind(P: np.ndarray, trim: float = 0.05, cross_validate: bool = True,
             seed: int = 20261023) -> tuple[str, dict]:
    """返回 (选中的种类, 各种类的残差)。

    `cross_validate=False` 复刻朴素做法（在同一批点上拟合又评估），留着做对照——
    两列一起报才看得出交叉验证到底有没有用。
    """
    Q = normalise(P, trim)
    if not cross_validate or len(Q) < 40:
        res = {k: f(Q)(Q) for k, f in FITTERS.items()}
        return min(res, key=res.get), res
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Q))
    half = len(Q) // 2
    train, test = Q[order[:half]], Q[order[half:]]
    res = {k: f(train)(test) for k, f in FITTERS.items()}
    return min(res, key=res.get), res


def vote(kinds: list[str]) -> str | None:
    """逐帧各拟合一次再投票。并云被离群点污染时，这一列才是能用的那个。"""
    return Counter(kinds).most_common(1)[0][0] if kinds else None


def gt_shapes(program: dict) -> dict[str, str]:
    """真值节点 id → 基本体种类。"""
    out = {}
    for node in (program.get("objects") or []) + (program.get("static") or []):
        geom = node.get("geom") or {}
        if geom.get("kind") == "primitive" and geom.get("shape") in KINDS:
            out[str(node.get("id"))] = geom["shape"]
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True,
                    help="<产出目录>:<真值目录>，可以给多次")
    ap.add_argument("--min-points", type=int, default=200)
    a = ap.parse_args(argv)

    truth: list[str] = []
    pred: list[str] = []
    pred_voted: list[str] = []
    pred_cv: list[str] = []
    skipped = Counter()
    for spec in a.pair:
        report, _s, gt_root = spec.partition(":")
        if not gt_root:
            ap.error(f"--pair 要写成 <report>:<gt_root>，收到 {spec!r}")
        for clip_dir in sorted(Path(report).glob("test_*")):
            per = clip_dir / "perception"
            gz, mz = per / "geometry.npz", per / "masks.npz"
            gt_prog = Path(gt_root) / clip_dir.name / "program.json"
            if not (gz.is_file() and mz.is_file() and gt_prog.is_file()):
                skipped["缺文件"] += 1; continue
            shapes = gt_shapes(json.loads(gt_prog.read_text(encoding="utf-8")))
            G, M = np.load(gz), np.load(mz)
            depth, K, c2w = G["depth"], G["intrinsics"], G["cam_to_world"]
            order = list(G["frame_indices"]) if "frame_indices" in G.files else list(range(len(depth)))
            H, W = depth[0].shape
            idx = json.loads((per / "masks_index.json").read_text(encoding="utf-8"))
            for oid, meta in (idx.get("objects") or {}).items():
                # 真值掩码的轨迹 id 形如 "<真值节点>" 或 "<真值节点>_<实例>"
                node = oid if oid in shapes else oid.rsplit("_", 1)[0]
                if node not in shapes:
                    skipped["对不上真值节点"] += 1; continue
                pts = []
                for fi in meta.get("frames", []):
                    key = f"{oid}__{fi}"
                    if key not in M.files:
                        continue
                    try:
                        k = order.index(fi)
                    except ValueError:
                        continue
                    P = to_world(unproject(depth[k], K[k], resize_mask(M[key], (W, H))), c2w[k])
                    if len(P):
                        pts.append(P)
                if not pts:
                    skipped["没有点"] += 1; continue
                P_all = np.concatenate(pts, 0)
                if len(P_all) < a.min_points:
                    skipped["点太少"] += 1; continue
                # 三列。前两列是朴素做法（同一批点上拟合又评估）的两种点云来源，
                # 第三列换成交叉验证。朴素那两列已经量过是失败的（输基线 26~31 个点，
                # cylinder 和 cone 零召回），留着是为了看交叉验证到底改善了多少——
                # 只报新方法不报它要赢的那个东西，是没法判断的。
                kind_naive, _res = fit_kind(P_all, cross_validate=False)
                per_frame_kinds = [fit_kind(p, cross_validate=False)[0]
                                   for p in pts if len(p) >= a.min_points // 4]
                kind_voted = vote(per_frame_kinds) or kind_naive
                kind_cv, _res_cv = fit_kind(P_all, cross_validate=True)
                truth.append(shapes[node])
                pred.append(kind_naive)
                pred_voted.append(kind_voted)
                pred_cv.append(kind_cv)

    print("=" * 72)
    if skipped:
        print("跳过的：" + "，".join(f"{k} {v}" for k, v in skipped.items()))
    n = len(truth)
    if not n:
        print("一个物体都没算成，下面没有数。"); return 1
    dist = Counter(truth)
    print(f"\n参与比较的物体 {n} 个，真值构成：{dict(dist)}")

    # —— 必须先摆基线：类别不平衡下，「恒答最多的那一类」能拿多少 ——
    major, major_n = dist.most_common(1)[0]
    base = major_n / n
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / n
    print(f"\n  恒答「{major}」的基线准确率   {base:.3f}")
    acc_voted = sum(1 for t, p in zip(truth, pred_voted) if t == p) / n
    acc_cv = sum(1 for t, p in zip(truth, pred_cv) if t == p) / n
    print(f"  并云·朴素残差的准确率      {acc:.3f}   "
          f"（{'赢' if acc > base else '没赢'}基线 {acc - base:+.3f}）")
    print(f"  逐帧·朴素残差投票的准确率  {acc_voted:.3f}   "
          f"（{'赢' if acc_voted > base else '没赢'}基线 {acc_voted - base:+.3f}）")
    print(f"  并云·**交叉验证**残差      {acc_cv:.3f}   "
          f"（{'赢' if acc_cv > base else '没赢'}基线 {acc_cv - base:+.3f}）")
    print("    前两列是朴素做法，留着做对照——只报新方法不报它要赢的东西，没法判断。")
    print("    并云在这条链路上被离群点污染过——98a48da 量到并云拟合的")
    print("    位置误差 0.642 米，比现算的首帧 0.420 还差——所以并云那一列明显更差的")
    print("    话，形状分类也不该用并云。")

    for label, P in (("并云·朴素", pred), ("逐帧·朴素投票", pred_voted), ("并云·交叉验证", pred_cv)):
        print(f"\n  逐类召回（{label}）——总准确率会被 box 主导，这张表才看得出有没有真的在分类：")
        for k in KINDS:
            tot = dist.get(k, 0)
            if not tot:
                continue
            hit = sum(1 for t, p in zip(truth, P) if t == k and p == k)
            print(f"    {k:9s} {hit:3d}/{tot:3d} = {hit / tot:.3f}")
        print(f"  混淆矩阵（{label}；行=真值，列=拟合给出的）：")
        print("            " + "".join(f"{k:>10s}" for k in KINDS))
        for t in KINDS:
            if not dist.get(t):
                continue
            row = [sum(1 for tt, pp in zip(truth, P) if tt == t and pp == p) for p in KINDS]
            print(f"    {t:9s} " + "".join(f"{v:10d}" for v in row))

    print("""
怎么读：
  只看总准确率会被 box 骗——它占了四分之三，恒答 box 就有七成五。
  真正要看的是**非 box 的那三类召回有没有明显大于零**：
  大于零  → 几何拟合确实在分类，稠密几何能监督这个离散选择，创新点一有东西；
  接近零  → 它只是学会了跟着多数类走，和恒答 box 没区别，这条要重新设计。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
