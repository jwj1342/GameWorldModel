"""把每个物体的点**跨帧汇到一起**再拟合包围盒，能买到多少——「教师」最便宜的那一版。

背景。几何后端已经是 VGGT，它本来就是多帧联合预测相机和深度的**全局**重建。
但 `evidence.py` 把这份全局几何扔掉了：`obb_from_points` 每帧只拿那一帧的
**可见点**算一个包围盒，算完就扔。`docs/design-notes.md` 里写得很清楚——

> 可见点集的任何逐帧汇总量都给不出干净的轨迹。包围盒中心是可见点在各轴上
> 3/97 百分位的中点，可见面一变两端就变。

掩码完全准确时位置误差中位数仍有 0.244 米，三条廉价修法（沿视线推、刚性偏移、
换中位数/均值）全败——但那三条**全都还是逐帧统计量**。跨帧聚合只在
`registration.py` 里以「序贯 ICP 累积」的形式试过，拿到 26%。

**真正平凡的那一版从来没试过：静止物体直接在世界坐标里把各帧的点并起来。**
不需要配准、不需要解位姿，多视角自然看全整个物体。而真值里 static 有 69 个、
object 只有 59 个——一半以上的场景属于这一档。

这个脚本不动管线，只拿存下来的 `geometry.npz` 和 `masks.npz` 离线重算，比两件事：

  逐帧   复刻现在的做法（**必须和 evidence.json 里存的对得上，这是自查**）
  汇聚   把该物体所有帧的点并起来，再拟合一个包围盒

比的是**尺寸**不是位置：位置要和真值配对才有意义，而尺寸有一个不需要配对的判据——
逐帧只看得见一个面，所以尺寸会被系统性地低估；汇聚之后该变大。长维度被截断到
真值的 0.70（下四分位 0.19）这件事已经量过，所以「变大多少」直接对应能补回多少。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.perception.evidence import obb_from_points        # noqa: E402
from gwm.perception.masks import resize_mask               # noqa: E402


def unproject(depth, K, mask, stride=2):
    H, W = depth.shape
    ys, xs = np.where(mask[::stride, ::stride])
    ys, xs = ys * stride, xs * stride
    ok = depth[ys, xs] > 1e-4
    ys, xs = ys[ok], xs[ok]
    if not len(ys):
        return np.zeros((0, 3))
    z = depth[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def to_world(P, c2w):
    if not len(P):
        return P
    return P @ c2w[:3, :3].T + c2w[:3, 3]


CV2THREE = np.diag([1.0, -1.0, -1.0])


def fix(c2w):
    F = np.eye(4); F[:3, :3] = CV2THREE
    return F @ c2w


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", action="append", required=True)
    ap.add_argument("--still-frac", type=float, default=0.5,
                    help="逐帧中心的铺开范围小于物体对角线这个倍数的，当作静止。"
                         "无量纲，别换成米。")
    a = ap.parse_args(argv)

    grew: list[float] = []
    missing_keys: list[str] = []
    sample_keys: list[str] = []
    mismatch_examples: list[str] = []
    long_grew: list[float] = []
    n_obj = n_still = n_clip = 0
    selfcheck_ok = selfcheck_bad = 0
    for d in a.report_dir:
        root = Path(d)
        print(f"\n########## {root.name} ##########")
        for clip_dir in sorted(root.glob("test_*")):
            per = clip_dir / "perception"
            gz, mz = per / "geometry.npz", per / "masks.npz"
            if not (gz.is_file() and mz.is_file() and (per / "evidence.json").is_file()):
                continue
            G = np.load(gz); M = np.load(mz)
            if n_clip == 0:
                print(f"  geometry.npz 的键：{sorted(G.files)}")
                print(f"  masks.npz 的键（前 3 个）：{sorted(M.files)[:3]}")
            if not sample_keys:
                sample_keys = sorted(M.files)[:3]
            n_clip += 1
            try:
                depth, K, c2w = G["depth"], G["intrinsics"], G["cam_to_world"]
            except KeyError as exc:
                print(f"  {clip_dir.name}: geometry.npz 少了键 {exc}，跳过"); continue
            idx = json.loads((per / "masks_index.json").read_text(encoding="utf-8"))
            ev = json.loads((per / "evidence.json").read_text(encoding="utf-8"))
            stored = {o["id"]: o for o in ev.get("objects", [])}
            scale_factor = float(((ev.get("meta") or {}).get("alignment") or {}).get("scale_factor") or 1.0)
            frame_order = list(G["frame_indices"]) if "frame_indices" in G.files else list(range(len(depth)))
            H, W = depth[0].shape
            for oid, meta in (idx.get("objects") or {}).items():
                pooled_pts, per_frame = [], []
                for fi in meta.get("frames", []):
                    # masks.npz 的键是 "<物体>__<帧号>"（双下划线）。第一版猜成了
                    # "<物体>/<帧号>"，于是一个都匹配不上——而且是**静默跳过**，
                    # 跑出来「物体 0 个」看起来像「没有可比的物体」，不像「键写错了」。
                    # 所以这里匹配不上要记下来，最后出声。
                    key = f"{oid}__{fi}"
                    if key not in M.files:
                        missing_keys.append(key)
                        continue
                    try:
                        k = frame_order.index(fi)
                    except ValueError:
                        continue
                    m = resize_mask(M[key], (W, H))
                    P = to_world(unproject(depth[k], K[k], m), fix(c2w[k]))
                    if len(P) < 20:
                        continue
                    ob = obb_from_points(P)
                    if ob is None:
                        continue
                    per_frame.append(ob); pooled_pts.append(P)
                if len(per_frame) < 3:
                    continue
                n_obj += 1
                # 自查：复算的逐帧包围盒必须和 evidence.json 里存的对得上。
                #
                # **要先把缩放折回去。** 管线是在世界对齐（T：把地面法向转到 y 轴
                # 的旋转 + 一个 y 平移）并乘上 scale_factor 之后才拟合包围盒的，
                # 而这里离线只做了相机到世界。第一版漏了这一步，53 个对上、219 个
                # 对不上——自查挡住了，没出数。（同一个坑这个项目里踩过一次：
                # 早先一个探针漏了 align_world 的平移，把 119/128 读成 0.93。）
                #
                # 折回去只需要 scale_factor（evidence.json 里有）；T 的旋转部分
                # 没存（候选的法向被剥掉了），但这批片段的地面法向离 y 轴都在 7 度
                # 以内，对**边长**的影响进不了 25% 的容差。
                #
                # 下面真正要报的量是**比值**（汇聚 ÷ 逐帧），相似变换对比值没有影响，
                # 所以结论本身不依赖这一步——这一步只是为了让自查能做。
                st = stored.get(oid)
                if st and st.get("obb"):
                    mine = np.asarray(per_frame[0]["size"]) * scale_factor
                    theirs = np.asarray(st["obb"][0]["size"])
                    if np.allclose(mine, theirs, rtol=0.25, atol=0.02):
                        selfcheck_ok += 1
                    else:
                        selfcheck_bad += 1
                        if len(mismatch_examples) < 3:
                            mismatch_examples.append(
                                f"{clip_dir.name}/{oid} 复算×{scale_factor:.2f}="
                                f"{np.round(mine, 3).tolist()} 存的={np.round(theirs, 3).tolist()}")
                centres = np.asarray([o["center"] for o in per_frame])
                diag = float(np.median([np.linalg.norm(o["size"]) for o in per_frame]))
                spread = float(np.linalg.norm(centres - centres[0], axis=1).max())
                if diag <= 1e-9 or spread > a.still_frac * diag:
                    continue                       # 会动的物体世界坐标汇聚会抹开，不算
                n_still += 1
                pooled = obb_from_points(np.concatenate(pooled_pts, 0))
                if pooled is None:
                    continue
                med = np.median(np.asarray([o["size"] for o in per_frame]), axis=0)
                pl = np.asarray(pooled["size"])
                if med.min() > 1e-9:
                    grew.append(float(np.prod(pl) / np.prod(med)) ** (1 / 3))
                    long_grew.append(float(pl.max() / med.max()))
    print("\n" + "=" * 70)
    if missing_keys:
        print(f"**masks.npz 里找不到 {len(missing_keys)} 个键**，例如 {missing_keys[:3]}，"
              f"而实际存在的形如 {sample_keys[:3]}。")
        print("  键的格式对不上，下面的数不可信——这种事会静默跳过，所以必须显式报出来。")
    print(f"自查：复算的逐帧包围盒和 evidence.json 对得上 {selfcheck_ok}，对不上 {selfcheck_bad}")
    if selfcheck_bad > selfcheck_ok:
        print("  **对不上的比对得上的多，下面的数一个都不能信**——复算没复现现有做法。")
        for e in mismatch_examples:
            print(f"    {e}")
        return 1
    print(f"物体 {n_obj} 个，其中判为静止、参与比较的 {n_still} 个")
    if grew:
        print(f"\n汇聚之后尺寸涨了多少（体积的立方根之比）：中位 {statistics.median(grew):.3f}，"
              f"四分位 {np.percentile(grew, 25):.3f} / {np.percentile(grew, 75):.3f}")
        print(f"最长边之比：中位 {statistics.median(long_grew):.3f}，"
              f"四分位 {np.percentile(long_grew, 25):.3f} / {np.percentile(long_grew, 75):.3f}")
        print("""
怎么读：
  明显大于 1  → 逐帧确实只看见一个面，汇聚补回了被截断的部分，**教师这条路有东西**；
  接近 1      → 跨帧并没有带来更多覆盖，便宜版本没用，得上真正的稠密重建；
  小于 1      → 汇聚把噪声也并进来了，反而更糟，这条要重新设计。
提醒：长维度被截断到真值的 0.70（下四分位 0.19）这件事已经量过，
      所以「最长边之比」直接对应能补回多少——1.43 左右就正好补平。""")
    else:
        print("没有可比的物体")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
