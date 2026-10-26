#!/usr/bin/env python
"""不靠类名、不靠形状，靠「它到底动没动」来判结构件——这个信号到底有没有。

为什么走到这一步（#56）。固定结构和可动物体分不开，是现在精细度上最具体的一个
缺口：位置配上的 16 个真值结构件，我们只认出 1 个，另外 15 个全写进了 objects。
两条试过的路都断了：

  **类名**  判据要求「类名像结构件」，而类名准确率只有 12.5%，规则压根不触发。
  **形状**  真值上扁平度分得很开（static 0.156 / object 0.591），但在我们自己
            拟合的包围盒上两组之差从 0.435 塌到 0.035。

还剩一个没量过的：**这东西在整段视频里有没有动过。** 固定结构按定义不动。

要诚实地说在前面：#7 已经证明这条链路的运动**分类**（七类）贡献精确为零，
二分「会不会动」也没有信息量。但那问的是「它怎么动」和「哪些物体在动」；
这里问的是另一件事——**把节点分进 static 还是 objects 这两个列表**，标签是真值的
节点类别，不是运动类型。信号源也不同：这里直接用逐帧包围盒中心的离散程度，
不经过运动拟合、不经过周期性闸、不经过 ICP。那几步正是 #7 里查出问题的地方。

所以这不是重跑一个已知为零的实验，是换了问题、换了信号源。**但如果它也分不开，
那就是三条路全断，结构件判据这件事得换一个层面想。**

## 量什么

对每个预测节点，从 `evidence.json` 的逐帧 `obb.center` 算：

  抖动 = 中心点相对自身中位数的均方半径 / 节点自身的对角线长

除以自身尺寸是必须的：绝对位移不是尺度不变的，大物体的包围盒噪声本来就大，
#56 里栽过一次绝对米制阈值的跟头。

标签用评测的场景配对：这个节点配上的真值是 static 还是 object。

用法:
  python -m scripts.probe_static_by_stillness --report <产物目录> --eval <scene_eval.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def jitter(entry: dict) -> float | None:
    """逐帧中心的均方半径，按节点自身对角线归一。"""
    obb = [o for o in (entry.get("obb") or [])
           if isinstance(o.get("center"), list) and len(o["center"]) == 3]
    if len(obb) < 4:
        return None
    C = np.asarray([o["center"] for o in obb], float)
    S = np.asarray([o["size"] for o in obb if isinstance(o.get("size"), list)], float)
    if not np.isfinite(C).all() or not len(S):
        return None
    diag = float(np.linalg.norm(np.median(S, axis=0)))
    if diag <= 1e-6:
        return None
    spread = float(np.sqrt(((C - np.median(C, 0)) ** 2).sum(1).mean()))
    return spread / diag


def centres(entry: dict) -> np.ndarray | None:
    obb = [o for o in (entry.get("obb") or [])
           if isinstance(o.get("center"), list) and len(o["center"]) == 3]
    if len(obb) < 2:
        return None
    C = np.asarray([o["center"] for o in obb], float)
    return C if np.isfinite(C).all() else None


def describe(name: str, xs: list[float]) -> str:
    if not xs:
        return f"  {name:<22} （没有样本）"
    a = np.sort(np.asarray(xs, float))
    return (f"  {name:<22} n={len(a):<4} 中位 {np.median(a):.4f}  "
            f"四分位 {np.percentile(a, 25):.4f} / {np.percentile(a, 75):.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gt-root", default=None,
                    help="给了就顺便量第二件事：位置取首帧还是取逐帧中位，哪个离真值更近")
    a = ap.parse_args(argv)

    rows = json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", [])
    still_s: list[float] = []      # 配上真值 static 的
    still_o: list[float] = []      # 配上真值 object 的
    unmatched: list[float] = []

    for r in rows:
        scene = ((r.get("metrics") or {}).get("full") or {}).get("scene") or {}
        pairs = {q: g for g, q in (scene.get("pairs") or {}).items() if q}
        ev = Path(a.report) / r["clip"] / "perception" / "evidence.json"
        if not ev.is_file():
            continue
        for entry in (json.loads(ev.read_text(encoding="utf-8")).get("objects") or []):
            j = jitter(entry)
            if j is None:
                continue
            gt_key = pairs.get(str(entry.get("id")))
            if gt_key is None:
                unmatched.append(j)
            elif gt_key.startswith("static::"):
                still_s.append(j)
            else:
                still_o.append(j)

    # 第二件事：抖动这么大，那**位置取首帧**就是个坏估计。
    # direct.py 里结构件那一支已经用逐帧中位了，可动物体那一支还在用 obb[0]。
    # 这里不改代码，先量：对我们自己写成静止的节点（它们的 pose.pos 就是最终位置），
    # 首帧中心和逐帧中位中心，哪个离配上的真值更近。
    # **必须按真值类别分开看。** 第一版把两类混在一起，得出「首帧更好」的结论，
    # 那是个混淆的比较：混进来的可动物体在真值里是会动的，它们的 pose.pos 是**起点**,
    # 首帧估计按构造就赢。只有配上真值 static 的那些才是干净的对照——真值结构件
    # 位置恒定，首帧和中位比的是同一个靶子。
    EST = ("首帧", "逐帧中位", "best_frame", "最高 conf 帧")
    tally = {"真值结构件": [0, 0], "真值可动物体": [0, 0]}
    dist: dict[str, dict[str, list[float]]] = {
        k: {e: [] for e in EST} for k in ("真值结构件", "真值可动物体")}
    drift: list[tuple[float, float]] = []      # (帧序归一化, 该帧中心离真值多远)
    if a.gt_root:
        for r in rows:
            full = ((r.get("metrics") or {}).get("full") or {})
            scene = full.get("scene") or {}
            g = full.get("gauge_alignment") or {}
            pairs = {q: gk for gk, q in (scene.get("pairs") or {}).items() if q}
            ev = Path(a.report) / r["clip"] / "perception" / "evidence.json"
            pdp = Path(a.report) / r["clip"] / "program.json"
            gtp = Path(a.gt_root) / r["clip"] / "program.json"
            if not (ev.is_file() and gtp.is_file() and pdp.is_file() and "yaw_deg" in g):
                continue
            yaw = np.radians(float(g["yaw_deg"])); c, sn = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, 0, sn], [0, 1.0, 0], [-sn, 0, c]])
            t = np.asarray([g["translation_xz_m"][0], 0.0, g["translation_xz_m"][1]], float)
            sc = float(g.get("scale") or 1.0)
            gt = json.loads(gtp.read_text(encoding="utf-8"))
            gt_pos = {f"static::{n['id']}": n["pose"]["pos"] for n in (gt.get("static") or [])
                      if isinstance((n.get("pose") or {}).get("pos"), list)}
            gt_pos.update({n["id"]: n["pose"]["pos"] for n in (gt.get("objects") or [])
                           if isinstance((n.get("pose") or {}).get("pos"), list)})
            pd = json.loads(pdp.read_text(encoding="utf-8"))
            # 只看我们自己写成静止的节点：它们的 pose.pos 就是最终位置，
            # 会动的节点位置由轨迹决定，换估计量是另一件事。
            still_ids = {n["id"] for n in (pd.get("objects") or [])
                         if (n.get("motion") or {}).get("type", "static") == "static"}
            still_ids |= {n["id"] for n in (pd.get("static") or [])}
            for entry in (json.loads(ev.read_text(encoding="utf-8")).get("objects") or []):
                nid = str(entry.get("id"))
                gk = pairs.get(nid)
                if nid not in still_ids or gk not in gt_pos:
                    continue
                C = centres(entry)
                if C is None:
                    continue
                target = np.asarray(gt_pos[gk], float)
                kind = "真值结构件" if gk.startswith("static::") else "真值可动物体"
                obb = [o for o in (entry.get("obb") or [])
                       if isinstance(o.get("center"), list) and len(o["center"]) == 3]

                def err(vec) -> float:
                    return float(np.linalg.norm(sc * (R @ np.asarray(vec, float)) + t - target))

                bf = entry.get("best_frame")
                at_best = next((o["center"] for o in obb if o.get("frame") == bf), C[0])
                confs = [o.get("conf") for o in obb]
                if any(isinstance(c, (int, float)) for c in confs):
                    at_conf = obb[int(np.argmax([c if isinstance(c, (int, float)) else -1
                                                 for c in confs]))]["center"]
                else:
                    at_conf = C[0]
                vals = {"首帧": err(C[0]), "逐帧中位": err(np.median(C, 0)),
                        "best_frame": err(at_best), "最高 conf 帧": err(at_conf)}
                for e, v in vals.items():
                    dist[kind][e].append(v)
                tally[kind][0 if vals["逐帧中位"] < vals["首帧"] - 1e-9 else 1] += 1
                # 漂移：真值结构件位置恒定，所以逐帧误差随帧序怎么变是干净的
                if kind == "真值结构件" and len(C) >= 4:
                    for k, cc in enumerate(C):
                        drift.append((k / (len(C) - 1), err(cc)))

    if not (still_s and still_o):
        print("两组里有一组没有样本，结论不成立"); return 1

    print("逐帧包围盒中心的抖动（按节点自身对角线归一）\n")
    print(describe("配上真值结构件", still_s))
    print(describe("配上真值可动物体", still_o))
    print(describe("没配上", unmatched))

    s, o = np.sort(np.asarray(still_s)), np.sort(np.asarray(still_o))
    print(f"\n真值结构件该更小（它按定义不动）。两组中位数：{np.median(s):.4f} "
          f"对 {np.median(o):.4f}")
    print("\n各个阈值下，把「抖动小于阈值」判成结构件的效果：")
    print(f"{'阈值':>8}{'结构件查全':>12}{'可动物体误判':>14}{'差':>10}")
    best = (None, -1.0)
    for thr in np.percentile(np.concatenate([s, o]), [10, 25, 40, 50, 60, 75, 90]):
        tp = float((s <= thr).mean()); fp = float((o <= thr).mean())
        if tp - fp > best[1]:
            best = (float(thr), tp - fp)
        print(f"{thr:>8.4f}{tp:>12.3f}{fp:>14.3f}{tp - fp:>+10.3f}")
    print(f"\n最好的那个阈值 {best[0]:.4f}，查全减误判 = {best[1]:+.3f}。")
    print("这个差接近零就是没有信号——那就是类名、形状、静止三条路全断，")
    print("结构件判据不是在感知这一层能解决的，得换层面想。")
    if any(dist[k]["首帧"] for k in dist):
        print("\n—— 顺便量的第二件事：位置取首帧还是取逐帧中位 ——")
        print("  **按真值类别分开看。** 混在一起比是错的：可动物体在真值里会动，")
        print("  它的 pose.pos 是起点，首帧估计按构造就赢，把结论带偏。")
        for kind in ("真值结构件", "真值可动物体"):
            if not dist[kind]["首帧"]:
                print(f"\n  {kind}：没有样本"); continue
            print(f"\n  {kind}（n={len(dist[kind]['首帧'])}）")
            for e in EST:
                v = np.asarray(dist[kind][e])
                print(f"    {e:<12} 离真值 中位 {np.median(v):.3f} m，"
                      f"容差 0.45 内 {(v <= 0.45).mean():.3f}")
            print(f"    逐帧中位比首帧更近的 {tally[kind][0]} 个，不更近 {tally[kind][1]} 个")
        if drift:
            d = np.asarray(drift)
            print("\n  逐帧误差随帧序怎么走（只用真值结构件，位置恒定所以干净）：")
            for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
                m = (d[:, 0] >= lo) & (d[:, 0] < hi)
                if m.any():
                    print(f"    前 {lo:.0%}–{hi:.0%} 段  n={int(m.sum()):<5} "
                          f"误差中位 {np.median(d[m, 1]):.3f} m")
            print("    误差随帧序单调变大 = 包围盒在**漂移**，不是在真值附近抖。")
            print("    那样取中位就是把估计往漂移里拖，首帧反而最接近。")
        print("\n  只有「真值结构件」那一组是干净的对照：真值结构件位置恒定，")
        print("  两种估计比的是同一个靶子。可动物体那一组只能当参考。")

    print("\n口径提醒：这不是重跑 #7 那个已知为零的实验。#7 问的是「它怎么动」和")
    print("「哪些物体在动」，走的是运动拟合、周期性闸、ICP 那一串；这里问的是")
    print("「分进哪个列表」，信号直接取逐帧包围盒中心，不经过那几步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
