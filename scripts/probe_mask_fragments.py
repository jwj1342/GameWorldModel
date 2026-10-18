#!/usr/bin/env python
"""被切碎的结构件，它的那几块掩码在**图像平面上**是连着的还是断开的。

这是碎片这条线的收口（#56）。前面量清楚的事实：

  真值结构件里，位置配上的那 16 个，水平最长边只有真值的 0.70，下四分位 **0.19**
  ——四分之一的结构件被重建成不到真实长度的五分之一。那就是碎片。

但「碎片」有两种成因，该做的事完全不同：

  **分割切碎**：图像上本来是一整片连通区域，分割后端把它切成了几块。
                 那就该在分割那一层合回去，信息没丢。
  **遮挡断开**：中间被前景物体挡住了，图像上本来就不连通。
                 那是真的断开，合回去需要别的依据（比如共面、同材质）。

两者在图像上的区别很直接：**两块掩码之间的那条缝里有没有别的东西。**

## 怎么量

1. 用评测的规范对齐，把预测节点按「最近的真值结构件」分组（容差放宽到 1 米——
   碎片本来就偏，用 0.45 会把要研究的对象筛掉）。只看成员 ≥ 2 的组。
2. 同一帧里取组内两块掩码，算它们之间的**最小像素缝宽**（距离变换）。
3. 取缝里的那条走廊（到 A 的距离加到 B 的距离 ≈ 缝宽的那些像素），看它被
   **别的**检出的掩码盖住了多少。

缝宽接近零 = 本来就挨着，分割硬切开的。
缝宽不小但走廊被别的掩码盖满 = 遮挡。
缝宽不小而走廊是空的 = 两块之间什么都没有，那既不是切碎也不是遮挡，
  是检测漏掉了中间那一段。

用法:
  python -m scripts.probe_mask_fragments --report <产物目录> \
      --eval <scene_eval.json> --gt-root <真值根>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.feedback.gt_metrics import _rot_y          # noqa: E402
from gwm.perception.masks import resize_mask        # noqa: E402
from gwm.perception.run import load_geometry, load_masks   # noqa: E402

TOUCHING_PX = 2.0          # 缝宽小于这个就算「本来挨着」
COVERED_FRAC = 0.5         # 走廊被别的掩码盖住这么多就算遮挡


def nodes_with_pos(program: dict) -> dict[str, np.ndarray]:
    out = {}
    for n in (program.get("objects") or []) + (program.get("static") or []):
        pos = (n.get("pose") or {}).get("pos")
        if isinstance(pos, list) and len(pos) == 3:
            out[str(n.get("id"))] = np.asarray(pos, float)
    return out


def gt_static_pos(program: dict) -> dict[str, np.ndarray]:
    out = {}
    for n in (program.get("static") or []):
        pos = (n.get("pose") or {}).get("pos")
        if isinstance(pos, list) and len(pos) == 3:
            out[str(n.get("id"))] = np.asarray(pos, float)
    return out


def classify(a: np.ndarray, b: np.ndarray, others: list[np.ndarray]) -> tuple[float, float]:
    """返回 (最小缝宽像素, 走廊被别的掩码盖住的比例)。"""
    da = distance_transform_edt(~a)
    db = distance_transform_edt(~b)
    gap = float(da[b].min()) if b.any() else float("inf")
    if not np.isfinite(gap):
        return gap, 0.0
    corridor = (da + db) <= (gap + 1.5)
    corridor &= ~a & ~b
    if not corridor.any():
        return gap, 0.0
    covered = np.zeros_like(corridor)
    for o in others:
        covered |= (o & corridor)
    return gap, float(covered.sum()) / float(corridor.sum())


def boundary_depth_jump(a: np.ndarray, b: np.ndarray, depth: np.ndarray) -> float | None:
    """两块挨着的掩码，在它们共同的边界上深度跳多少（相对深度，取中位数之比）。

    这是「该不该合」的判据候选。一块被硬切开的表面，边界两侧深度是连续的；
    两个真的挨在一起的物体（球搁在轨道上），边界处有深度断层。

    返回 |深度差| / 该处深度，0 附近是连续，越大越像断层。用相对值是因为深度是
    相对尺度的，绝对差会被远近主导。
    """
    if depth.shape != a.shape:
        return None
    da = distance_transform_edt(~a)
    db = distance_transform_edt(~b)
    near = (da <= 3) & (db <= 3) & (depth > 1e-4)
    if near.sum() < 20:
        return None
    side_a = near & a
    side_b = near & b
    if side_a.sum() < 8 or side_b.sum() < 8:
        return None
    ma, mb = float(np.median(depth[side_a])), float(np.median(depth[side_b]))
    base = max(min(ma, mb), 1e-6)
    return abs(ma - mb) / base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--group-tol", type=float, default=1.0,
                    help="按最近的真值结构件分组时的容差（米）。放宽到 1 米是故意的："
                         "碎片本来就偏，用评测的 0.45 会把要研究的对象筛掉。")
    ap.add_argument("--max-pairs", type=int, default=400)
    a = ap.parse_args(argv)

    gauge = {}
    for r in json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", []):
        g = ((r.get("metrics") or {}).get("full") or {}).get("gauge_alignment") or {}
        if "yaw_deg" in g:
            gauge[r.get("clip")] = g

    kinds = {"分割硬切开": 0, "遮挡": 0, "中间整段漏掉": 0}
    gaps, groups_seen, pairs_seen = [], 0, 0
    per_clip = []
    # 判据可分性：挨着的掩码对里，**同一个**真值节点的（该合）和**不同**真值节点的
    # （不该合）各自的边界深度跳。分不开就说明这个判据不成立。
    jump_same: list[float] = []
    jump_diff: list[float] = []
    for run in sorted(Path(a.report).iterdir()):
        clip = run.name
        pdp, gtp = run / "program.json", Path(a.gt_root) / clip / "program.json"
        pdir = run / "perception"
        if not (pdp.is_file() and gtp.is_file() and (pdir / "masks.npz").is_file()
                and clip in gauge):
            continue
        g = gauge[clip]
        R = _rot_y(float(g["yaw_deg"]))
        t = np.asarray([g["translation_xz_m"][0], 0.0, g["translation_xz_m"][1]], float)
        scale = float(g.get("scale") or 1.0)

        pred = {k: scale * (R @ v) + t for k, v in nodes_with_pos(json.loads(pdp.read_text(encoding="utf-8"))).items()}
        gt = gt_static_pos(json.loads(gtp.read_text(encoding="utf-8")))
        if not pred or not gt:
            continue
        grouped: dict[str, list[str]] = {}
        for pid, p in pred.items():
            best, bd = None, float("inf")
            for gid, q in gt.items():
                d = float(np.linalg.norm(p - q))
                if d < bd:
                    best, bd = gid, d
            if best is not None and bd <= a.group_tol:
                grouped.setdefault(best, []).append(pid)
        groups = {k: v for k, v in grouped.items() if len(v) >= 2}
        owner = {pid: gid for gid, members in grouped.items() for pid in members}
        if not groups:
            continue
        geom = load_geometry(pdir / "geometry.npz")
        slot = {int(fi): k for k, fi in enumerate(geom.frame_indices)}

        mi = json.loads((pdir / "masks_index.json").read_text(encoding="utf-8"))
        size = tuple(mi["frame_size"])
        masks = load_masks(pdir)
        clip_pairs = 0
        for gid, members in groups.items():
            groups_seen += 1
            frames = [set(masks.get(m, {}).keys()) for m in members]
            common = set.intersection(*frames) if frames else set()
            for fi in sorted(common)[:3]:                 # 每组最多看三帧就够看出性质
                full = {m: resize_mask(masks[m][fi], size).astype(bool) for m in members
                        if fi in masks.get(m, {})}
                others = [resize_mask(v[fi], size).astype(bool)
                          for k, v in masks.items() if fi in v and k not in full]
                keys = sorted(full)
                # **深度图和掩码不是一个分辨率**（实测 294x518 对 360x640），
                # 所以深度那一档要按深度图的尺寸单独缩一次掩码。第一版拿掩码的
                # 尺寸去对，形状永远对不上，那一档一个样本都没采到。
                dep = geom.depth[slot[fi]] if fi in slot else None
                dsize = (dep.shape[1], dep.shape[0]) if dep is not None else None
                full_d = ({m: resize_mask(masks[m][fi], dsize).astype(bool) for m in full}
                          if dsize else {})
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        if pairs_seen >= a.max_pairs:
                            break
                        gap, cov = classify(full[keys[i]], full[keys[j]], others)
                        if not np.isfinite(gap):
                            continue
                        gaps.append(gap); pairs_seen += 1; clip_pairs += 1
                        if gap <= TOUCHING_PX:
                            kinds["分割硬切开"] += 1
                            if dep is not None and keys[i] in full_d and keys[j] in full_d:
                                jp = boundary_depth_jump(full_d[keys[i]], full_d[keys[j]], dep)
                                if jp is not None:
                                    jump_same.append(jp)
                        elif cov >= COVERED_FRAC:
                            kinds["遮挡"] += 1
                        else:
                            kinds["中间整段漏掉"] += 1
        # 反例：挨着、但归属**不同**真值节点的掩码对——这些是不该合的。
        for fi in sorted(slot)[:3]:
            dep = geom.depth[slot[fi]]
            dsize = (dep.shape[1], dep.shape[0])
            avail = {k: resize_mask(v[fi], dsize).astype(bool)
                     for k, v in masks.items() if fi in v and k in owner}
            ks = sorted(avail)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    if owner[ks[i]] == owner[ks[j]] or len(jump_diff) >= a.max_pairs:
                        continue
                    da = distance_transform_edt(~avail[ks[i]])
                    if not avail[ks[j]].any() or float(da[avail[ks[j]]].min()) > TOUCHING_PX:
                        continue
                    jp = boundary_depth_jump(avail[ks[i]], avail[ks[j]], dep)
                    if jp is not None:
                        jump_diff.append(jp)

        if clip_pairs:
            per_clip.append((clip, len(groups), clip_pairs))

    if not pairs_seen:
        print("没有找到成员 ≥ 2 的碎片组，这个结论不成立"); return 1

    print(f"分组容差 {a.group_tol} m，碎片组 {groups_seen} 个，掩码对 {pairs_seen} 对\n")
    print(f"{'片段':<14}{'碎片组':>8}{'掩码对':>8}")
    for clip, ng, npair in per_clip:
        print(f"{clip:<14}{ng:>8}{npair:>8}")
    arr = np.sort(np.asarray(gaps, float))
    print(f"\n两块掩码之间的最小缝宽（像素，画面 {size[0]}x{size[1]}）：")
    print(f"  中位 {np.median(arr):.1f}，四分位 {np.percentile(arr,25):.1f} / "
          f"{np.percentile(arr,75):.1f}，最大 {arr.max():.1f}")
    print(f"  缝宽 ≤ {TOUCHING_PX:.0f} 像素的占 {(arr <= TOUCHING_PX).mean():.3f}")
    print("\n成因构成：")
    for k, v in kinds.items():
        print(f"  {k:<12} {v:>5}  {v / pairs_seen:.3f}")
    print(f"\n判据：缝宽 ≤ {TOUCHING_PX:.0f} 像素 = 本来挨着，分割硬切开的；")
    print(f"      缝更宽但走廊被别的掩码盖住 ≥ {COVERED_FRAC:.0%} = 遮挡；")
    print("      缝更宽而走廊是空的 = 中间那一段根本没被检出。")
    if jump_same and jump_diff:
        sa = np.sort(np.asarray(jump_same)); df = np.sort(np.asarray(jump_diff))
        print("\n—— 「该不该合」的判据：挨着的两块，边界上深度跳多少 ——")
        print(f"  同一个真值节点（该合）  n={len(sa):<4} 中位 {np.median(sa):.4f}  "
              f"四分位 {np.percentile(sa,25):.4f} / {np.percentile(sa,75):.4f}")
        print(f"  不同真值节点（不该合）  n={len(df):<4} 中位 {np.median(df):.4f}  "
              f"四分位 {np.percentile(df,25):.4f} / {np.percentile(df,75):.4f}")
        thr = float(np.percentile(sa, 75))
        tp = float((sa <= thr).mean()); fp = float((df <= thr).mean())
        print(f"  拿「该合」那组的 75 分位 {thr:.4f} 当阈值：该合的留下 {tp:.3f}，"
              f"不该合的误合 {fp:.3f}")
        print("  两组中位数差不开、或者误合率和留存率差不多，这个判据就不成立——")
        print("  那就别在掩码层合，碎片连链那次的教训是判据不成立时调阈值没有用。")
    else:
        print("\n（深度跳那一档样本不够，判据可分性没量到）")

    print("\n三种成因该做的事完全不同：第一种在分割那一层合回去就行，信息没丢；")
    print("第二种要靠共面、同材质之类的依据跨过遮挡；第三种是检测召回的问题，")
    print("合并帮不上忙。所以先量这个比例，再决定在哪一层动手。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
