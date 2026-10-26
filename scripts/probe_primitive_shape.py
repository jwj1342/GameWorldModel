#!/usr/bin/env python
"""能不能从观测到的点云判出该用哪个基本体——球、盒、锥、柱。

为什么要问这个（#56）。真值场景**全部由基本体组成**：objects 是 28 box / 16 sphere
/ 9 cone / 6 cylinder，static 是 69 box。而我们产出的每一个物体节点都是
`kind: "asset"`——其中 **51.4% 连 `ref` 都没有**（资产库里没有 spool、totem、beacon
这些类），内核对它们走 `fallbackForClass`。

而 `fallbackForClass` 是**按类名的正则**挑形状的（`ball|sphere|orb|marble` → 球，
`coin|disc|ring|wheel` → 柱…），类名准确率只有 12.5%，并且真值用的那些类名一个都
不匹配那些正则。**结果是所有物体都渲成盒子。** 真值里盒子只占 28/59，所以
**超过一半的物体形状是错的**——这是看得见的精细度缺陷，而且一直没被量过
（形状一致度那个指标比的是**边长比例**，不是形状种类，所以它看不见这件事）。

这个脚本问的是：**不靠类名，能不能从点云判出形状种类。**

  球    —— 点到中心的距离几乎恒定，径向离散度很低
  盒    —— 角落比面心远，径向离散度明显更高
  锥/柱 —— 介于两者之间，而且沿主轴的截面半径变化不同（锥收敛、柱恒定）

先只量最基本的那个：**径向离散度能不能把球和盒分开。** 分不开就别往下做了。

用法:
  python -m scripts.probe_primitive_shape --report <产物目录> --eval <eval.json> \
      --gt-root <真值根>
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

from gwm.config import load_config                                    # noqa: E402
from gwm.perception.evidence import _fix, align_world, to_world, unproject  # noqa: E402
from gwm.perception.masks import resize_mask                          # noqa: E402
from gwm.perception.run import load_geometry, load_masks              # noqa: E402
from scripts.probe_cluster_quality import load_tracks                 # noqa: E402


def radial_spread(pts: np.ndarray) -> float | None:
    """点到质心距离的变异系数，先把点按包围盒各轴归一化成各向同性。

    不归一化的话，一个被拉长的球和一个方盒子会给出一样的数——我们要问的是形状，
    不是长宽比，长宽比已经由 extent 单独描述了。
    """
    if len(pts) < 40:
        return None
    c = pts.mean(0)
    q = pts - c
    scale = q.max(0) - q.min(0)
    if (scale <= 1e-6).any():
        return None
    q = q / scale                       # 归一化成单位立方体内，各向同性
    r = np.linalg.norm(q, axis=1)
    m = float(r.mean())
    return float(r.std() / m) if m > 1e-9 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config or [str(REPO / "configs" / "default.yaml")])

    by_shape: dict[str, list[float]] = {}
    for r in json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", []):
        scene = ((r.get("metrics") or {}).get("full") or {}).get("scene") or {}
        pairs = {q: g for g, q in (scene.get("pairs") or {}).items() if q}
        run = Path(a.report) / r["clip"]
        gtp = Path(a.gt_root) / r["clip"] / "program.json"
        pdir = run / "perception"
        if not (gtp.is_file() and (pdir / "geometry.npz").is_file()):
            continue
        gt = json.loads(gtp.read_text(encoding="utf-8"))
        shape_of = {n["id"]: ((n.get("geom") or {}).get("shape") or "?")
                    for n in (gt.get("objects") or [])}
        shape_of.update({f"static::{n['id']}": ((n.get("geom") or {}).get("shape") or "?")
                         for n in (gt.get("static") or [])})

        geom = load_geometry(pdir / "geometry.npz")
        tracks = load_tracks(pdir)
        T, sc, _ = align_world(geom, tracks, cfg)
        W, H = geom.size()
        masks = load_masks(pdir)
        slot = {int(fi): k for k, fi in enumerate(geom.frame_indices)}
        for oid, per_frame in masks.items():
            g = pairs.get(oid) or pairs.get(f"static::{oid}")
            if g is None or g not in shape_of:
                continue
            cloud = []
            for fi, m in sorted(per_frame.items())[:8]:
                k = slot.get(int(fi))
                if k is None:
                    continue
                pts = unproject(geom.depth[k], geom.intrinsics[k],
                                resize_mask(m, (W, H)), stride=2)
                if len(pts):
                    cloud.append((to_world(pts, _fix(geom.cam_to_world[k]))
                                  @ T[:3, :3].T + T[:3, 3]) * sc)
            if not cloud:
                continue
            v = radial_spread(np.vstack(cloud))
            if v is not None:
                by_shape.setdefault(shape_of[g], []).append(v)

    if not by_shape:
        print("没有配上真值的样本"); return 1

    print("径向离散度（点到质心距离的变异系数，先按包围盒归一化成各向同性）\n")
    print(f"{'真值形状':<10}{'n':>5}{'中位':>9}{'四分位':>18}")
    for sh, v in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
        A = np.sort(np.asarray(v))
        print(f"{sh:<10}{len(A):>5}{np.median(A):>9.4f}"
              f"{f'{np.percentile(A,25):.4f} / {np.percentile(A,75):.4f}':>18}")

    sph, box = by_shape.get("sphere"), by_shape.get("box")
    if sph and box:
        S, B = np.sort(np.asarray(sph)), np.sort(np.asarray(box))
        print(f"\n球 对 盒：中位 {np.median(S):.4f} 对 {np.median(B):.4f}")
        best = (None, -1.0)
        for thr in np.percentile(np.concatenate([S, B]), [10, 25, 50, 75, 90]):
            k, b = float((S <= thr).mean()), float((B > thr).mean())
            if k + b - 1 > best[1]:
                best = (float(thr), k + b - 1, k, b)
        print(f"  最好的阈值 {best[0]:.4f}：球判对 {best[2]:.3f}，盒判对 {best[3]:.3f}，"
              f"差 {best[1]:+.3f}")
        print("\n  **这个差是乐观偏置的**（五个阈值里取最大，样本又少），所以它只用来")
        print("  回答「有没有信号」。真要用，得在留出集上固定阈值再验一次——我今天已经")
        print("  被一个 0/7 的假信号骗过一次（见 #56）。")
    else:
        print("\n球或盒的样本不够，判别力没量到")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
