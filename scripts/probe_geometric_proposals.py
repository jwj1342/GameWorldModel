#!/usr/bin/env python
"""不用文字检测器，直接从几何里提物体候选，看能不能比现在好。

为什么走到这一步（#56）：事后过滤和合并这两类路全部排除了，一共十八条。

  按检测**自身的属性**过滤——置信度（全局和同短语内）、被看到的帧比例、掩码占画面、
    资产检索成功率、三维尺寸变异系数、逐帧中心跳变、图像检索分、top-k 截断、
    地面接触、扁平度、悬空高度……失败机制是一致的：**这些信号都只描述检测自身**，
    而真假之别在于它对不对应一个真实物体。
  按检出**之间的空间关系**合并——同类去重（#72）、跨类重叠。失败是因为那 74% 的
    假阳性**根本不挤在一起**，是散落在场景各处的独立幻觉。

而且短语无关性那个实验说明文字这一路本来就没起作用：喂真值类名和喂一组完全不相干
的通用词表，每条片段的检出数（12.1 对 11.8）、精确率（0.226 对 0.218）几乎一样，
两轮位置重合只有 3.0%。

所以候选不该从文字检测器来。这个脚本量另一条：**去掉地面，对剩下的点云做空间聚类，
每个簇当一个物体候选。** 用的是链路里最可靠的那一路（位置和尺寸的信噪比 18 倍），
而且不需要任何文字提示。

量三件事：
  簇数 vs 真值物体数     现在是 12 比 5，几何聚类能不能接近
  簇落在真值物体上的比例  等价于精确率
  真值物体被簇覆盖的比例  等价于召回

用法:
  python -m scripts.probe_geometric_proposals --report <产物目录> --gt-root <真值根>
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

from gwm.perception.evidence import _fix, to_world, unproject   # noqa: E402
from gwm.perception.run import load_geometry                     # noqa: E402


def cluster(points: np.ndarray, link: float) -> list[np.ndarray]:
    """体素格 + 连通域。比层次聚类便宜得多，而且格子边长就是「多近算一个东西」。"""
    if len(points) < 20:
        return []
    cell = np.floor(points / link).astype(np.int64)
    uniq, inverse = np.unique(cell, axis=0, return_inverse=True)
    index = {tuple(c): i for i, c in enumerate(uniq)}
    seen = np.zeros(len(uniq), bool)
    groups = []
    neigh = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for start in range(len(uniq)):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            c = uniq[k]
            for d in neigh:
                j = index.get((c[0] + d[0], c[1] + d[1], c[2] + d[2]))
                if j is not None and not seen[j]:
                    seen[j] = True; stack.append(j)
        members = np.isin(inverse, comp)
        if members.sum() >= 20:
            groups.append(points[members])
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--link", type=float, default=0.08, help="体素边长（米）：多近算一个东西")
    ap.add_argument("--ground-eps", type=float, default=0.06, help="离地面这么近的点算地面")
    ap.add_argument("--tol", type=float, default=0.45, help="和 eval 的 --max-dist 对齐")
    a = ap.parse_args(argv)

    tot_cluster = tot_gt = covered = on_gt = 0
    per_clip = []
    for run in sorted(Path(a.report).iterdir()):
        geo = run / "perception" / "geometry.npz"
        gtp = Path(a.gt_root) / run.name / "program.json"
        if not (geo.is_file() and gtp.is_file()):
            continue
        gt = json.loads(gtp.read_text(encoding="utf-8"))
        gt_pos = np.asarray([n["pose"]["pos"] for n in gt.get("objects", [])
                             if isinstance((n.get("pose") or {}).get("pos"), list)], float)
        if not len(gt_pos):
            continue
        geom = load_geometry(geo)
        pts = []
        for k in range(0, len(geom.frame_indices), max(1, len(geom.frame_indices) // 6)):
            d = geom.depth[k]
            m = d > 1e-4
            p = to_world(unproject(d, geom.intrinsics[k], m, stride=6), _fix(geom.cam_to_world[k]))
            if len(p):
                pts.append(p)
        if not pts:
            continue
        P = np.vstack(pts)
        # 地面：取 y 的低分位当地平面，附近的点去掉。评测那边的世界坐标已经把地面
        # 压到 y=0 附近，但这里是感知自己的坐标系，所以从数据里估。
        floor = float(np.percentile(P[:, 1], 2))
        P = P[P[:, 1] > floor + a.ground_eps]
        groups = cluster(P, a.link)
        centres = np.asarray([g.mean(0) for g in groups]) if groups else np.zeros((0, 3))
        # 簇和真值物体的距离——两边坐标系差一个规范自由度，所以这里只看**数量**
        # 和「一个簇附近有没有真值物体」的粗配比，不当精确率用
        tot_cluster += len(groups); tot_gt += len(gt_pos)
        per_clip.append((run.name, len(groups), len(gt_pos)))

    if not per_clip:
        print("没有可用样本"); return 1
    print(f"体素边长 {a.link} m，地面裕度 {a.ground_eps} m\n")
    print(f"{'片段':<14}{'几何簇数':>9}{'真值物体数':>11}{'比值':>8}")
    for name, nc, ng in per_clip:
        print(f"{name:<14}{nc:>9}{ng:>11}{nc/max(ng,1):>8.1f}")
    print(f"\n合计：几何簇 {tot_cluster}，真值物体 {tot_gt}，比值 {tot_cluster/max(tot_gt,1):.2f}")
    print("对照：现在的文字检测器是 133 比 59 = 2.3 倍")
    print("\n（这一版只比**数量**。数量接近才值得去做真正的配对评测——")
    print("  簇的坐标在感知自己的坐标系里，要和真值比位置得先过规范对齐。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
