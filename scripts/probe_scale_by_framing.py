#!/usr/bin/env python
"""同一批数据里相机高度差六倍，而标定假设是一个常数——按取景高度分组量尺度。

起因：查几何聚类的时候顺手打了一张表，发现 `ab_off_report` 这十二条片段在
`align_world` 里分成泾渭分明的两组：

  五条走 `facing_camera` 兜底分支（选中的平面法向离「上」73°–80°），
  七条走正常分支（法向离「上」3.7°–17.2°）。

去真值里一对，两组**完全由取景高度决定**：

  走兜底的五条，真值相机高 2.96 – 3.89 米
  走正常的七条，真值相机高 0.62 – 1.11 米

#68 的更正里写过「ab015 真值相机高度中位 **1.11 米**，和 `assumed_camera_height_m:
1.2` 几乎吻合，所以在它上面标定是对的」。那个中位数掩盖了一个六倍的分布——
**同一批数据里有三分之一的片段相机在三米半以上。**

这正是我这几天反复犯的同一类错误：合计/中位掩盖分布。先是「0/1 的中位数」，
再是「一半塌成一坨、一半碎成渣的簇数平均」，现在是相机高度的中位数。

## 这个脚本量什么

`assumed_camera_height_m` 是整条链路把相对尺度标定到米的唯一依据：
`scale = 1.2 / 重建出来的相机高度`。真值相机在 3.5 米而我们按 1.2 米标定，重建出来
的场景就该被压小约 **1.2 / 3.5 ≈ 0.34** 倍。

所以预期是：**尺度之比 ≈ 1.2 / 真值相机高度**。量两个**不经过配对**的比值来验
（配对集会把尺度错的筛掉，这个坑 #68 里记着）：

  物体最长边的中位数之比
  节点中心铺开程度之比（各轴四分位距，取模）

用法:
  python -m scripts.probe_scale_by_framing --report <产物目录> \
      --gt-root <真值根> --rendered-root <渲染根>
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

from gwm.feedback.gt_metrics import node_extent   # noqa: E402


def nodes(program: dict) -> list[dict]:
    return list(program.get("objects") or []) + list(program.get("static") or [])


def longest_edge_median(program: dict) -> float:
    vals = []
    for n in nodes(program):
        e = node_extent(n)
        if e is not None and np.isfinite(e).all() and e.max() > 0:
            vals.append(float(e.max()))
    return float(np.median(vals)) if vals else float("nan")


def spread(program: dict) -> float:
    """节点中心的铺开程度：各轴四分位距取模。

    用四分位距不用包围盒：一个飞到几十米外的离群节点会把包围盒整个带偏，而尺度
    是个整体量，不该由一个点决定。
    """
    pos = [n["pose"]["pos"] for n in nodes(program)
           if isinstance((n.get("pose") or {}).get("pos"), list) and len(n["pose"]["pos"]) == 3]
    if len(pos) < 3:
        return float("nan")
    P = np.asarray(pos, float)
    iqr = np.percentile(P, 75, axis=0) - np.percentile(P, 25, axis=0)
    return float(np.linalg.norm(iqr))


def gt_camera_height(path: Path) -> float:
    hs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            cam = (json.loads(line).get("camera") or {}).get("pos")
        except json.JSONDecodeError:
            continue
        if isinstance(cam, list) and len(cam) == 3:
            hs.append(float(cam[1]))
    return float(np.median(hs)) if hs else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--rendered-root", required=True)
    ap.add_argument("--assumed", type=float, default=1.2,
                    help="配置里的 assumed_camera_height_m，用来算预期比值")
    a = ap.parse_args(argv)

    rows = []
    for run in sorted(Path(a.report).iterdir()):
        pdp, gtp = run / "program.json", Path(a.gt_root) / run.name / "program.json"
        st = Path(a.rendered_root) / run.name / "gt_states.jsonl"
        ev = run / "perception" / "evidence.json"
        if not (pdp.is_file() and gtp.is_file() and st.is_file() and ev.is_file()):
            continue
        gt, pd = (json.loads(p.read_text(encoding="utf-8")) for p in (gtp, pdp))
        align = (json.loads(ev.read_text(encoding="utf-8")).get("meta") or {}).get("alignment") or {}
        h = gt_camera_height(st)
        rows.append({
            "clip": run.name, "h": h, "mode": align.get("ground_mode") or "normal",
            "angle": align.get("ground_normal_angle_to_up_deg"),
            "edge": longest_edge_median(pd) / (longest_edge_median(gt) or np.nan),
            "spread": spread(pd) / (spread(gt) or np.nan),
            "expect": a.assumed / h if h and np.isfinite(h) else float("nan"),
        })
    if not rows:
        print("没有可用样本"); return 1

    rows.sort(key=lambda r: r["h"])
    print(f"假设的相机高度 {a.assumed} m，{len(rows)} 条片段\n")
    print(f"{'片段':<14}{'真值相机高':>11}{'地面模式':>15}{'法向偏角':>9}"
          f"{'边长之比':>10}{'铺开之比':>10}{'预期比值':>10}")
    for r in rows:
        print(f"{r['clip']:<14}{r['h']:>11.2f}{r['mode']:>15}"
              f"{(r['angle'] if r['angle'] is not None else float('nan')):>9.1f}"
              f"{r['edge']:>10.2f}{r['spread']:>10.2f}{r['expect']:>10.2f}")

    lo = [r for r in rows if r["h"] < 2.0]
    hi = [r for r in rows if r["h"] >= 2.0]
    print()
    for label, grp in (("相机低于 2 米", lo), ("相机高于 2 米", hi)):
        if not grp:
            continue
        print(f"{label}（{len(grp)} 条）：真值高度中位 "
              f"{np.median([r['h'] for r in grp]):.2f} m，"
              f"边长之比中位 {np.nanmedian([r['edge'] for r in grp]):.2f}，"
              f"铺开之比中位 {np.nanmedian([r['spread'] for r in grp]):.2f}，"
              f"预期 {np.nanmedian([r['expect'] for r in grp]):.2f}")
    print("\n「预期比值」= 假设高度 / 真值高度。实测的两个比值要是跟着它一起掉，")
    print("就说明压小确实来自标定假设，而不是检测或者摆位。两组之间的差才是关键：")
    print("中位数会把这个差抹平，必须分组看。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
