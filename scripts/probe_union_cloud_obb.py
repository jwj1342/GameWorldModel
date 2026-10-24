#!/usr/bin/env python
"""包围盒改成在**跨帧并起来的点云**上拟合一次，而不是逐帧各拟合一次再取统计量。

为什么（#56）。逐帧包围盒中心的抖动是节点自身对角线的 0.43–0.46 倍，而且**不是抖、
是单调漂移**（真值结构件上逐帧误差 0.366 → 0.419 → 0.511 → 0.578 米）。机制很直白：

  `obb_from_points` 拟合的是**当前帧可见点**的分位包围盒。相机一动，物体可见的
  那部分就变，可见点集的包围盒跟着滑——**框跟的是可见区域，不是物体。**

对**静止**物体，各帧的点在同一个世界坐标系里描述同一个刚体的同一个位姿，所以把它们
并起来拟合一次，是严格多用了信息，而不是又一个启发式。这也正是现在没做的事：
`direct.py` 拿到的是逐帧包围盒的序列，只能取首帧或取中位，两种都是在**统计一串本身
就偏掉的估计**。

## 这次特别要防的一个坑

上一轮我在「配上真值结构件的节点」上量出首帧远好于中位，据此改了 `direct.py` 的
**结构件分支**，A/B 下来精细度一个指标没动——因为这条链路把 16 个真值结构件里的
15 个写进了 `objects`，**它们走的是另一个分支**。量具的 population 和被改的代码路径
几乎不重叠。

所以这个脚本**不按我们自己的归类分组**，按**真值**的节点类别分组，并且把两类都报
出来；提出的改动也要同时覆盖两个分支，否则又白跑一轮。

用法:
  python -m scripts.probe_union_cloud_obb --report <产物目录> \
      --eval <scene_eval.json> --gt-root <真值根>
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

from gwm.config import load_config                                     # noqa: E402
from gwm.perception.evidence import (_fix, align_world, obb_from_points,  # noqa: E402
                                     to_world, unproject)
from gwm.perception.masks import resize_mask                           # noqa: E402
from gwm.perception.run import load_geometry, load_masks               # noqa: E402
from scripts.probe_cluster_quality import load_tracks                  # noqa: E402


def estimators(pdir: Path, cfg: dict, stride: int = 2) -> dict[str, dict]:
    """对每条轨迹算三个位置估计：首帧框心、逐帧框心中位、并起来拟合一次的框心。

    复原链路自己的世界坐标：align_world 的 T 和尺度一个都不能漏，否则和程序里的
    位置不在一个坐标系里——这个坑在 probe_cluster_quality 里栽过一次。
    """
    geom = load_geometry(pdir / "geometry.npz")
    tracks = load_tracks(pdir)
    T, scale, _ = align_world(geom, tracks, cfg)
    W, H = geom.size()
    masks = load_masks(pdir)
    slot = {int(fi): k for k, fi in enumerate(geom.frame_indices)}

    out: dict[str, dict] = {}
    for oid, per_frame in masks.items():
        centres, cloud = [], []
        for fi, m in sorted(per_frame.items()):
            k = slot.get(int(fi))
            if k is None:
                continue
            mres = resize_mask(m, (W, H))
            pts = unproject(geom.depth[k], geom.intrinsics[k], mres, stride=stride)
            if not len(pts):
                continue
            w = (to_world(pts, _fix(geom.cam_to_world[k])) @ T[:3, :3].T + T[:3, 3]) * scale
            ob = obb_from_points(w)
            if ob is None:
                continue
            centres.append(ob["center"])
            cloud.append(w)
        if len(centres) < 2:
            continue
        C = np.asarray(centres, float)
        union = obb_from_points(np.vstack(cloud)) if cloud else None
        out[oid] = {
            "首帧": C[0],
            "逐帧中位": np.median(C, 0),
            "并云拟合": np.asarray(union["center"], float) if union else None,
            "并云尺寸": np.asarray(union["size"], float) if union else None,
            "逐帧中位尺寸": None,
            "n_frames": len(centres),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args(argv)
    cfg = load_config(a.config or [str(REPO / "configs" / "default.yaml")])

    # 「产物里的首帧」= 直接读 evidence.json 存下来的包围盒中心。它和我们现算的
    # 「首帧」差在两处**链路自己做的后处理**：
    #   1. `smooth_series` 对中心序列做了时间平滑；
    #   2. `contact_center` 把贴地物体的中心换成了掩码最低点与地面的交点
    #      ——注意它只对 `not is_structure` 的轨迹生效，而 `is_static_class` 几乎
    #      总是 False，所以实际上几乎所有轨迹都吃到了这个修正。
    # 两者的差就是这些后处理值多少。不把它列出来，就会拿「没后处理的新估计」去比
    # 「有后处理的旧估计」，又是一次口径不对等的比较。
    EST = ("首帧", "逐帧中位", "并云拟合", "产物里的首帧")
    err: dict[str, dict[str, list[float]]] = {
        k: {e: [] for e in EST} for k in ("真值结构件", "真值可动物体")}
    wins = {k: {e: 0 for e in EST} for k in err}

    for r in json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", []):
        full = (r.get("metrics") or {}).get("full") or {}
        g = full.get("gauge_alignment") or {}
        pairs = {q: gk for gk, q in ((full.get("scene") or {}).get("pairs") or {}).items() if q}
        run = Path(a.report) / r["clip"]
        gtp = Path(a.gt_root) / r["clip"] / "program.json"
        if not (gtp.is_file() and (run / "perception" / "geometry.npz").is_file()
                and "yaw_deg" in g):
            continue
        yaw = np.radians(float(g["yaw_deg"])); c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        t = np.asarray([g["translation_xz_m"][0], 0.0, g["translation_xz_m"][1]], float)
        sc = float(g.get("scale") or 1.0)
        gt = json.loads(gtp.read_text(encoding="utf-8"))
        gt_pos = {f"static::{n['id']}": n["pose"]["pos"] for n in (gt.get("static") or [])
                  if isinstance((n.get("pose") or {}).get("pos"), list)}
        gt_pos.update({n["id"]: n["pose"]["pos"] for n in (gt.get("objects") or [])
                       if isinstance((n.get("pose") or {}).get("pos"), list)})

        est = estimators(run / "perception", cfg, a.stride)
        stored: dict[str, np.ndarray] = {}
        evp = run / "perception" / "evidence.json"
        if evp.is_file():
            for entry in (json.loads(evp.read_text(encoding="utf-8")).get("objects") or []):
                ob = (entry.get("obb") or [])
                if ob and isinstance(ob[0].get("center"), list):
                    stored[str(entry.get("id"))] = np.asarray(ob[0]["center"], float)
        for oid, e in est.items():
            gk = pairs.get(oid)
            if gk not in gt_pos or e["并云拟合"] is None:
                continue
            kind = "真值结构件" if gk.startswith("static::") else "真值可动物体"
            target = np.asarray(gt_pos[gk], float)
            e = {**e, "产物里的首帧": stored.get(oid)}
            if e["产物里的首帧"] is None:
                continue
            vals = {name: float(np.linalg.norm(sc * (R @ e[name]) + t - target)) for name in EST}
            for name, v in vals.items():
                err[kind][name].append(v)
            wins[kind][min(vals, key=vals.get)] += 1

    if not any(err[k]["首帧"] for k in err):
        print("没有可用样本"); return 1

    for kind in err:
        n = len(err[kind]["首帧"])
        print(f"\n—— {kind}（n={n}）——")
        if not n:
            print("  没有样本"); continue
        for name in EST:
            v = np.asarray(err[kind][name])
            print(f"  {name:<10} 离真值 中位 {np.median(v):.3f} m，"
                  f"四分位 {np.percentile(v,25):.3f} / {np.percentile(v,75):.3f}，"
                  f"容差 0.45 内 {(v <= 0.45).mean():.3f}")
        print(f"  各自最优的次数：" +
              "，".join(f"{name} {wins[kind][name]}" for name in EST))
    print("\n真值结构件那一组是干净的对照：位置恒定，三种估计比的是同一个靶子。")
    print("可动物体那一组真值 pose.pos 是起点，首帧按构造占便宜，只能当参考。")
    sk, fk = "真值结构件", "首帧"
    if err[sk][fk] and err[sk]["产物里的首帧"]:
        d = np.median(np.asarray(err[sk][fk])) - np.median(np.asarray(err[sk]["产物里的首帧"]))
        print(f"\n「产物里的首帧」比现算的首帧好 {d:+.3f} m（中位）。这个差就是链路自己的")
        print("后处理——中心序列时间平滑 + 贴地物体换成接触点——值多少。")
        print("注意接触点那一步只对 `not is_structure` 生效，而 is_static_class 几乎总是")
        print("False，所以几乎所有轨迹都吃到了它。**真把结构件判对了，它们反而会失去这个")
        print("修正**，这是个得先想清楚的反向耦合。")

    print("\n要点：并云拟合如果明显更好，那它是**多用了信息**，不是又一个启发式")
    print("——同一个静止物体在各帧的点本来就描述同一个刚体的同一个位姿。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
