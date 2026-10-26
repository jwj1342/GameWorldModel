#!/usr/bin/env python
"""几何簇当候选，到底对不对得上真东西——这次做配对，不只比数量。

`probe_geometric_proposals` 只比了**数量**（119 个簇对 128 个真实几何体，比值 0.93），
而且那个脚本自己在结尾写了「不当精确率用」。数量接近是必要条件，远不是充分条件：
119 个簇完全可以全长在错的位置上。这里补上配对。

要回答的是一个 go / no-go：**把候选从文字检测器换成几何聚类，值不值得做。**
现在文字检测器这一路的构成已经量清楚了（#56）——112 个预测物体里 21.4% 配上真值
object、22.3% 是真实结构件被切成的碎片、17.9% 压在另一个 object 上、38.4% 什么都
没有。而点云层面结构件是完整的。如果几何簇确实落在真东西上，这条路就该走。

## 口径

规范对齐（偏航 + 水平平移）直接从评测产物里取：`align_gauge` 把 y 方向的平移钉死成
0（地面已经压到 y=0，高度是可观测的），所以 eval.json 里存的 `yaw_deg` 和
`translation_xz_m` 就是完整的变换，不用重新估。**这很重要**——我在这上面栽过一次：
不做规范对齐直接比世界坐标，量出来精确率 0.015，而评测是 0.226，因为那条片段的
规范偏航是 157°。

真值位置用 `program.json` 里的 `pose.pos`。对**不动的**节点（全部 static，加上
motion 是 static 的 object）这就是精确位置；对会动的 object，`pose.pos` 是起点或者
振荡中心，而簇是几帧点云堆在一起的，两边都带着各自的平均，对不齐。所以主结论只看
不动的那一档，会动的单列一行并标明它不准。不动的那一档正好就是碎片问题所在的地方。

## 自查

同一套配对代码先跑一遍**现有的预测程序**，和评测自己报的配对数并排打出来。两者
差太多就说明这个脚本的口径和评测对不上，后面簇的数一个都不能信。

自查本身写错过一次：第一版拿「不动节点 + 含 static」去比评测的「全部 object」，
还在输出里断言前者「该明显大于」后者。那两个集合既不是包含关系也不是同一个集合
——含 static 把真值这边放宽了，只看不动的又把会动的那边收窄了，净效果不确定。
量具自己也要有对照口径，不能想当然。现在换成同口径：只取 objects。

用法:
  python -m scripts.probe_cluster_quality --report <产物目录> --eval <eval.json> \
      --gt-root <真值根> [--link 0.08]
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

from gwm.config import load_config                              # noqa: E402
from gwm.feedback.gt_metrics import _rot_y                       # noqa: E402
from gwm.perception.base import Tracks, TrackedObject            # noqa: E402
from gwm.perception.evidence import _fix, align_world, to_world, unproject  # noqa: E402
from gwm.perception.run import load_geometry, load_masks         # noqa: E402
from scripts.probe_geometric_proposals import cluster            # noqa: E402


def load_tracks(out_dir: Path) -> Tracks:
    """从缓存的 masks.npz 复原 Tracks。`align_world` 要用它把物体像素从静态点里剔掉，
    而且候选平面的打分里有一项是「多少物体点在这个平面上方」——没有 tracks 就会选中
    不同的平面，T 也就不是链路实际用的那个。"""
    mi = json.loads((out_dir / "masks_index.json").read_text(encoding="utf-8"))
    masks = load_masks(out_dir)
    objs = [TrackedObject(id=oid, phrase=m["phrase"], score=m["score"], masks=masks.get(oid, {}),
                          identity_hypotheses=m.get("identity_hypotheses", []),
                          source_detection_ids=m.get("source_detection_ids", []))
            for oid, m in mi["objects"].items()]
    return Tracks(objects=objs, frame_size=tuple(mi["frame_size"]), backend=mi["backend"],
                  association_diagnostics=mi.get("association_diagnostics", []))


def still_nodes(program: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回 (不动节点的位置, 会动节点的位置)。

    static 列表整个算不动；objects 里要看 motion——没有 motion 字段等于 static。
    """
    still, moving = [], []
    for node in program.get("static", []) or []:
        pos = (node.get("pose") or {}).get("pos")
        if isinstance(pos, list) and len(pos) == 3:
            still.append(pos)
    for node in program.get("objects", []) or []:
        pos = (node.get("pose") or {}).get("pos")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        kind = (node.get("motion") or {}).get("type") or "static"
        (still if kind == "static" else moving).append(pos)
    return (np.asarray(still, float).reshape(-1, 3),
            np.asarray(moving, float).reshape(-1, 3))


def object_positions(program: dict) -> np.ndarray:
    """只取 objects 列表的位置，不含 static——评测的召回和精确率就是在这个集合上算的，
    自查要和它对齐口径。"""
    out = [n["pose"]["pos"] for n in (program.get("objects") or [])
           if isinstance((n.get("pose") or {}).get("pos"), list)
           and len((n["pose"]["pos"])) == 3]
    return np.asarray(out, float).reshape(-1, 3)


def greedy_match(gt: np.ndarray, pred: np.ndarray, tol: float) -> int:
    """一对一贪心配对，返回配上的对数。近的先配，配过的两边都不再用。

    用贪心不用匈牙利：容差内的配对在这个尺度上几乎没有竞争，而贪心的结果更容易
    手算复核。真有竞争的时候贪心会略低估，方向是保守的。
    """
    if not len(gt) or not len(pred):
        return 0
    d = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=-1)
    used_g, used_p, hits = set(), set(), 0
    for g, p in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[g, p] > tol:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p); hits += 1
    return hits


def clip_clusters(out_dir: Path, cfg: dict, link: float, ground_eps: float,
                  frames: int) -> tuple[np.ndarray, dict]:
    """在**链路自己的对齐坐标系里**做聚类，返回 (簇中心, 诊断量)。

    第一版栽在这里：它拿 `to_world(unproject(...), _fix(cam_to_world))` 得到的**原始**
    世界坐标就直接聚类了，而链路里「地面压到 y=0、y 轴朝上」是 `align_world` 返回的
    变换 T 和尺度因子干的，探针从来没应用过 T。后果有两个，都致命：

      1. 原始坐标里 y 不是「上」，拿 y 的低分位当地平面没有意义，再加上天空和远处的
         离群点把低分位拖得很低——实测「去地」只删掉了 2% 到 9% 的点，地面全留着，
         于是十二条里有八条整个场景连成一个连通域。
      2. 簇中心在未对齐坐标系、程序里的物体位置在对齐后的坐标系，两者之间的距离
         没有意义。

    所以这里直接调 `align_world` 本人，而不是重写一遍它的平面挑选逻辑（那段有俯拍
    兜底、有「多少物体点在平面上方」的打分，重写必错）。它要 tracks，所以得先从
    masks.npz 复原。

    `evidence.json` 里存了 `camera0_height_before_scale` 和 `scale_factor`，诊断量里
    带上我们自己算出来的同名值——对不上就说明 T 重建错了，后面的数一个都不能信。
    """
    geom = load_geometry(out_dir / "geometry.npz")
    tracks = load_tracks(out_dir)
    T, scale, info = align_world(geom, tracks, cfg)

    pc = cfg["perception"]
    far = pc.get("far_depth_factor", 3.0)
    pts = []
    step = max(1, len(geom.frame_indices) // frames)
    for k in range(0, len(geom.frame_indices), step):
        d = geom.depth[k]
        good = d > 1e-4
        if not good.any():
            continue
        # 天空和远处的垃圾点必须先扔：align_world 自己就是这么做的，留着它们
        # 既会把地面的分位数估歪，也会拖出一堆连不上任何东西的离群簇。
        m = good & (d < far * float(np.median(d[good])))
        p = to_world(unproject(d, geom.intrinsics[k], m, stride=6), _fix(geom.cam_to_world[k]))
        if len(p):
            pts.append(p)
    if not pts:
        return np.zeros((0, 3)), {}
    P = np.vstack(pts)
    P = (P @ T[:3, :3].T + T[:3, 3]) * scale        # 对齐 + 定尺度，和链路一致
    before = len(P)
    P = P[P[:, 1] > ground_eps]                     # 地面现在确实在 y=0
    groups = cluster(P, link)
    sizes = sorted((len(g) for g in groups), reverse=True)
    diag = {"pts_before": before, "pts_after": len(P), "kept": len(P) / max(before, 1),
            "top_share": (sizes[0] / max(len(P), 1)) if sizes else 0.0,
            "n": len(groups), "ground_mode": info.get("ground_mode", ""),
            "ground_angle": info.get("ground_normal_angle_to_up_deg"),
            "cam_h": info.get("camera0_height_before_scale"),
            "scale": info.get("scale_factor")}
    centres = np.asarray([g.mean(0) for g in groups]) if groups else np.zeros((0, 3))
    return centres, diag


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True, help="评测产物，规范对齐和自查的对照数从这里取")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--link", type=float, default=0.08)
    ap.add_argument("--ground-eps", type=float, default=0.06)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.45, help="和 eval 的 --max-dist 对齐")
    ap.add_argument("--config", action="append", default=[],
                    help="重建 T 要用和产出这批结果时同一份配置")
    a = ap.parse_args(argv)
    cfg = load_config(a.config or [str(REPO / "configs" / "default.yaml")])

    rows = json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", [])
    gauge = {}
    for r in rows:
        g = ((r.get("metrics") or {}).get("full") or {}).get("gauge_alignment") or {}
        pairs = ((r.get("metrics") or {}).get("matching") or {}).get("pairs") or {}
        if "yaw_deg" in g and "translation_xz_m" in g:
            gauge[r.get("clip")] = (g, sum(1 for v in pairs.values() if v))

    out = []
    for run in sorted(Path(a.report).iterdir()):
        name = run.name
        geo = run / "perception" / "geometry.npz"
        gtp = Path(a.gt_root) / name / "program.json"
        pdp = run / "program.json"
        if not (geo.is_file() and gtp.is_file() and pdp.is_file() and name in gauge):
            continue
        g, eval_matched = gauge[name]
        ev = json.loads((run / "perception" / "evidence.json").read_text(encoding="utf-8"))
        recorded = (ev.get("meta") or {}).get("alignment") or {}
        R = _rot_y(float(g["yaw_deg"]))
        t = np.asarray([g["translation_xz_m"][0], 0.0, g["translation_xz_m"][1]], float)

        gt_prog = json.loads(gtp.read_text(encoding="utf-8"))
        pd_prog = json.loads(pdp.read_text(encoding="utf-8"))
        gt_still, gt_moving = still_nodes(gt_prog)
        pd_still, pd_moving = still_nodes(pd_prog)
        gt_obj, pd_obj = object_positions(gt_prog), object_positions(pd_prog)
        pd_still, pd_moving = pd_still @ R.T + t, pd_moving @ R.T + t
        pd_obj = pd_obj @ R.T + t if len(pd_obj) else pd_obj
        centres, diag = clip_clusters(run / "perception", cfg, a.link, a.ground_eps, a.frames)
        centres = centres @ R.T + t if len(centres) else centres

        out.append({
            "clip": name,
            "gt_still": len(gt_still), "gt_moving": len(gt_moving),
            "pred": len(pd_still) + len(pd_moving),
            "pred_hit": greedy_match(gt_still, pd_still, a.tol),
            "n_cluster": len(centres),
            "clu_hit": greedy_match(gt_still, centres, a.tol),
            # 每个真值不动节点到**最近**簇中心的距离。这一项用来分辨两件完全不同的事：
            # 距离中位要是两三米，说明簇根本不在物体上；要是零点六米上下，那就只是
            # 「簇中心取的是可见表面点的质心，不是物体中心」这个系统偏差顶出了容差。
            "near": (np.linalg.norm(gt_still[:, None, :] - centres[None, :, :],
                                    axis=-1).min(1).tolist()
                     if len(gt_still) and len(centres) else []),
            "near_pred": (np.linalg.norm(
                gt_still[:, None, :] - pd_still[None, :, :], axis=-1).min(1).tolist()
                if len(gt_still) and len(pd_still) else []),
            "clu_hit_all": greedy_match(np.vstack([gt_still, gt_moving])
                                        if len(gt_moving) else gt_still, centres, a.tol),
            "eval_matched": eval_matched,
            "pred_hit_all": greedy_match(
                np.vstack([gt_still, gt_moving]) if len(gt_moving) else gt_still,
                np.vstack([pd_still, pd_moving]) if len(pd_moving) else pd_still, a.tol),
            "gt_obj": len(gt_obj), "pred_obj_hit": greedy_match(gt_obj, pd_obj, a.tol),
            "pred_obj": len(pd_obj),
            **diag,
            "rec_cam_h": recorded.get("camera0_height_before_scale"),
            "rec_scale": recorded.get("scale_factor"),
            "rec_mode": recorded.get("ground_mode", ""),
        })

    if not out:
        print("没有同时具备几何、真值、评测记录的片段"); return 1

    def col(k):
        return sum(r[k] for r in out)

    print(f"体素边长 {a.link} m，地面裕度 {a.ground_eps} m，容差 {a.tol} m，"
          f"{len(out)} 条片段\n")
    print(f"{'片段':<14}{'真值不动':>9}{'真值会动':>9}{'检测器':>8}{'检测器配上':>11}"
          f"{'簇数':>7}{'簇配上':>8}{'去地后留下':>11}{'最大簇占比':>11}")
    for r in out:
        print(f"{r['clip']:<14}{r['gt_still']:>9}{r['gt_moving']:>9}{r['pred']:>8}"
              f"{r['pred_hit']:>11}{r['n_cluster']:>7}{r['clu_hit']:>8}"
              f"{r.get('kept', 0):>11.2f}{r.get('top_share', 0):>11.2f}")
    print("  「最大簇占比」接近 1 就是塌成一坨了——去地没干净，整个场景连成一个连通域。")
    print("  只看合计的簇数会把「一半塌成一坨、一半碎成渣」读成「数量刚好」。")

    gs, gm = col("gt_still"), col("gt_moving")
    print(f"\n—— 只看**不动的**真值节点（位置精确，碎片问题就在这一档）——")
    print(f"真值不动节点 {gs} 个")
    print(f"  文字检测器：{col('pred')} 个预测，配上 {col('pred_hit')}，"
          f"查全 {col('pred_hit')/max(gs,1):.3f}，查准 {col('pred_hit')/max(col('pred'),1):.3f}")
    print(f"  几何聚类　：{col('n_cluster')} 个簇，　配上 {col('clu_hit')}，"
          f"查全 {col('clu_hit')/max(gs,1):.3f}，查准 {col('clu_hit')/max(col('n_cluster'),1):.3f}")
    print(f"\n—— 加上会动的真值节点（{gm} 个，**这一行不准**：真值取的是起点/振荡中心，")
    print(f"   簇是几帧堆在一起的平均，两边各带各的平均，对不齐）——")
    print(f"  几何聚类：配上 {col('clu_hit_all')} / {gs + gm}，"
          f"查全 {col('clu_hit_all')/max(gs+gm,1):.3f}")
    near = [v for r in out for v in r["near"]]
    near_pred = [v for r in out for v in r["near_pred"]]
    if near:
        print(f"\n—— 每个真值不动节点到最近候选的距离（米）——")
        print(f"  到最近的**簇**　　：中位 {np.median(near):.2f}，"
              f"四分位 {np.percentile(near,25):.2f} / {np.percentile(near,75):.2f}")
        if near_pred:
            print(f"  到最近的**检测框**：中位 {np.median(near_pred):.2f}，"
                  f"四分位 {np.percentile(near_pred,25):.2f} / {np.percentile(near_pred,75):.2f}")
        print(f"  容差是 {a.tol} m。簇那一行要是两三米，说明簇根本不在物体上；")
        print("  要是零点六米上下，那只是「簇中心是可见表面的质心、不是物体中心」")
        print("  这个系统偏差顶出了容差——两种情况该做的事完全不一样。")

    print(f"\n—— 对齐自查：重建出来的 T 必须和 evidence.json 记的一致 ——")
    print(f"{'片段':<14}{'相机高(算)':>11}{'相机高(记)':>11}{'尺度(算)':>10}{'尺度(记)':>10}"
          f"{'地面模式':>16}{'法向偏角':>9}")
    bad = 0
    for r in out:
        ok = (r.get("cam_h") is not None and r.get("rec_cam_h") is not None
              and abs(r["cam_h"] - r["rec_cam_h"]) < 1e-3)
        bad += 0 if ok else 1
        print(f"{r['clip']:<14}{r.get('cam_h', float('nan')):>11.4f}"
              f"{r.get('rec_cam_h', float('nan')):>11.4f}{r.get('scale', float('nan')):>10.4f}"
              f"{r.get('rec_scale', float('nan')):>10.4f}{str(r.get('ground_mode')):>16}"
              f"{r.get('ground_angle', float('nan')):>9.1f}" + ("" if ok else "   ← 对不上"))
    print(f"  对不上的 {bad} 条。只要有一条对不上，T 就重建错了，上面簇的数全部作废。")
    print("  「地面模式」是 facing_camera、法向偏角几十度的，说明选中的「地面」基本是")
    print("  竖直的——那是给俯拍视角准备的兜底分支，ab015 不该落进去，单独查。")

    print(f"\n—— 自查 ——")
    print(f"  评测自己报的配对数合计          {col('eval_matched')}")
    print(f"  本脚本同口径（只 object）给检测器 {col('pred_obj_hit')} / {col('pred_obj')} 个预测")
    print("  这两个数该接近。差异只该来自一件事：评测用的是观察窗内逐时刻位置的平均，")
    print("  这里用的是 program 里的 pose.pos——对会动的物体两者不是一回事。")
    print("  差得多就说明规范对齐或者坐标系用错了，上面簇的数一个都不能信。")
    print(f"\n  （第一版的自查写错了：拿「不动节点 + 含 static」去比评测的「全部 object」，")
    print(f"   两边既不是包含关系也不是同一个集合，比出来 {col('pred_hit')} < {col('eval_matched')}")
    print(f"   并不说明有 bug。现在换成同口径。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
