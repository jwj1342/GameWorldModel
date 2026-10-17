#!/usr/bin/env python
"""运动信号的诊断：现在喂给运动估计的那个量到底错在哪，换个量行不行。

#7 要把运动估计从包围盒序列换成三维点轨迹。换之前先把问题量清楚，也把几种"不换信号、
只在包围盒中心上做手脚"的廉价修法试掉——这个脚本就是那几次实验，固化下来是为了换完
实现之后能用同一套口径对照。

实验都用**真值感知**跑：掩码取自源程序渲染的 ID 通道，物体 id 和程序节点 id 一一对应，
世界坐标系也和真值一致，所以观测值和真值可以直接相减，不需要对齐规范自由度。

三种模式：
  bias    把位置误差拆成沿视线和垂直视线两个分量，并试「按半个深度推回去」
  offset  按物体估一个与时间无关的沿视线标量偏移（对静止物体有闭式解）
  signal  比较包围盒中心、点云中位数、点云均值三种逐帧统计量

用法:
  python -m scripts.probe_motion_signal --mode signal \
      --rendered $GWM_SCRATCH/dataset/pilot50_rendered --work $SLURM_TMPDIR/probe --limit 8
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.config import load_config, site_name                                      # noqa: E402
from gwm.perception.evidence import _fix, obb_from_points, to_world, unproject     # noqa: E402
from gwm.perception.run import load_geometry, load_masks, run_perception           # noqa: E402


def gt_series(clip_dir: Path) -> tuple[dict, dict]:
    """真值轨迹和运动类型，都按物体 id 索引。"""
    truth: dict = {}
    for line in (clip_dir / "gt_states.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        for obj in state.get("objects", []):
            truth.setdefault(obj["id"], {})[round(float(state["t"]), 3)] = np.array(obj["pos"], float)
    kinds = {o["id"]: (o.get("motion") or {}).get("type", "static")
             for o in json.loads((clip_dir / "program.json").read_text(encoding="utf-8"))["objects"]}
    return truth, kinds


def perceive(clip_dir: Path, work: Path) -> dict:
    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml",
             REPO / "configs/perception/gt.yaml"]
    cfg = load_config([f for f in files if Path(f).is_file()])
    cfg["perception"]["gt_program"] = str(clip_dir / "program.json")
    return run_perception(str(clip_dir / "clip.mp4"), clip_dir.name, work / clip_dir.name, cfg)


def mode_bias(evidence: dict, truth: dict, _k, _work) -> dict:
    """误差沿视线分解。负的沿视线分量 = 观测中心偏在靠近相机的一侧。"""
    poses = {round(float(p["t"]), 3): np.array(p["pos"], float)
             for p in (evidence.get("camera") or {}).get("poses", []) if "t" in p}
    out = {"raw": [], "along": [], "perp": [], "pushed": []}
    for obj in evidence["objects"]:
        series = truth.get(obj["id"])
        if not series:
            continue
        for box in obj["obb"]:
            t = round(float(box["t"]), 3)
            if t not in poses or t not in series:
                continue
            observed, gt = np.array(box["center"], float), series[t]
            view = observed - poses[t]
            norm = float(np.linalg.norm(view))
            if norm < 1e-6:
                continue
            view = view / norm
            error = observed - gt
            along = float(error @ view)
            out["raw"].append(float(np.linalg.norm(error)))
            out["along"].append(along)
            out["perp"].append(float(np.linalg.norm(error - along * view)))
            half = 0.5 * float(np.abs(np.array(box["size"], float)) @ np.abs(view))
            out["pushed"].append(float(np.linalg.norm(observed + half * view - gt)))
    return out


def mode_offset(evidence: dict, truth: dict, kinds: dict, _work) -> dict:
    """每个物体一个与时间无关的沿视线偏移 d：最小化 c(t) - d·v(t) 的方差，闭式解。"""
    poses = {round(float(p["t"]), 3): np.array(p["pos"], float)
             for p in (evidence.get("camera") or {}).get("poses", []) if "t" in p}
    out = {"before": [], "after": [], "d": []}
    for obj in evidence["objects"]:
        if kinds.get(obj["id"]) != "static":
            continue
        rows = []
        for box in obj["obb"]:
            t = round(float(box["t"]), 3)
            if t not in poses:
                continue
            centre = np.array(box["center"], float)
            view = centre - poses[t]
            norm = float(np.linalg.norm(view))
            if norm > 1e-6:
                rows.append((centre, view / norm))
        if len(rows) < 5:
            continue
        C = np.stack([r[0] for r in rows]); V = np.stack([r[1] for r in rows])
        Cc, Vc = C - C.mean(0), V - V.mean(0)
        denom = float((Vc * Vc).sum())
        d = float((Cc * Vc).sum() / denom) if denom > 1e-9 else 0.0
        fixed = C - d * V
        out["before"].append(float(np.linalg.norm(C.max(0) - C.min(0))))
        out["after"].append(float(np.linalg.norm(fixed.max(0) - fixed.min(0))))
        out["d"].append(d)
    return out


def mode_signal(evidence: dict, truth: dict, kinds: dict, run: Path) -> dict:
    """比较三种逐帧统计量。包围盒中心是可见点 3/97 百分位的中点。"""
    from gwm.perception.masks import resize_mask
    geom = load_geometry(run / "geometry.npz")
    masks = load_masks(run)
    frame_of = {f: i for i, f in enumerate(geom.frame_indices)}
    times = {f["frame_index"]: round(float(f["t"]), 3) for f in evidence.get("frames", [])}
    names = ("obb_centre", "point_median", "point_mean")
    out = {f"err/{n}": [] for n in names} | {f"drift/{n}": [] for n in names}
    for oid, per_frame in masks.items():
        if oid not in truth:
            continue
        series = {n: [] for n in names}
        for fi, mask in sorted(per_frame.items()):
            k, t = frame_of.get(fi), times.get(fi)
            if k is None or t is None or t not in truth[oid]:
                continue
            m = np.asarray(mask).astype(bool)
            if m.shape != geom.depth[k].shape:
                m = resize_mask(m, (geom.depth[k].shape[1], geom.depth[k].shape[0]))
            pts = to_world(unproject(geom.depth[k], geom.intrinsics[k], m, stride=4),
                           _fix(geom.cam_to_world[k]))
            if len(pts) < 20:
                continue
            gt = truth[oid][t]
            box = obb_from_points(pts)
            values = {"point_median": np.median(pts, 0), "point_mean": pts.mean(0)}
            if box:
                values["obb_centre"] = np.array(box["center"], float)
            for name, value in values.items():
                out[f"err/{name}"].append(float(np.linalg.norm(value - gt)))
                series[name].append(value)
        if kinds.get(oid) == "static":
            for name, values in series.items():
                if len(values) >= 5:
                    V = np.stack(values)
                    out[f"drift/{name}"].append(float(np.linalg.norm(V.max(0) - V.min(0))))
    return out


MODES = {"bias": mode_bias, "offset": mode_offset, "signal": mode_signal}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in Path(a.rendered).iterdir() if (p / "clip.mp4").is_file())[:a.limit]
    merged: dict[str, list] = {}
    for clip_dir in clips:
        try:
            evidence = perceive(clip_dir, work)
        except Exception as error:                   # 一条挂了不该拖垮整轮
            print(f"{clip_dir.name}: 失败 {type(error).__name__}: {error}"[:200], flush=True)
            continue
        truth, kinds = gt_series(clip_dir)
        for key, values in MODES[a.mode](evidence, truth, kinds, work / clip_dir.name).items():
            merged.setdefault(key, []).extend(values)
    if not any(merged.values()):
        print("没有可比对的样本")
        return 1
    print(f"模式 {a.mode}，片段 {len(clips)} 条")
    for key, values in merged.items():
        if values:
            print(f"  {key:<24} 中位数 {statistics.median(values):>8.3f}   样本 {len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
