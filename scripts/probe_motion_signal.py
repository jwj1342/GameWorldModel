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
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.config import load_config, site_name                                      # noqa: E402
from gwm.perception.evidence import _fix, obb_from_points, to_world, unproject     # noqa: E402
from gwm.perception.registration import infer_articulation, looks_static, track_rigid  # noqa: E402
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


def mode_register(evidence: dict, truth: dict, kinds: dict, run: Path) -> dict:
    """跨帧刚体配准 vs 包围盒中心：同一批物体、同一批帧，比轨迹形状和静止漂移。

    位置的绝对值不可比——配准给的是相对第一帧锚点的轨迹，锚点落在可见表面上，本来就
    偏。所以比的是**相对第一帧的位移**，那才是运动估计真正用到的量。
    """
    from gwm.perception.masks import resize_mask
    geom = load_geometry(run / "geometry.npz")
    masks = load_masks(run)
    frame_of = {f: i for i, f in enumerate(geom.frame_indices)}
    times = {f["frame_index"]: round(float(f["t"]), 3) for f in evidence.get("frames", [])}
    boxes = {o["id"]: {round(float(b["t"]), 3): np.array(b["center"], float) for b in o["obb"]}
             for o in evidence["objects"]}
    out: dict[str, list] = {"step/obb": [], "step/register": [],
                            "drift_static/obb": [], "drift_static/register": [],
                            "rmse": [],
                            # looks_static 在真值静止 / 真值运动的物体上各自多久触发一次。
                            # 它在运动物体上触发就是误判成静止，而「全判成静止」正是
                            # 现在混淆矩阵的样子，所以必须单独量。
                            "static_detector/on_static": [],
                            "static_detector/on_moving": [],
                            # 每步位移相对物体对角线的比例。静止和运动的分布差多少，
                            # 决定了这个判据有没有可用的阈值——两条分布重叠得厉害
                            # 就说明判据本身不行，不是阈值没调好。
                            "step_frac/static": [],
                            "step_frac/moving": [],
                            # 点到面 vs 点到点：平板上点到点会沿面滑动，而我们的物体
                            # 大多是盒子和平板，所以这是最该试的一条
                            "step/register_plane": [],
                            "drift_static/register_plane": [],
                            "step_frac/plane_static": [],
                            "step_frac/plane_moving": []}
    for oid, per_frame in masks.items():
        if oid not in truth:
            continue
        clouds, stamps = [], []
        for fi, mask in sorted(per_frame.items()):
            k, t = frame_of.get(fi), times.get(fi)
            if k is None or t is None or t not in truth[oid]:
                continue
            m = np.asarray(mask).astype(bool)
            if m.shape != geom.depth[k].shape:
                m = resize_mask(m, (geom.depth[k].shape[1], geom.depth[k].shape[0]))
            pts = to_world(unproject(geom.depth[k], geom.intrinsics[k], m, stride=4),
                           _fix(geom.cam_to_world[k]))
            if len(pts) >= 30:
                clouds.append(pts); stamps.append(t)
        if len(clouds) < 6:
            continue
        tracked = track_rigid(clouds)
        if not tracked["ok"]:
            continue
        is_static = kinds.get(oid) == "static"
        sizes_ = [b["size"] for b in (next((o["obb"] for o in evidence["objects"]
                                            if o["id"] == oid), []) or [])]
        diag = float(np.linalg.norm(np.median(sizes_, axis=0))) if sizes_ else 1.0
        registered = np.array(tracked["positions"], float)
        out["rmse"].extend(r for r in tracked["residuals"] if r is not None)
        fired = 1.0 if looks_static(tracked, diag) else 0.0
        out["static_detector/on_static" if is_static else "static_detector/on_moving"].append(fired)
        steps = tracked.get("step_translation") or []
        if steps and diag > 1e-6:
            out["step_frac/static" if is_static else "step_frac/moving"].append(
                float(np.median(steps)) / diag)
        gt = np.stack([truth[oid][t] for t in stamps])
        obb = np.stack([boxes.get(oid, {}).get(t, np.full(3, np.nan)) for t in stamps])
        gt_rel = gt - gt[0]
        reg_rel = registered - registered[0]
        out["step/register"].extend(np.linalg.norm(reg_rel - gt_rel, axis=1).tolist())
        # 模型吸收多少帧：转过半圈的物体，后半段看到的面可能根本不在模型里
        for frames_in_model, tag in ((12, "m12"), (len(clouds), "mall")):
            variant = track_rigid(clouds, model_frames=frames_in_model)
            if variant["ok"] and len(variant["positions"]) == len(gt):
                v = np.array(variant["positions"], float)
                out.setdefault(f"step/{tag}", []).extend(
                    np.linalg.norm((v - v[0]) - gt_rel, axis=1).tolist())
                if is_static:
                    out.setdefault(f"drift_static/{tag}", []).append(
                        float(np.linalg.norm(v.max(0) - v.min(0))))
        plane = track_rigid(clouds, point_to_plane=True)
        if plane["ok"] and len(plane["positions"]) == len(gt):
            p_pos = np.array(plane["positions"], float)
            out["step/register_plane"].extend(
                np.linalg.norm((p_pos - p_pos[0]) - gt_rel, axis=1).tolist())
            p_steps = plane.get("step_translation") or []
            if p_steps and diag > 1e-6:
                out["step_frac/plane_static" if is_static else "step_frac/plane_moving"].append(
                    float(np.median(p_steps)) / diag)
            if is_static:
                out["drift_static/register_plane"].append(
                    float(np.linalg.norm(p_pos.max(0) - p_pos.min(0))))
        if not np.isnan(obb).any():
            obb_rel = obb - obb[0]
            out["step/obb"].extend(np.linalg.norm(obb_rel - gt_rel, axis=1).tolist())
        if kinds.get(oid) == "static":
            out["drift_static/register"].append(
                float(np.linalg.norm(registered.max(0) - registered.min(0))))
            if not np.isnan(obb).any():
                out["drift_static/obb"].append(float(np.linalg.norm(obb.max(0) - obb.min(0))))
    return out


def mode_articulation(evidence: dict, truth: dict, kinds: dict, run: Path) -> dict:
    """关节推断准不准。合成片段里 revolute 和 prismatic 就是真的关节，别的都不是，
    所以可以直接算查准和查全——这是判断阈值该定多松的唯一依据。"""
    from gwm.perception.masks import resize_mask
    geom = load_geometry(run / "geometry.npz")
    masks = load_masks(run)
    frame_of = {f: i for i, f in enumerate(geom.frame_indices)}
    joint_types = {"revolute": "hinge", "prismatic": "slide"}
    out: dict[str, list] = {"true_joint/fired": [], "non_joint/fired": [], "correct_kind": [],
                            "fired/axis_consistency": [], "fired/rotation_deg": [],
                            "fired/straightness": []}
    for oid, per_frame in masks.items():
        kind = kinds.get(oid)
        if kind is None:
            continue
        clouds = []
        for fi, mask in sorted(per_frame.items()):
            k = frame_of.get(fi)
            if k is None:
                continue
            m = np.asarray(mask).astype(bool)
            if m.shape != geom.depth[k].shape:
                m = resize_mask(m, (geom.depth[k].shape[1], geom.depth[k].shape[0]))
            pts = to_world(unproject(geom.depth[k], geom.intrinsics[k], m, stride=4),
                           _fix(geom.cam_to_world[k]))
            if len(pts) >= 30:
                clouds.append(pts)
        if len(clouds) < 6:
            continue
        tracked = track_rigid(clouds)
        spec = infer_articulation(tracked)
        # 原始判据一并记下来：这样扫阈值不用重跑感知，一次测量能回答「阈值定多少」
        total_rot = float(np.degrees(np.arccos(np.clip(
            (np.trace(np.asarray(tracked.get("total_rotation") or np.eye(3), float)) - 1.0) / 2.0, -1.0, 1.0))))
        # #63 的第二条建议：不用阈值，比两个模型的拟合优度。把轨迹按「绕固定轴的
        # 铰链」重建一遍，看位置残差比观测到的轨迹高多少。真铰链应该几乎一样好
        # （约束没有损失解释力），假的会明显更差。比值本身是连续的，不用定阈值就能看
        # 它有没有判别力。
        hinge_ratio = None
        positions = np.asarray(tracked.get("positions") or [], float)
        if len(positions) >= 4:
            # 轴直接从累积旋转矩阵估，**不经过关节推断那道阈值闸**。只在已触发的子集上
            # 算，等于在一个被筛过的分布上看判别力，而这条建议本来的意思是用拟合优度
            # 替代阈值，不是在阈值之后再加一层。第一版就是那么写的，结论因此不作数。
            if spec and spec.get("axis"):
                axis = np.asarray(spec["axis"], float)
                pivot = np.asarray(spec.get("pivot") or positions[0], float)
            else:
                rot = np.asarray(tracked.get("total_rotation") or np.eye(3), float)
                # 旋转矩阵的轴是它对应特征值 1 的特征向量；反对称部分给出方向
                skew = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0],
                                 rot[1, 0] - rot[0, 1]], float)
                axis = skew if np.linalg.norm(skew) > 1e-9 else np.array([0.0, 1.0, 0.0])
                pivot = positions.mean(axis=0)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                axis = axis / norm
                rel = positions - pivot
                along = rel @ axis                      # 沿轴的分量，铰链下应当不变
                flat = rel - np.outer(along, axis)      # 垂直于轴的分量，应当等半径旋转
                radius = np.linalg.norm(flat, axis=1)
                # 约束模型的自由度只剩「转过多少角度」：半径和沿轴位置都该是常数。
                # 残差就取它们相对第一帧的偏离。
                constrained = float(np.sqrt(np.mean(
                    (radius - radius[0]) ** 2 + (along - along[0]) ** 2)))
                # 自由模型的尺度：轨迹自身的位移幅度。两者同量纲才可比。
                free = float(np.sqrt(np.mean(np.sum((positions - positions[0]) ** 2, axis=1))))
                hinge_ratio = round(constrained / max(free, 1e-9), 3)

        # 运动分类器对同一个物体的判断。#63 的第一条建议是「两个估计器同时同意时
        # 置信度更高」，要验它就得把两边放在同一行里——它们是独立的两条路：
        # 关节推断解的是配准逐步变换里的轴和支点，运动分类拟合的是整条轨迹的形状。
        guess = next((o.get("motion_guess") or {} for o in evidence.get("objects", [])
                      if o.get("id") == oid), {})
        out.setdefault("rows", []).append({
            "id": oid, "gt_kind": kind, "is_joint": kind in joint_types,
            "fired": spec["grab"] if spec else None,
            "motion_type": str(guess.get("type", "unknown")),
            "hinge_fit_ratio": hinge_ratio,
            "total_rotation_deg": round(total_rot, 3),
            "axis_consistency": round(float((spec.get("evidence") or {}).get("axis_consistency", 0.0)), 3) if spec else None,
            "straightness": round(float((spec.get("evidence") or {}).get("straightness", 0.0)), 3) if spec else None,
            "travel_m": round(float(np.linalg.norm(np.asarray(tracked.get("total_translation") or np.zeros(3), float))), 3),
            "residual": round(float(np.median([r for r in tracked.get("residuals") or [] if r is not None] or [0.0])), 4),
        })
        fired = 1.0 if spec else 0.0
        if kind in joint_types:
            out["true_joint/fired"].append(fired)
            if spec:
                out["correct_kind"].append(1.0 if spec["grab"] == joint_types[kind] else 0.0)
        else:
            out["non_joint/fired"].append(fired)
        if spec:
            ev = spec.get("evidence") or {}
            for key, name in (("axis_consistency", "fired/axis_consistency"),
                              ("total_rotation_deg", "fired/rotation_deg"),
                              ("straightness", "fired/straightness")):
                if key in ev:
                    out[name].append(float(ev[key]))
    return out


def mode_icp(evidence: dict, truth: dict, kinds: dict, run: Path) -> dict:
    """把 ICP 这一步单独拎出来：解出来的转角和真值转角差多少。

    为什么要单独量它。#7 那边量到的链条是：

        点云 → track_rigid 逐帧配准 → 累积旋转 → 角度序列 → 正弦拟合 → periodic_rotate

    而在**真值感知**上（真值深度 + 真值掩码渲出来的点云），真会周期性转动的物体，
    正弦拟合的 r2 中位只有 0.124、最大 0.316；真值**静止**的物体反而能到 0.679。
    分布是反的——噪声比真信号更像正弦。往下调闸只会放进更多假的（排除清单第十三条）。

    所以问题在角度序列本身。这个模式不经过分类、不经过配对、不经过阈值，直接把
    `track_rigid` 解出的逐帧转角和 `gt_states.jsonl` 里记的真值四元数比。差在这里，
    就是 ICP 的问题；不差在这里，就得往上游找。
    """
    from gwm.perception.masks import resize_mask
    gt_quat = _gt_quaternions(run)
    if not gt_quat:
        return {}
    geom = load_geometry(run / "geometry.npz")
    masks = load_masks(run)
    frame_of = {f: i for i, f in enumerate(geom.frame_indices)}
    out: dict[str, list] = {}
    for oid, per_frame in masks.items():
        kind = kinds.get(oid)
        series = gt_quat.get(oid)
        if kind is None or not series:
            continue
        clouds, frames = [], []
        for fi, mask in sorted(per_frame.items()):
            k = frame_of.get(fi)
            if k is None:
                continue
            m = np.asarray(mask).astype(bool)
            if m.shape != geom.depth[k].shape:
                m = resize_mask(m, (geom.depth[k].shape[1], geom.depth[k].shape[0]))
            pts = to_world(unproject(geom.depth[k], geom.intrinsics[k], m, stride=4),
                           _fix(geom.cam_to_world[k]))
            if len(pts) >= 30:
                clouds.append(pts); frames.append(int(fi))
        if len(clouds) < 6:
            continue
        tracked = track_rigid(clouds)
        orientations = tracked.get("orientations") or []
        if len(orientations) != len(clouds):
            continue
        truth_q = [series.get(f) for f in frames]
        if any(q is None for q in truth_q):
            continue
        base = R.from_quat(truth_q[0]).inv()
        true_deg = np.degrees([(base * R.from_quat(q)).magnitude() for q in truth_q])
        solved_deg = np.degrees([R.from_matrix(np.asarray(m, float)).magnitude()
                                 for m in orientations])
        out.setdefault(f"{kind}/true_range_deg", []).append(float(np.ptp(true_deg)))
        out.setdefault(f"{kind}/solved_range_deg", []).append(float(np.ptp(solved_deg)))
        out.setdefault(f"{kind}/angle_rmse_deg", []).append(
            float(np.sqrt(np.mean((solved_deg - true_deg) ** 2))))
        if np.ptp(true_deg) > 1e-6 and np.ptp(solved_deg) > 1e-6:
            out.setdefault(f"{kind}/corr", []).append(
                float(np.corrcoef(solved_deg, true_deg)[0, 1]))
    return out


def _gt_quaternions(run: Path) -> dict:
    """逐帧真值朝向，按物体 id 再按帧序号索引。"""
    states = run.parent / "gt_states.jsonl"
    if not states.is_file():
        states = Path(str(run).replace("/perception", "")) / "gt_states.jsonl"
    if not states.is_file():
        return {}
    out: dict = {}
    for line in states.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        for obj in state.get("objects", []):
            out.setdefault(obj["id"], {})[int(state["i"])] = list(obj["quat"])
    return out


MODES = {"bias": mode_bias, "offset": mode_offset, "signal": mode_signal, "icp": mode_icp,
         "register": mode_register, "articulation": mode_articulation}


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
    rows = merged.pop("rows", None)
    if rows:
        import os
        target = Path(a.work) / "rows.json"
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  逐物体记录写到 {target}")
    for key, values in merged.items():
        if not values:
            continue
        extra = (f"  均值 {statistics.mean(values):>6.3f}"
                 if key.endswith("fired") or key == "correct_kind" or "static_detector" in key else "")
        print(f"  {key:<26} 中位数 {statistics.median(values):>8.3f}   样本 {len(values)}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
