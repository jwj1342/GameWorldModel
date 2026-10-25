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
import random
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
from gwm.perception.evidence import smooth_series                                  # noqa: E402
from gwm.perception.registration import infer_articulation, kabsch, looks_static, track_rigid  # noqa: E402
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
    program = json.loads((clip_dir / "program.json").read_text(encoding="utf-8"))
    kinds = {o["id"]: (o.get("motion") or {}).get("type", "static")
             for o in program["objects"]}
    # **结构件也要算进来。** 第一版只读 program["objects"]，于是真值里的 static
    # 节点（这批数据有 69 个，比 objects 的 59 个还多）一个都没进测量，「静止物体」
    # 只剩 objects 里标了 static 的那 5 个——而静止转角正是最需要样本的那个量，
    # 结构件又是最纯粹的静止物体。n=5 的中位数对任何改动都不敏感：转动闸的接受率
    # 从 0.548 变到 0.355，rot_static 的中位数纹丝不动，看起来像「闸没用」。
    #
    # 「只给一半的场景打分」这个坑这个项目里记过一次，这里是第二次。
    for o in program.get("static") or []:
        kinds.setdefault(o["id"], "static")
    return truth, kinds


def perceive(clip_dir: Path, work: Path, perception: str = "gt") -> dict:
    """默认跑**真值感知**（真值掩码 + 真值深度）。

    `perception="gtseg"` 保住真值分割、把几何换成真实的 VGGT；`"neural"` 全换。为什么需要这个开关：残差挑资产的所有数据都是在
    真值感知上测的（33 个候选挑对 0.297、385 个 0.162），而管线跑的是神经感知，
    端到端命中率 **0/78**，和按衰减曲线算出的期望（约 5/76）对不上。

    残差判据比的是**表面细节**，而掩码溢出和深度噪声正好吃掉的就是这个。
    所以「残差挑资产成不成立」这个结论目前只在干净几何上成立，必须单独验一遍
    真实感知——这正是 [[synthetic-gains-need-a-real-data-leg]] 那条的同一类问题，
    只不过这次的「合成」是真值感知。
    """
    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"]
    if perception == "gt":
        files.append(REPO / "configs/perception/gt.yaml")
    elif perception == "gtcamseg":
        # **只把相机位姿换成真值**，深度仍走神经。真值分割照旧保住物体身份。
        # 用来分解真实几何那一轮的失效：残差挑从 0.297 掉到 0.041，是相机位姿
        # 还是深度在主导？线索是真值位姿模板在真实几何下仍然有效（15.07°/8.34°，
        # 基线 35.19°/19.30°）——点云还带着足够的形状信息，坏掉的是配准的条件数，
        # 而条件数最直接的杀手是相机位姿（点云正是靠它摆到世界坐标里的）。
        #
        # 反过来不行：真值深度是从真值相机渲出来的，深度和相机必须一起换
        # （见 configs/perception/gt_geometry.yaml 的注释）。
        files.append(REPO / "configs/perception/gt_segmentation.yaml")
        files.append(REPO / "configs/perception/gt_camera.yaml")
    elif perception == "gtseg":
        # 真值分割保住物体身份（否则 id 变成检测器的证据 id，真值资产 ref 对不上，
        # 残差那一臂根本不触发），几何换成真实的 VGGT。
        files.append(REPO / "configs/perception/gt_segmentation.yaml")
    cfg = load_config([f for f in files if Path(f).is_file()])
    cfg["perception"]["gt_program"] = str(clip_dir / "program.json")
    if perception == "gtcamseg":
        cfg["perception"]["gt_camera_program"] = str(clip_dir / "program.json")
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
        clouds, stamps, cams = [], [], []
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
                # 相机在世界坐标里的位置。配准那一侧原本完全没用上它——而可见面
                # 为什么变？因为相机动了。见 registration.visible_subset。
                cams.append(_fix(geom.cam_to_world[k])[:3, 3].copy())
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
        # 包围盒中心那条路本来就过一道时间平滑，配准这条没有。每步噪声是物体尺寸的 17%，
        # 而真实运动是平滑的——平滑应该能吃掉不少。窗口扫几个值看哪个合适。
        for window in (5, 9, 15):
            if len(registered) >= window:
                smoothed = smooth_series(registered, window)
                out.setdefault(f"step/smooth{window}", []).extend(
                    np.linalg.norm((smoothed - smoothed[0]) - gt_rel, axis=1).tolist())
                if is_static:
                    out.setdefault(f"drift_static/smooth{window}", []).append(
                        float(np.linalg.norm(smoothed.max(0) - smoothed.min(0))))
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
        # 把相机轨迹用进配准：每帧只对「它看得见的那部分模型」配。
        # 172d5ae 排除了「对完整模型配」，理由是部分对完整更欠定；但欠定的根源是
        # **没告诉它该对哪一部分**，而相机位置正好给出这个信息，且一直现成放着没用。
        # 转动闸：让转动「挣」出来，只有显著降低残差才认。扫几个门槛。
        # 位置那两个数已经被时间平滑治住了（0.045→0.028、0.077→0.039），
        # **平滑不影响转动**，所以 rot_static 才是现在挡住第三档的那一个数。
        gated_arms = []
        for gain in (0.05, 0.10, 0.15, 0.20, 0.30):
            g = track_rigid(clouds, rotation_min_gain=gain)
            if g["ok"] and len(g["positions"]) == len(gt):
                tag = f"gate{int(gain * 100):02d}"
                gp = np.array(g["positions"], float)
                out.setdefault(f"step/{tag}", []).extend(
                    np.linalg.norm((gp - gp[0]) - gt_rel, axis=1).tolist())
                # 和上面 smooth15 那一臂**同样的条件**：长度不够就不算，不要退回
                # 不平滑。否则两边的样本population 不同（848 对 1008），横着比是
                # 在比口径不是比方法。
                sm = smooth_series(gp, 15) if len(gp) >= 15 else None
                if sm is not None:
                    out.setdefault(f"step/{tag}_smooth15", []).extend(
                        np.linalg.norm((sm - sm[0]) - gt_rel, axis=1).tolist())
                if is_static:
                    out.setdefault(f"drift_static/{tag}", []).append(
                        float(np.linalg.norm(gp.max(0) - gp.min(0))))
                    if sm is not None:
                        out.setdefault(f"drift_static/{tag}_smooth15", []).append(
                            float(np.linalg.norm(sm.max(0) - sm.min(0))))
                gated_arms.append((tag, g))     # 转角那一段在外面统一记，这里只收集
                # 闸到底放行了多少步。三个门槛给出完全相同的 rot_static，说明增益
                # 分布离它们都很远——「几乎总放行」还是「几乎总拦住」不能靠猜。
                rate = g.get("rotation_gate_accept_rate")
                if rate is not None:
                    out.setdefault(f"gate_accept/{tag}", []).append(float(rate))

        vis = track_rigid(clouds, cam_positions=cams)
        if vis["ok"] and len(vis["positions"]) == len(gt):
            v_pos = np.array(vis["positions"], float)
            out.setdefault("step/visible", []).extend(
                np.linalg.norm((v_pos - v_pos[0]) - gt_rel, axis=1).tolist())
            if is_static:
                out.setdefault("drift_static/visible", []).append(
                    float(np.linalg.norm(v_pos.max(0) - v_pos.min(0))))

        # 第三个验收数：静止物体被解出多少转角。平滑治得了位置，治不了转动，
        # 所以这是现在挡住第三档的那一个数（铰链门槛 8 度，实测 13.5 度）。
        #
        # **这一段不能嵌在 vis 那个块里**——vis 失败时会把闸那几臂的转角也一起丢掉，
        # 而且丢得悄无声息。
        # 静止物体**和**会转的物体都要记。只报「静止物体的假转动被压下去了」而不报
        # 「真转动有没有被一起压住」，是只摆好处不摆代价——今天反复在提防的那类错误。
        #
        # 真值里带转动的类型：spin / revolute / periodic_rotate。平移类（linear、
        # trajectory）不算，它们本来就不该解出转动。
        gt_kind = kinds.get(oid, "static")
        rotating = any(k in str(gt_kind) for k in ("spin", "revolute", "rotate"))
        bucket = "rot_static" if is_static else ("rot_rotating" if rotating else "rot_translating")
        arms = [("register", tracked), *gated_arms]
        if vis.get("ok"):
            arms.append(("visible", vis))
        for tag, trk in arms:
            rot = np.asarray(trk.get("total_rotation") or np.eye(3), float)
            ang = float(np.degrees(np.arccos(
                np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))
            out.setdefault(f"{bucket}/{tag}", []).append(ang)

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
    poses = _gt_poses(run)
    if not gt_quat:
        return {}
    geom = load_geometry(run / "geometry.npz")
    masks = load_masks(run)
    frame_of = {f: i for i, f in enumerate(geom.frame_indices)}
    out: dict[str, list] = {}
    skipped: dict[str, int] = {}
    _gtref_paired: list = []            # 和 gtref 同一批物体上的基线值，见下面的注释
    for oid, per_frame in masks.items():
        kind = kinds.get(oid)
        series = gt_quat.get(oid)
        if kind is None or not series:
            skipped["no_truth_for_object"] = skipped.get("no_truth_for_object", 0) + 1
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
        # 第十条：**直接解运动模型参数**，不自由解每帧位姿（172d5ae 留下的另一条路）。
        # 前九条都是先自由解再事后修，而自由解本来就欠定，事后修只能修症状。
        # 这条把自由度从 6N 降到 3~7，转动只有在**所有帧上同时**降低残差时才赢。
        # 时间戳用帧序号即可——四个模型吃的是同一条时间轴，比的是相对残差。
        mfit = _model_fit_rotation_deg(clouds, list(range(len(clouds))))
        if mfit:
            gt_kind = kinds.get(oid, "static")
            out.setdefault(f"modelfit/{gt_kind}/pick_{mfit['pick']}", []).append(1.0)
            out.setdefault(f"modelfit/{gt_kind}/gain_over_static", []).append(
                float(mfit["gain_over_static"]))
        orientations = tracked.get("orientations") or []
        if len(orientations) != len(clouds):
            skipped["orientations_length_mismatch"] = skipped.get("orientations_length_mismatch", 0) + 1
            continue
        # 第二条路：用**完整模型**当模板，而不是让局部点云互相配。
        # 合成实验里这条路把假转动压小五倍而真转动不丢（#7），但那是完美的长方体模板。
        # 这里用真值位姿把各帧点云摆回物体自身坐标系、并起来当模型——它是个**预言机
        # 上界**（真实链路拿不到真值位姿），用来回答「完整性这个论点在真实数据上
        # 还成不成立」，再决定要不要去啃资产网格那条路。
        template_deg = _template_rotation_deg(clouds, frames, poses.get(oid) or {})
        # 第三条路，也是唯一能落地的一条：模板来自**检索到的资产网格**。
        # 前两条都不干净——局部点云互配的两边都随视角扫动；用真值位姿拼的「完整模型」
        # 本身是各帧图像格采样并起来的，格子的痕迹还在，而且第一版还把查询点云
        # 逐字喂进了模型（同义反复，已更正）。资产网格是从三角面上采的，和相机无关，
        # 也和观测完全独立。
        asset_deg = _asset_template_rotation_deg(clouds, _class_of(oid, evidence))
        # 第四条：模板用**对观测拟合出来的长方体**，不需要检索也不需要真值。
        prim_deg = _primitive_template_rotation_deg(clouds)
        # 同一个盒子模板，只换搭法：按逐帧包围盒的中位边长，而不是并帧点云。
        primmed_deg, box_oversize = _per_frame_box_rotation_deg(clouds)
        # 第五条，也是合成实验里唯一三条判据全过的那条：**不换模板，换被配的那两块**
        # ——把相邻两帧都裁到「两个相机都看得见」的公共表面再配。合成上限是
        # 静止 12.5°→0.0°、纯平移 26.9°→0.8°、真转 90° 保住 90.4°。
        # 这里在真实点云上验它，回答「即使前向模型完美，深度噪声会不会把收益吃掉」。
        cam_list = [_fix(geom.cam_to_world[frame_of[f]])[:3, 3] for f in frames
                    if frame_of.get(f) is not None]
        # _gt_poses 返回的是 {物体: {帧号: (pos, quat)}}，取 pos 那一半
        _p = poses.get(oid) or {}
        centres = {f: _p[f][0] for f in frames if f in _p}
        common_deg = (_common_visible_rotation_deg(clouds, frames, cam_list, centres)
                      if len(cam_list) == len(clouds) else np.array([]))
        # 唯一还没试过、而且直指机制的判别子：解出的转角跟不跟着相机的角步长走。
        # **链条的预言机：拿真值资产网格当模板。** 只预言检索，不预言位姿。
        # 回答「把点一修好之后，点二能拿到什么」——现在的 asset 那一臂量的是拿错的
        # 网格（检索 0/71），回答不了这个。
        # 创新点一的字面检验：残差能不能把真值资产从 N 个干扰项里挑出来。
        # 两个 N 一起报，好看出它随候选集增大怎么衰减。
        _gt_ref = _gt_asset_refs().get(oid, "")
        _yaw_deg = np.array([])    # 轮廓解出的位姿，在 true_deg 算出来之后才记
        _respick33 = None          # 下面按物体大小分桶时用，见 bysize/
        # **观测点云的三轴尺寸和真值差多少。** 噪声容差扫完之后结论是：VGGT 的深度
        # 误差**不是噪声**——连 5% 的各向同性噪声都只把挑对率压到 0.108，而真实
        # VGGT 是 0.041（随机 0.030）。衰减对噪声是平缓的，真实情况掉得狠得多，
        # 所以是**结构性**的：重建出来的表面不是正确形状的抖动版本，形状本身就错。
        #
        # 最直接的验证就是三轴尺寸比——压扁、和背景粘连都会在这上面显形。
        # 按降序比（不配对哪根轴对哪根），所以不受朝向影响。
        _gt_ext = _gt_node_extent(oid)
        if _gt_ext is not None and clouds:
            _obs = np.asarray(clouds[len(clouds) // 2], float)
            if len(_obs) >= 30:
                _o = np.sort(_obs.max(0) - _obs.min(0))[::-1]
                _g = np.sort(np.asarray(_gt_ext, float))[::-1]
                if _g.min() > 1e-9 and _o.min() > 1e-9:
                    # 先把整体尺度对齐（最长边归一），只看**比例**错多少——
                    # 绝对尺度错是另一回事，这里问的是形状有没有被压扁。
                    _r = (_o / _o[0]) / (_g / _g[0])
                    out.setdefault(f"{kind}/extent_ratio_mid", []).append(float(_r[1]))
                    out.setdefault(f"{kind}/extent_ratio_min", []).append(float(_r[2]))
                    out.setdefault(f"{kind}/extent_scale", []).append(float(_o[0] / _g[0]))
        if _gt_ref:
            # **这个物体自己的**真值资产在全库里的图像名次。之前报的 365 是集合判据
            # （片段里任何一个真值资产排得靠前就算），那是放水的上界；端到端要的是
            # 这个。两者差多少，直接决定「正确答案进不进候选集」。
            _rank = _image_rank_of_asset(run, evidence, oid, _gt_ref)
            if _rank is not None:
                out.setdefault(f"{kind}/own_image_rank", []).append(float(_rank))
                for _n in (64, 256, 384):
                    out.setdefault(f"{kind}/own_in_top{_n}", []).append(
                        1.0 if _rank <= _n else 0.0)
            for _n in (8, 32):
                _hit = _residual_pick_rate(clouds, _gt_ref, _n)
                if _hit is not None:
                    out.setdefault(f"{kind}/residual_pick_of_{_n + 1}", []).append(_hit)
            # 密度匹配的难干扰项：形状最接近真值的那 32 个。随机干扰项太好认，
            # 而真实方法面对的候选集个个都像。这一行才是可引用的数。
            # **候选集多大还成立。** 视角数固定 7（已验到顶），初值固定 1（加初值
            # 反而更差）。已知 9 个候选 0.459、33 个 0.311。图像嵌入把正确资产放在
            # 全库中位第 365 名——如果 128 个候选时还有两成，那就不用做两段式，
            # 直接把图像前 128 名全过一遍残差即可。衰减多快决定下一步的形状。
            # 深度噪声的容差线：真值深度 + 可控噪声，噪声按包围盒对角线的比例给。
            # 这条曲线是任何未来几何后端的**指标**，比「再试一个后端」有用。
            # **按观测位姿渲染候选轮廓**，和观测掩码比 IoU。只用相机朝向 +
            # 掩码，深度一点不碰——正好对上根因。
            _cams = {f: _fix(geom.cam_to_world[frame_of[f]])
                     for f in frames if frame_of.get(f) is not None}
            # **用轮廓解位姿**：每帧扫一圈偏航取 IoU 最高的。全程不碰深度，
            # 而今天所有位姿实验都建在深度点云的 ICP 上——这条一次都没试过。
            _yaw_deg = _silhouette_yaw_series(per_frame, frames, _cams, _gt_ref)
            # **真实形态**：图像提候选 + 轮廓挑。包含「正确答案不在候选里」这一项。
            for _nc in (64, 256):
                _cst: dict = {}
                _comb = _combined_pick_rate(run, evidence, per_frame, frames, _cams,
                                            oid, _gt_ref, n_cand=_nc, stats=_cst)
                if _comb is not None:
                    out.setdefault(f"{kind}/combined_top{_nc}", []).append(_comb)
                    out.setdefault(f"{kind}/gt_in_top{_nc}", []).append(
                        float(_cst.get("gt_in_cands", 0.0)))
            _rst: dict = {}
            _rend = _render_pick_rate(per_frame, frames, _cams, _gt_ref, 32,
                                      n_views=3, n_yaw=8, stats=_rst)
            # 掩码退化的容差线：真值掩码 + 膨胀/腐蚀若干格（96 格的轮廓上，
            # 2 格约等于物体宽度的 2%）。这条给出任何分割后端要达到的指标。
            for _mp in (1, 2, 4):
                _md = _render_pick_rate(per_frame, frames, _cams, _gt_ref, 32,
                                        n_views=3, n_yaw=8, mask_px=_mp)
                if _md is not None:
                    out.setdefault(f"{kind}/renderpick_mask{_mp}", []).append(_md)
            if _rend is not None:
                out.setdefault(f"{kind}/renderpick_33", []).append(_rend)
                out.setdefault(f"{kind}/render_margin", []).append(
                    float(_rst.get("margin", 0.0)))
                out.setdefault(f"{kind}/render_n_scored", []).append(
                    float(_rst.get("n_scored", 0)))
            # **轮廓判据**：绕开已坐实的根因（深度误差沿视线，轮廓是二维的）。
            for _c in (32, 128):
                _st: dict = {}
                _sil = _silhouette_pick_rate(per_frame, frames, _gt_ref, _c,
                                             n_views=7, stats=_st)
                if _sil is not None:
                    out.setdefault(f"{kind}/silpick_cands{_c + 1}", []).append(_sil)
                    # 自检：实际打分的候选数该等于 _c+1；margin 是第一名和第二名的
                    # 相对差，太大也可疑（说明真值那条有别人没有的东西）。
                    out.setdefault(f"{kind}/sil_n_scored{_c + 1}",
                                   []).append(float(_st.get("n_scored", 0)))
                    out.setdefault(f"{kind}/sil_margin{_c + 1}",
                                   []).append(float(_st.get("margin", 0.0)))
            # **错配零假设**：拿这个物体的掩码去配**另一个**物体的候选集。
            # 挑对率该掉到随机；掉不下去就说明判据里有泄漏。
            _other = _gt_other_ref(oid)
            if _other:
                _bad = _silhouette_pick_rate(per_frame, frames, _other, 32, n_views=7)
                if _bad is not None:
                    out.setdefault(f"{kind}/silpick_mismatched", []).append(_bad)
            # 把压扁补偿掉：沿最短主轴拉 1/0.526≈1.9 倍。1.0 是对照。
            for _uf in (1.0, 1.5, 1.9, 2.5):
                _u = _residual_pick_rate(clouds, _gt_ref, 32, hard=True,
                                         n_views=7, unflatten=_uf)
                if _u is not None:
                    out.setdefault(f"{kind}/respick_unflat{int(_uf * 10):02d}",
                                   []).append(_u)
            for _nz in (0.005, 0.01, 0.02, 0.05):
                _n = _residual_pick_rate(clouds, _gt_ref, 32, hard=True,
                                         n_views=7, noise=_nz)
                if _n is not None:
                    out.setdefault(f"{kind}/respick_noise{int(_nz * 1000):03d}", []).append(_n)
            for _c in (32, 128, 384):
                _hard = _residual_pick_rate(clouds, _gt_ref, _c, hard=True, n_views=7)
                if _hard is not None:
                    out.setdefault(f"{kind}/respick_cands{_c + 1}", []).append(_hard)
                    if _c == 32:
                        _respick33 = _hard
            # 判据换成 D2 形状分布，其余协议完全对齐（干扰项、候选数、视角数），
            # 所以可以和上面那一行直接并排比——唯一的变量是判据。
            for _c in (32, 128):
                _d2 = _d2_pick_rate(clouds, _gt_ref, _c, n_views=7)
                if _d2 is not None:
                    out.setdefault(f"{kind}/d2pick_cands{_c + 1}", []).append(_d2)
        gtref_deg = _template_rotation_from_points(
            clouds, _asset_points_by_id(_gt_asset_refs().get(oid, "")))
        # **形状第 k 近的邻居当模板。** 不用「管线实际选中的那个」是因为管线的物体
        # id 来自检测器、这里是真值节点 id，连起来要配对，而配对条件含被测量。
        #
        # 这条给出的是一条曲线：形状差多少 → 判别比掉多少。再把管线实测的形状距离
        # （残差挑之后中位 0.0440、上四分位 0.0844）往上一套，就知道「精确 id 挑不中」
        # 到底要不要紧。
        nbr_deg: dict = {}
        if _gt_ref:
            _nbrs = _hard_distractors(_gt_ref, 32)
            for _k in (1, 8, 32):
                if len(_nbrs) >= _k:
                    nbr_deg[_k] = _template_rotation_from_points(
                        clouds, _asset_points_by_id(_nbrs[_k - 1]))
                    _d = _shape_distance(_gt_ref, _nbrs[_k - 1])
                    if _d is not None:
                        out.setdefault(f"{kind}/nbr{_k}_shape_dist", []).append(_d)
        _cam_corr = _camera_step_correlation(tracked, cam_list)
        if _cam_corr is not None and np.isfinite(_cam_corr):
            out.setdefault(f"{kind}/cam_step_corr", []).append(float(_cam_corr))
        # 量具自检：每步相机转角本身变不变。太小的话上面那个相关系数是噪声。
        _cam_spread = _camera_step_spread(cam_list)
        if _cam_spread is not None and np.isfinite(_cam_spread):
            out.setdefault(f"{kind}/cam_step_spread", []).append(float(_cam_spread))
        # 机制的直接检验：假转动的幅度该随相机总角位移增长。
        # 比值稳定（散度小）才说明机制解释得了这个量；散开好几倍就解释不了。
        _cam_total = _camera_total_deg(cam_list)
        if _cam_total:
            out.setdefault(f"{kind}/cam_total_deg", []).append(float(_cam_total))
        # 第六条：同样裁公共可见区，但可见性用**真值基本体的解析法向**算。
        # 第五条用估计法向，真实数据上比现状还差（静止 RMSE 13.07°→19.71°）——
        # 那关闭的是「用估计法向近似可见性」，不是前向模型本身。这一条把法向估计
        # 这个误差源整个去掉，只留深度噪声，才回答得了「噪声本身会不会吃掉收益」。
        # 第七条，也是唯一真正去掉根因的：**跨帧特征点对应**（光流跟踪 + Kabsch），
        # 完全不用 ICP。前六条都在「拿什么去配」上做文章，而只要两边不是同一批物理
        # 点，那些都改变不了问题的性质。合成实验证明有精确对应就全好了。
        flow_deg = _flow_correspondence_rotation_deg(run, geom, clouds, frames, per_frame)
        # 第八条：稠密光流。稀疏角点在这个物体尺度上拿不到足够的点（掩码中位
        # 264 像素、角点中位 4 个，而阈值是 8），稠密给的是每个掩码像素一个对应。
        dense_deg = _dense_flow_rotation_deg(run, geom, frames, per_frame)
        # 第九条，也是对应关系这条路的**公平版**：锚点固定在第 0 帧，累积对应关系
        # 而不是累积旋转。前两条光流臂都在逐帧串，而 ICP 那侧用的是累积模型——
        # 两边基准设定不同，那组对比不公平。
        anchor_deg = _anchored_flow_rotation_deg(run, geom, frames, per_frame)
        _shapes = _gt_shapes_and_extents(CURRENT_CLIP_DIR) if CURRENT_CLIP_DIR else {}
        _node = oid if oid in _shapes else oid.rsplit("_", 1)[0]
        analytic_deg = np.array([])
        if _node in _shapes and len(cam_list) == len(clouds) and _p:
            _shape, _half = _shapes[_node]
            analytic_deg = _common_visible_analytic_deg(
                clouds, frames, cam_list, _p, _shape, _half)
        truth_q = [series.get(f) for f in frames]
        if any(q is None for q in truth_q):
            skipped["truth_missing_for_some_frame"] = skipped.get("truth_missing_for_some_frame", 0) + 1
            continue
        base = R.from_quat(truth_q[0]).inv()
        true_deg = np.degrees([(base * R.from_quat(q)).magnitude() for q in truth_q])
        solved_deg = np.degrees([R.from_matrix(np.asarray(m, float)).magnitude()
                                 for m in orientations])

        # **沿视线反压扁之后再配准**——动观测，不是换判据。
        # 位姿这一级所有判据都失败于同一根因：相机在动、逐帧采到的表面在变。
        # 而那个畸变量过了、有方向（沿视线压到 0.526），而且**方向逐帧不同**，
        # 拼起来正好长得像物体在转。沿各自的视线各自拉回来，这个不一致该被消掉。
        if len(cam_list) == len(clouds):
            for _f in (1.0, 1.5, 1.9):
                _uc = _unflatten_along_ray(clouds, cam_list, factor=_f)
                _ut = track_rigid(_uc)
                _uo = _ut.get("orientations") or []
                if len(_uo) != len(clouds):
                    continue
                _ud = np.degrees([R.from_matrix(np.asarray(m, float)).magnitude()
                                  for m in _uo])
                _tag = f"rayunflat{int(_f * 10):02d}"
                out.setdefault(f"{_tag}/{kind}/range_deg", []).append(float(np.ptp(_ud)))

        # **二次配准：用管线实际挑中的资产当模板，按文字分闸。**
        #
        # 模板路线一直卡在一个算术上：正确网格把静止/转动判别比从 2.60 提到 4.31，
        # 但检索命中只有 9%，整体开 = 9% 受益、91% 受害，必然净亏。
        # 实测文字分能闸出一个子集：最高 25%（score>=1.0）里**检索命中 0.625**
        # （整体 0.209），而且与裁剪尺寸无关（r=+0.038，不是「大物体」的代名词）。
        # 静止那一侧算过盈亏平衡点是 p*=0.312，0.625 富余两倍。
        #
        # 真转动那一侧**不能靠加权平均推**——模板对真转动是均匀阻尼，而判别比是
        # 人群统计量不是逐物体量。所以必须实测，这就是这一臂存在的理由。
        #
        # 配准跑在感知里、在检索**之前**，所以这是货真价实的「第二遍」：
        # 拿第一遍产出的 decisions 回头再配一次。两臂都**配对记基线**，
        # 免得比的是不同的物体。
        _pick = _chosen_assets(run).get(oid)
        if _pick:
            _pid, _pscore = _pick
            _ptpl = _asset_points_by_id(_pid)
            if _ptpl is not None:
                _pdeg = _template_rotation_from_points(clouds, _ptpl)
                if len(_pdeg) == len(true_deg):
                    _arm = "gated" if _pscore >= 0.999 else "open"
                    _pfx = f"pick2/{_arm}/{kind}"
                    out.setdefault(f"{_pfx}/tmpl_range", []).append(float(np.ptp(_pdeg)))
                    out.setdefault(f"{_pfx}/base_range", []).append(float(np.ptp(solved_deg)))
                    out.setdefault(f"{_pfx}/tmpl_rmse", []).append(
                        float(np.sqrt(np.mean((_pdeg - true_deg) ** 2))))
                    out.setdefault(f"{_pfx}/base_rmse", []).append(
                        float(np.sqrt(np.mean((solved_deg - true_deg) ** 2))))
                    out.setdefault(f"{_pfx}/true_range", []).append(float(np.ptp(true_deg)))
        # 物体在画面里多大，用来分桶。对应关系方法在小物体上必然差（孔径问题 +
        # 邻域平滑），所以报误差之前先把尺寸摆出来——掩码中位只有 264 像素时，
        # 「方法不行」和「物体太小」是两件事。
        _areas = [int(np.asarray(per_frame[f]).astype(bool).sum())
                  for f in frames if f in per_frame]
        _med_area = float(np.median(_areas)) if _areas else 0.0
        _bucket = ("小 <200px" if _med_area < 200 else
                   "中 200~600px" if _med_area < 600 else "大 >600px")
        out.setdefault(f"area/{_bucket}", []).append(_med_area)
        # **按物体大小分桶看残差挑对率。** 真实深度下残差和 D2 都掉到随机水平
        # （0.027 对随机 0.030），两条解释：VGGT 的深度本身不行，或者这批物体
        # 太小（stage_scale 0.15，掩码中位只有几百像素）已经到了它的分辨极限。
        # 后者最便宜也最该先排除——大物体上明显更高的话，问题是尺度不是后端。
        if _respick33 is not None:
            out.setdefault(f"bysize/{_bucket}/respick33", []).append(_respick33)
        if len(anchor_deg) == len(true_deg):
            out.setdefault(f"{kind}/anchor_range_deg", []).append(float(np.ptp(anchor_deg)))
            out.setdefault(f"{kind}/anchor_rmse_deg", []).append(
                float(np.sqrt(np.mean((anchor_deg - true_deg) ** 2))))
        if len(dense_deg) == len(true_deg):
            out.setdefault(f"bysize/{_bucket}/dense_rmse_deg", []).append(
                float(np.sqrt(np.mean((dense_deg - true_deg) ** 2))))
        out.setdefault(f"bysize/{_bucket}/icp_rmse_deg", []).append(
            float(np.sqrt(np.mean((solved_deg - true_deg) ** 2))))
        out.setdefault(f"{kind}/true_range_deg", []).append(float(np.ptp(true_deg)))
        out.setdefault(f"{kind}/solved_range_deg", []).append(float(np.ptp(solved_deg)))
        # 机制的直接检验：假转动的幅度该随相机总角位移增长。比值稳定（散度小）
        # 才说明机制解释得了这个量；散开好几倍就解释不了。
        # 放在这里是因为 solved_deg 到这一行才算出来。
        if _cam_total:
            out.setdefault(f"{kind}/rot_per_cam_deg", []).append(
                float(np.ptp(solved_deg) / _cam_total))
        out.setdefault(f"{kind}/angle_rmse_deg", []).append(
            float(np.sqrt(np.mean((solved_deg - true_deg) ** 2))))
        if len(template_deg) == len(true_deg):
            out.setdefault(f"{kind}/template_range_deg", []).append(float(np.ptp(template_deg)))
            out.setdefault(f"{kind}/template_rmse_deg", []).append(
                float(np.sqrt(np.mean((template_deg - true_deg) ** 2))))
        for _k, _deg in nbr_deg.items():
            if len(_deg) == len(true_deg):
                out.setdefault(f"{kind}/nbr{_k}_range_deg", []).append(float(np.ptp(_deg)))
                out.setdefault(f"{kind}/nbr{_k}_rmse_deg", []).append(
                    float(np.sqrt(np.mean((_deg - true_deg) ** 2))))
        # 轮廓解出的位姿。放在这里是因为 true_deg 到上面几行才算出来——
        # 第一版写在 _gt_ref 块里，那时 true_deg 还没定义。
        if len(_yaw_deg) == len(true_deg):
            out.setdefault(f"{kind}/silyaw_range_deg", []).append(float(np.ptp(_yaw_deg)))
            out.setdefault(f"{kind}/silyaw_rmse_deg", []).append(
                float(np.sqrt(np.mean((_yaw_deg - true_deg) ** 2))))
        if len(gtref_deg) == len(true_deg):
            out.setdefault(f"{kind}/gtref_range_deg", []).append(float(np.ptp(gtref_deg)))
            out.setdefault(f"{kind}/gtref_rmse_deg", []).append(
                float(np.sqrt(np.mean((gtref_deg - true_deg) ** 2))))
            # **配对口径。** gtref 只覆盖带资产 ref 的物体（60 条片段里静止桶 67 个
            # 物体只有 8 个是），拿它的中位数和全体 67 个的基线比是混口径——这个坑
            # 记过。把基线和逐帧盒子在**同一批物体**上的值单独记一份。
            _gtref_paired.append((float(np.ptp(solved_deg)),
                                  float(np.sqrt(np.mean((solved_deg - true_deg) ** 2))),
                                  float(np.ptp(primmed_deg)) if len(primmed_deg) == len(true_deg) else None,
                                  kind))
        if np.isfinite(box_oversize):
            out.setdefault(f"{kind}/box_oversize", []).append(float(box_oversize))
        if len(primmed_deg) == len(true_deg):
            out.setdefault(f"{kind}/primmed_range_deg", []).append(float(np.ptp(primmed_deg)))
            out.setdefault(f"{kind}/primmed_rmse_deg", []).append(
                float(np.sqrt(np.mean((primmed_deg - true_deg) ** 2))))
        if len(prim_deg) == len(true_deg):
            out.setdefault(f"{kind}/prim_range_deg", []).append(float(np.ptp(prim_deg)))
            out.setdefault(f"{kind}/prim_rmse_deg", []).append(
                float(np.sqrt(np.mean((prim_deg - true_deg) ** 2))))
        if len(asset_deg) == len(true_deg):
            out.setdefault(f"{kind}/asset_range_deg", []).append(float(np.ptp(asset_deg)))
            out.setdefault(f"{kind}/asset_rmse_deg", []).append(
                float(np.sqrt(np.mean((asset_deg - true_deg) ** 2))))
        if len(common_deg) == len(true_deg):
            out.setdefault(f"{kind}/common_range_deg", []).append(float(np.ptp(common_deg)))
            out.setdefault(f"{kind}/common_rmse_deg", []).append(
                float(np.sqrt(np.mean((common_deg - true_deg) ** 2))))
        if len(dense_deg) == len(true_deg):
            out.setdefault(f"{kind}/dense_range_deg", []).append(float(np.ptp(dense_deg)))
            out.setdefault(f"{kind}/dense_rmse_deg", []).append(
                float(np.sqrt(np.mean((dense_deg - true_deg) ** 2))))
        if len(flow_deg) == len(true_deg):
            out.setdefault(f"{kind}/flow_range_deg", []).append(float(np.ptp(flow_deg)))
            out.setdefault(f"{kind}/flow_rmse_deg", []).append(
                float(np.sqrt(np.mean((flow_deg - true_deg) ** 2))))
        if len(analytic_deg) == len(true_deg):
            out.setdefault(f"{kind}/analytic_range_deg", []).append(float(np.ptp(analytic_deg)))
            out.setdefault(f"{kind}/analytic_rmse_deg", []).append(
                float(np.sqrt(np.mean((analytic_deg - true_deg) ** 2))))
        if np.ptp(true_deg) > 1e-6 and np.ptp(solved_deg) > 1e-6:
            out.setdefault(f"{kind}/corr", []).append(
                float(np.corrcoef(solved_deg, true_deg)[0, 1]))
    # 把配对口径的那一份摊进 out。键名带 paired_ 前缀，和全体那一份区分开，
    # 免得有人（包括以后的我）把 gtref 的中位数和全体基线的中位数并排读。
    for s_range, s_rmse, p_range, k in _gtref_paired:
        out.setdefault(f"{k}/paired_solved_range_deg", []).append(s_range)
        out.setdefault(f"{k}/paired_solved_rmse_deg", []).append(s_rmse)
        if p_range is not None:
            out.setdefault(f"{k}/paired_primmed_range_deg", []).append(p_range)
    # 一个样本都没有的时候要说清楚卡在哪，别只打印「没有可比对的样本」
    if not out and skipped:
        print(f"  icp: 一个都没留下，原因 {skipped}", flush=True)
    return out


# 片段目录只有 main 知道，而 icp 模式要从里面读 gt_states.jsonl 里的逐帧朝向。
# 改 gt_series 的返回会动到另外四个模式，所以这里用一个模块级变量，main 每轮设一次。
CURRENT_CLIP_DIR: Path | None = None


def _template_rotation_deg(clouds: list, frames: list, poses: dict) -> np.ndarray:
    """用真值位姿拼出完整模型，再把每一帧配回去。预言机上界，不是可落地的做法。

    拼模型**必须用真值位置**，不能用每一帧点云自己的质心：局部视角下可见点的质心
    随视角移动（看正面和看侧面，可见点的质心不在同一处），按它归零拼出来的模型是
    糊的，配进去的每一帧还带一个和姿态相关的平移偏差，ICP 得拿旋转去吸收。
    第一版就是按质心拼的，于是 spin 从 114° 被压到 38°，而那是拼法的错不是方法的。
    不转的物体不受影响（姿态不变，质心偏差是常数，不产生旋转），所以当时假转动
    那一半的结论是稳的。

    配准这一步**不用真值**：每一帧的原始点云直接配到模型上，ICP 自己解平移和旋转，
    上一帧的解当这一帧的初值（暖启动，`track_rigid` 在另一条路上也是这么做的）。
    """
    from gwm.perception.registration import icp
    pieces = []
    for cloud, f in zip(clouds, frames):
        pose = poses.get(f)
        if pose is None:
            return np.array([])
        pos, quat = pose
        pieces.append((cloud - np.asarray(pos, float)) @ R.from_quat(quat).inv().as_matrix().T)
    model = np.vstack(pieces)
    if len(model) > 20000:                       # 配准前先抽稀，否则 KD 树太慢
        model = model[np.linspace(0, len(model) - 1, 20000).astype(int)]
    warm_R, warm_t = np.eye(3), np.zeros(3)
    out = []
    for cloud in clouds:
        moved = cloud @ warm_R.T + warm_t
        step = icp(moved, model)
        rot = np.asarray(step["rotation"], float)
        warm_R, warm_t = rot @ warm_R, rot @ warm_t + np.asarray(step["translation"], float)
        out.append(float(np.degrees(R.from_matrix(warm_R).magnitude())))
    return np.asarray(out)


def _class_of(oid: str, evidence: dict) -> str:
    for o in evidence.get("objects", []):
        if o.get("id") == oid:
            return str(o.get("class_guess") or "")
    return ""


_ASSET_CACHE: dict = {}


def _asset_points(query: str) -> np.ndarray | None:
    """按类名检索资产，把网格表面采成点云。检索不到或读不动就返回 None。"""
    if not query:
        return None
    if query in _ASSET_CACHE:
        return _ASSET_CACHE[query]
    from gwm.assets.library import library_root, load_catalog
    from gwm.assets.mesh import load_points
    from gwm.assets.retrieve import retrieve

    root = library_root()
    if root is None:
        _ASSET_CACHE[query] = None; return None
    catalog = _ASSET_CACHE.setdefault("__catalog__", load_catalog(root))
    hits = retrieve(query, catalog, k=1)
    pts = load_points(Path(root) / hits[0][0].path, samples=3000) if hits else None
    _ASSET_CACHE[query] = pts
    return pts


_GT_REF_CACHE: dict = {}


def _gt_node_extent(oid: str):
    """真值节点的三轴尺寸（从程序的 geom.extent 读）。配对不需要——真值感知下
    物体 id 就是真值节点 id。"""
    clip_dir = globals().get("CURRENT_CLIP_DIR")
    if clip_dir is None:
        return None
    key = f"ext:{clip_dir}"
    if key not in _GT_REF_CACHE:
        out = {}
        try:
            program = json.loads((Path(clip_dir) / "program.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            program = {}
        for node in (program.get("objects") or []) + (program.get("static") or []):
            ext = (node.get("geom") or {}).get("extent")
            if ext and len(ext) == 3:
                out[str(node.get("id"))] = ext
        _GT_REF_CACHE[key] = out
    return _GT_REF_CACHE[key].get(oid)


def _gt_asset_refs() -> dict:
    """当前片段里每个真值节点挑的是哪个资产（`geom.ref`）。

    数据集用 `--use-assets` 从同一个库搭之后，真值资产 id 是已知的，于是可以问一个
    之前问不了的问题：**假如检索全对，模板路线能不能成立**。

    这是个只预言「检索」不预言「位姿」的预言机——比 `_template_rotation_deg`
    那个用真值位姿拼模型的弱得多，也更接近「把点一修好之后能拿到什么」。
    现在管线的检索是 0/71 全错，所以 `asset_range_deg` 那一臂量的是**拿错的网格
    当模板**，回答不了这个问题。
    """
    clip_dir = globals().get("CURRENT_CLIP_DIR")
    if clip_dir is None:
        return {}
    key = str(clip_dir)
    if key in _GT_REF_CACHE:
        return _GT_REF_CACHE[key]
    out: dict = {}
    try:
        program = json.loads((Path(clip_dir) / "program.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _GT_REF_CACHE[key] = out
        return out
    for node in (program.get("objects") or []) + (program.get("static") or []):
        geom = node.get("geom") or {}
        if geom.get("kind") in ("asset", "generated") and geom.get("ref"):
            out[str(node.get("id"))] = str(geom["ref"])
    _GT_REF_CACHE[key] = out
    return out


def _asset_points_by_id(asset_id: str, samples: int = 3000) -> np.ndarray | None:
    """按**资产 id**取网格点云（检索那条是按类名查的，会查错）。

    `samples` 给轮廓那条用：96×96 的格子有 9216 个，3000 个点填不实，
    轮廓会是散点而不是实心区域，IoU 同样失真。
    """
    if not asset_id:
        return None
    key = f"__id__{asset_id}:{samples}"
    if key in _ASSET_CACHE:
        return _ASSET_CACHE[key]
    from gwm.assets.library import library_root, load_catalog
    from gwm.assets.mesh import load_points

    root = library_root()
    if root is None:
        _ASSET_CACHE[key] = None; return None
    catalog = _ASSET_CACHE.setdefault("__catalog__", load_catalog(root))
    match = next((a for a in catalog if a.id == asset_id), None)
    pts = load_points(Path(root) / match.path, samples=samples) if match else None
    _ASSET_CACHE[key] = pts
    return pts


def _template_rotation_from_points(clouds: list, template) -> np.ndarray:
    """把给定模板缩到**逐帧包围盒中位对角线**上，逐帧暖启动配准，返回累积转角序列。

    缩到逐帧中位而不是并帧对角线：并帧点云随视角扫动，实测大 1.89 倍，而受控扫描
    说模板相对点云大 1.2 倍就越过对照、1.3 倍崩掉。这个坑踩过一次
    （资产模板 RMSE 80.92°，改掉之后 32.54°）。
    """
    from gwm.perception.registration import icp
    if template is None or not clouds:
        return np.array([])
    per_frame = [np.linalg.norm(np.asarray(c, float).max(0) - np.asarray(c, float).min(0))
                 for c in clouds if len(np.asarray(c, float)) >= 10]
    tpl_diag = float(np.linalg.norm(template.max(0) - template.min(0)))
    if not per_frame or tpl_diag <= 1e-9:
        return np.array([])
    obs_diag = float(np.median(per_frame))
    model = (template - template.mean(0)) * (obs_diag / tpl_diag)
    warm_R, warm_t = np.eye(3), np.zeros(3)
    out = []
    for cloud in clouds:
        moved = cloud @ warm_R.T + warm_t
        step = icp(moved, model)
        rot = np.asarray(step["rotation"], float)
        warm_R, warm_t = rot @ warm_R, rot @ warm_t + np.asarray(step["translation"], float)
        out.append(float(np.degrees(R.from_matrix(warm_R).magnitude())))
    return np.asarray(out)


def _fitted_primitive_points(clouds: list, n: int = 4000, seed: int = 0) -> np.ndarray | None:
    """对观测点云拟合一个长方体，在它**六个面**上均匀采点，当 ICP 的模板。

    为什么试这个（#7）。完整模板确实管用——用真值位姿拼出来的预言机模板把 static
    的角度 RMSE 从 12.53° 压到 7.00°。但唯一能落地的模板源（检索到的资产网格）实测
    反而差一个量级（85.24°），因为这个数据集的真值**全是基本体**，资产库里没有
    spool / totem / beacon 这些类，拿错形状的网格去配只会解出噪声。

    所以换一个**不需要检索、也不需要真值**的模板：直接按观测点云的包围盒造一个
    闭合长方体。赌的是预言机那一行暗示的东西——ICP 需要的是「有一个闭合、不随视角
    扫动的表面可对」，**不一定需要形状种类判对**。真值里盒子本来也占 28/59。

    **量完了：不成立，而且比资产模板还差。** 真值静止的物体（真转角 0°）：

        现状（局部互配）        角度 RMSE 12.53°
        预言机模板（真值位姿）              7.00°
        资产网格模板                       85.24°
        **拟合长方体模板**                 **96.74°**

    原因想明白了：这个长方体是**所有帧并起来**的点云的包围盒，而那团点云本身是
    随视角扫动的——包围盒因此比物体大一圈。拿单帧的局部点云去配一个过大的闭合盒子，
    解出来的旋转比不用模板还野。

    合起来看，三种模板源给出一个干净的结论：**「完整」这个论点只在模板就是物体
    真实形状时成立**（预言机那一行），任何近似（检索的网格、拟合的盒子）都比不用
    模板更糟。而拿到真实形状，正是我们一开始要解的问题——这是个循环依赖。

    函数留着，因为它是那条结论的证据。
    """
    if not clouds:
        return None
    span = np.vstack(clouds)
    lo, hi = span.min(0), span.max(0)
    size = hi - lo
    if (size <= 1e-6).any():
        return None
    rng = np.random.default_rng(seed)
    per = max(n // 6, 1)
    faces = []
    for axis in range(3):
        for end in (0.0, 1.0):
            uv = rng.random((per, 3))
            uv[:, axis] = end
            faces.append(lo + uv * size)
    pts = np.vstack(faces)
    return pts - pts.mean(0)








def _flow_correspondence_rotation_deg(run, geom, clouds: list, frames: list,
                                      masks_per_frame: dict) -> np.ndarray:
    """**跨帧特征点对应**解刚体变换，完全不用 ICP（#79 的小版本）。

    为什么是这条。当天关掉的六条修法里有五条都在「拿什么去配」或「配哪一块」上做
    文章，全部失败；而合成实验证明**有精确对应关系就全好了**（静止 0.0°、纯平移
    0.8°、真转 90° 保住 90.4°）。根因一直是同一句话——

        采样格子钉在**图像**上不钉在物体上，相机一动，两帧之间连一个共同物理点都没有。

    ICP 用最近邻**猜**对应，所以它改变不了这件事。光流跟踪的是**同一个物理点**，
    这是六条路里唯一真正去掉根因的做法。

    流程：掩码内取角点 → Lucas-Kanade 跟到下一帧 → 两端各用自己那帧的深度抬到三维
    → Kabsch 闭式解 (R, t)。对应关系是**跟出来的**，不是猜的。

    已知的弱点，先写在这里免得又把弱版本当强版本（当天栽过两次）：
      * 光流在低纹理表面上会失败——合成渲染的物体恰恰纹理很少；
      * 反向跟踪一致性只是必要条件，不保证对；
      * 抬到三维用的还是同一张深度图，深度噪声没有被去掉。
    所以这是「对应关系这条路值不值得投」的第一个数，不是落地方案。
    """
    import cv2
    frame_dir = Path(run) / "geom_frames"
    files = sorted(frame_dir.glob("g_*.jpg"))
    if len(files) < 2:
        return np.array([])
    idx_of = {f: i for i, f in enumerate(geom.frame_indices)}
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    R_acc = np.eye(3)
    out = [0.0]
    for i in range(1, len(frames)):
        ka, kb = idx_of.get(frames[i - 1]), idx_of.get(frames[i])
        if ka is None or kb is None or ka >= len(files) or kb >= len(files):
            out.append(out[-1]); continue
        ga = cv2.imread(str(files[ka]), cv2.IMREAD_GRAYSCALE)
        gb = cv2.imread(str(files[kb]), cv2.IMREAD_GRAYSCALE)
        if ga is None or gb is None:
            out.append(out[-1]); continue
        m = masks_per_frame.get(frames[i - 1])
        if m is None:
            out.append(out[-1]); continue
        m = np.asarray(m).astype(np.uint8)
        if m.shape != ga.shape:
            m = cv2.resize(m, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_NEAREST)
        pts = cv2.goodFeaturesToTrack(ga, maxCorners=400, qualityLevel=0.01,
                                      minDistance=4, mask=m)
        if pts is None or len(pts) < 8:
            out.append(out[-1]); continue
        nxt, st, _err = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, nxt, None, **lk)
        # 反向跟踪一致性：跟过去再跟回来，落点偏得远的扔掉
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & \
             (np.linalg.norm(pts - back, axis=2).ravel() < 1.0)
        if ok.sum() < 8:
            out.append(out[-1]); continue
        pa, pb = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
        P = _lift(geom, ka, pa)
        Q = _lift(geom, kb, pb)
        good = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
        if good.sum() < 8:
            out.append(out[-1]); continue
        rot, _t = kabsch(Q[good], P[good])       # 把第 i 帧配回第 i-1 帧
        R_acc = np.asarray(rot, float) @ R_acc
        out.append(float(np.degrees(R.from_matrix(R_acc).magnitude())))
    return np.asarray(out)






def _anchored_flow_rotation_deg(run, geom, frames: list, masks_per_frame: dict) -> np.ndarray:
    """跨帧对应的**公平版**：累积对应关系，不累积旋转。

    之前那两条光流臂（稀疏、稠密）都是逐帧链式累积旋转 `R_acc = rot @ R_acc`。
    而 `track_rigid` 的 docstring 早就写明——

        **不逐帧链式配准**。实测链式每步残差只有 0.026 m，轨迹误差却比包围盒中心
        还大——每步的小误差沿三十帧累积起来就盖过了信号。

    仓库已经查明链式不行并改成了「对累积模型配准」，而我给光流臂写的正是链式。
    `dense_range_deg` 稳定在 170°（几乎跑满量程）就是累积漂移的典型征兆。
    **所以那组对比不公平**：ICP 那侧用累积模型，光流这侧在串。

    这一版把锚点固定在第 0 帧：点从第 0 帧一路跟下去（跟踪本身是逐帧的，没办法），
    但到第 k 帧时拿**第 0 帧的三维点和第 k 帧的三维点直接解一次 Kabsch**。
    旋转每次都是从直接对应关系新解的，不沿帧累加——这才是和 ICP 可比的设定。

    跟丢的点会越来越多，所以同时返回「还剩多少点」，点太少时后面的数不可信。
    """
    import cv2
    from gwm.perception.masks import resize_mask
    frame_dir = Path(run) / "geom_frames"
    files = sorted(frame_dir.glob("g_*.jpg"))
    if len(files) < 2:
        return np.array([])
    idx_of = {f: i for i, f in enumerate(geom.frame_indices)}
    k0 = idx_of.get(frames[0])
    m0 = masks_per_frame.get(frames[0])
    if k0 is None or k0 >= len(files) or m0 is None:
        return np.array([])
    g0 = cv2.imread(str(files[k0]), cv2.IMREAD_GRAYSCALE)
    if g0 is None:
        return np.array([])
    m0 = np.asarray(m0).astype(bool)
    if m0.shape != g0.shape:
        m0 = resize_mask(m0, (g0.shape[1], g0.shape[0])).astype(bool)
    if m0.sum() < 40:
        return np.array([])
    # **只跟角点，不跟所有掩码像素。** 前两版各带一个互补的毛病：稀疏版只跟角点
    # 但物体只有 4 个角点（点太少）；第一版锚定跟掩码内所有像素，点够多但大部分是
    # 平坦区域，LK 在那上面必然漂移（孔径问题）。正确的组合是角点 + 锚定。
    # 质量阈值放松到 0.003，因为物体小、角点本来就少（普查：中位 4 个）。
    corners = cv2.goodFeaturesToTrack(g0, maxCorners=600, qualityLevel=0.003,
                                      minDistance=2, mask=m0.astype(np.uint8))
    if corners is None or len(corners) < 6:
        return np.array([])
    pts0 = corners.reshape(-1, 2).astype(np.float32)
    P0 = _lift(geom, k0, pts0)
    alive = np.isfinite(P0).all(1)
    if alive.sum() < 6:
        return np.array([])
    lk = dict(winSize=(15, 15), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    cur = pts0.reshape(-1, 1, 2).copy()
    prev_img, prev_k = g0, k0
    out = [0.0]
    for i in range(1, len(frames)):
        ki = idx_of.get(frames[i])
        if ki is None or ki >= len(files):
            out.append(out[-1]); continue
        gi = cv2.imread(str(files[ki]), cv2.IMREAD_GRAYSCALE)
        if gi is None:
            out.append(out[-1]); continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, gi, cur, None, **lk)
        alive = alive & (st.ravel() == 1)
        cur, prev_img, prev_k = nxt, gi, ki
        if alive.sum() < 6:
            out.append(out[-1]); continue
        Pk = _lift(geom, ki, cur.reshape(-1, 2))
        good = alive & np.isfinite(Pk).all(1)
        if good.sum() < 6:
            out.append(out[-1]); continue
        # **锚点在第 0 帧**：每次都从直接对应关系解一次，不累加旋转
        A, B = P0[good], Pk[good]
        rot, t = kabsch(B, A)
        res = np.linalg.norm(B @ np.asarray(rot, float).T + t - A, axis=1)
        keep = res <= np.percentile(res, 75)
        if keep.sum() >= 6:
            rot, _t = kabsch(B[keep], A[keep])
        out.append(float(np.degrees(R.from_matrix(np.asarray(rot, float)).magnitude())))
    return np.asarray(out)

def _dense_flow_rotation_deg(run, geom, frames: list, masks_per_frame: dict) -> np.ndarray:
    """**稠密光流**的跨帧对应 + Kabsch。稀疏角点在这个物体尺度上不够用。

    为什么换稠密。稀疏版（goodFeaturesToTrack + LK）两个方向同时崩：静止解出 84.9°、
    真转 114° 的 spin 解出 0.000°。我第一反应是「合成渲染没纹理」，**数出来是错的**
    ——587 个物体-帧样本里只有 6.6% 完全没角点，角点数中位 **4.0**，而掩码面积中位
    只有 **264 像素**（约 16×16）。真因是**物体在画面里太小**，4 个角点过不了
    `>= 8` 那道门，于是每帧都走「跟不到就沿用上一个值」。

    稠密光流给的是掩码内**每个像素**的对应——264 个像素就是 264 个对应。
    点数不再是瓶颈，而且 Kabsch 可以配上剔除。

    注意不要把阈值从 8 降到 3 来「解决」稀疏版的问题：4 个点解 Kabsch 对噪声极敏感，
    那是换一种方式失败。
    """
    import cv2
    from gwm.perception.masks import resize_mask
    frame_dir = Path(run) / "geom_frames"
    files = sorted(frame_dir.glob("g_*.jpg"))
    if len(files) < 2:
        return np.array([])
    idx_of = {f: i for i, f in enumerate(geom.frame_indices)}
    R_acc = np.eye(3)
    out = [0.0]
    for i in range(1, len(frames)):
        ka, kb = idx_of.get(frames[i - 1]), idx_of.get(frames[i])
        if ka is None or kb is None or ka >= len(files) or kb >= len(files):
            out.append(out[-1]); continue
        ga = cv2.imread(str(files[ka]), cv2.IMREAD_GRAYSCALE)
        gb = cv2.imread(str(files[kb]), cv2.IMREAD_GRAYSCALE)
        m = masks_per_frame.get(frames[i - 1])
        if ga is None or gb is None or m is None:
            out.append(out[-1]); continue
        m = np.asarray(m).astype(bool)
        if m.shape != ga.shape:
            m = resize_mask(m, (ga.shape[1], ga.shape[0])).astype(bool)
        # **把掩码往里腐蚀再取点。** 物体只有 20~36 像素宽，很大一部分像素离边界只有
        # 一两个像素；光流偏一个像素，_lift 在那个落点取到的深度就跳到背景上，三维点
        # 直接飞掉。所以坏的可能不是光流，是「二维对应 + 逐像素查深度」这个抬升方式。
        # 腐蚀之后只用物体内部的点，这个嫌疑就被排除或坐实。
        m_in = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        if m_in.sum() >= 40:
            m = m_in
        if m.sum() < 40:
            out.append(out[-1]); continue
        # **窗口要和物体尺度匹配。** 第一版用 winsize=15，而物体只有 20~36 像素宽——
        # 平滑窗口和物体本身一样大，算出来的流混进大量背景，跟到的是相机流不是物体流。
        # 实测那一版误差 86~95 度且**不随物体变大而改善**，正是这个征兆。
        # 窗口按掩码的等效边长定，无量纲地跟着物体走，不写死。
        side = int(np.sqrt(max(int(m.sum()), 1)))
        win = int(np.clip(side // 3, 5, 15)) | 1          # 奇数，物体三分之一左右
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, win, 3, 5, 1.2, 0)
        vs, us = np.nonzero(m)
        pa = np.stack([us.astype(np.float32), vs.astype(np.float32)], 1)
        pb = pa + flow[vs, us]
        P = _lift(geom, ka, pa)
        Q = _lift(geom, kb, pb)
        good = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
        if good.sum() < 20:
            out.append(out[-1]); continue
        P, Q = P[good], Q[good]
        # 一轮剔除：先解一次，把残差大的四分之一扔掉再解一次
        rot, t = kabsch(Q, P)
        res = np.linalg.norm(Q @ np.asarray(rot, float).T + t - P, axis=1)
        keep = res <= np.percentile(res, 75)
        if keep.sum() >= 20:
            rot, _t = kabsch(Q[keep], P[keep])
        R_acc = np.asarray(rot, float) @ R_acc
        out.append(float(np.degrees(R.from_matrix(R_acc).magnitude())))
    return np.asarray(out)

def _lift(geom, k: int, px: np.ndarray) -> np.ndarray:
    """像素坐标 + 该帧深度 → 世界坐标。深度图是几何后端那一档的分辨率。"""
    depth = geom.depth[k]
    h, w = depth.shape
    u = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
    z = depth[v, u]
    K = geom.intrinsics[k]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    out = to_world(cam, _fix(geom.cam_to_world[k]))
    out[z <= 1e-4] = np.nan
    return out

def _gt_shapes_and_extents(clip_dir) -> dict:
    """真值节点 id → (基本体种类, 半长 3 元组)。解析法向要用。"""
    out = {}
    try:
        program = json.loads((clip_dir / "program.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return out
    for node in (program.get("objects") or []) + (program.get("static") or []):
        geom = node.get("geom") or {}
        if geom.get("kind") != "primitive":
            continue
        shape = str(geom.get("shape") or "box")
        ext = geom.get("extent")
        if isinstance(ext, list) and len(ext) == 3:
            half = np.asarray(ext, float) / 2.0
        else:
            r = float(geom.get("radius") or 0.0)
            h = float(geom.get("height") or (2 * r))
            half = np.array([r, h / 2.0, r], float)
        if (half > 0).all():
            out[str(node.get("id"))] = (shape, half)
    return out


def _analytic_normals(pts_obj: np.ndarray, shape: str, half: np.ndarray) -> np.ndarray:
    """基本体表面的**解析**外法向（物体坐标系）。

    为什么必须解析算。上一版从带噪点云的邻域估法向，结果真实数据上比现状还差
    （静止 RMSE 13.07° → 19.71°）——噪声进了法向就裁错了面。那个否定结果关闭的是
    「用估计法向近似可见性」，**不是前向模型本身**。这里把法向估计这个误差源整个
    去掉，只留深度噪声，才回答得了「噪声本身会不会吃掉收益」。
    """
    p = np.asarray(pts_obj, float)
    if shape == "sphere":
        n = p.copy()
    elif shape in ("cylinder", "cone"):
        # 侧面是径向的；靠近端盖的点按 y 给法向
        radial = np.stack([p[:, 0], np.zeros(len(p)), p[:, 2]], 1)
        rad_len = np.linalg.norm(radial, axis=1)
        on_cap = np.abs(p[:, 1]) / max(half[1], 1e-9) > (rad_len / max(half[0], 1e-9))
        n = radial
        n[on_cap] = np.stack([np.zeros(on_cap.sum()),
                              np.sign(p[on_cap, 1]),
                              np.zeros(on_cap.sum())], 1)
        if shape == "cone":                      # 锥面法向往外上方倾
            slope = half[0] / max(2 * half[1], 1e-9)
            n[~on_cap, 1] = rad_len[~on_cap] * slope
    else:                                        # box：离哪个面最近就用那个面的法向
        rel = np.abs(p) / np.maximum(half, 1e-9)
        axis = np.argmax(rel, axis=1)
        n = np.zeros_like(p)
        n[np.arange(len(p)), axis] = np.sign(p[np.arange(len(p)), axis])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-9)


def _common_visible_analytic_deg(clouds: list, frames: list, cams: list,
                                 poses_obj: dict, shape: str, half: np.ndarray) -> np.ndarray:
    """公共可见区裁剪，可见性用**解析法向**算——干净的预言机版本。

    和 `_common_visible_rotation_deg` 唯一的差别就是法向从哪来，所以并排比能干净地
    回答「上一版的失败是法向估计的锅，还是深度噪声的锅」。
    """
    from gwm.perception.registration import icp
    if len(clouds) < 2 or len(cams) != len(clouds):
        return np.array([])

    def facing(cloud, f, eye):
        entry = poses_obj.get(f)
        if entry is None:
            return np.ones(len(cloud), bool)
        pos, quat = entry
        rot = R.from_quat(quat)
        obj = rot.inv().apply(np.asarray(cloud, float) - np.asarray(pos, float))
        n_world = rot.apply(_analytic_normals(obj, shape, half))
        view = np.asarray(eye, float) - cloud
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
        return np.einsum("ij,ij->i", n_world, view) > 0.05

    R_acc = np.eye(3)
    out = [0.0]
    for i in range(1, len(clouds)):
        keep_prev = facing(clouds[i - 1], frames[i - 1], cams[i - 1]) & \
            facing(clouds[i - 1], frames[i - 1], cams[i])
        keep_cur = facing(clouds[i], frames[i], cams[i]) & \
            facing(clouds[i], frames[i], cams[i - 1])
        src, dst = clouds[i][keep_cur], clouds[i - 1][keep_prev]
        if len(src) < 30 or len(dst) < 30:
            out.append(out[-1]); continue
        step = icp(src, dst)
        if step.get("ok"):
            R_acc = np.asarray(step["rotation"], float) @ R_acc
        out.append(float(np.degrees(R.from_matrix(R_acc).magnitude())))
    return np.asarray(out)



def _model_fit_rotation_deg(clouds: list, stamps: list) -> dict:
    """**直接解运动模型参数**，而不是自由地解每帧位姿——`172d5ae` 留下的另一条路。

    今天关掉的九条修法有个共同形状：都是先自由解出每帧位姿（6N 个自由度），
    再事后去修那个解（换模板、裁可见区、加转动闸、换对应关系）。而根因是

        采样格子钉在图像上，两帧之间连一个共同物理点都没有，

    所以自由解本来就是欠定的，**事后修只能修症状**。

    这条换一个方向：把整条轨迹参数化成一个运动模型，直接解模型的几个参数。
    自由度从 6N 降到 3~7，欠定就消失了。具体四个模型：

      static     一个位姿管所有帧；**按构造零转动**
      linear     p(t) = p0 + v·t，零转动
      oscillate  p(t) = p0 + A·sin(2πt/T + φ)，零转动
      spin       绕竖轴匀速转，角速度 ω 一个参数

    关键差别：转动只有在**所有帧上同时**降低残差时才赢得过 static，
    不是每一步各自赢。我那个逐步的转动闸压不住的那部分（0.588 饱和），
    正是因为它逐步判，而假转动恰恰是逐步都「说得通」的。

    返回每个模型的残差和选中的那个。残差口径统一：把模型预测的位置和**逐帧点云
    质心**比——质心有偏（落在可见面上），但四个模型吃的是同一个偏，所以可比。
    """
    if len(clouds) < 6 or len(stamps) != len(clouds):
        return {}
    t = np.asarray(stamps, float)
    t = t - t[0]
    c = np.asarray([np.asarray(p, float).mean(0) for p in clouds], float)

    def rmse(pred):
        return float(np.sqrt(np.mean(np.sum((c - pred) ** 2, axis=1))))

    out: dict[str, float] = {}
    out["static"] = rmse(np.tile(c.mean(0), (len(c), 1)))
    # 直线：每个坐标轴上一次多项式
    A = np.stack([np.ones_like(t), t], 1)
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    out["linear"] = rmse(A @ coef)
    # 往复：扫周期，每个周期下 sin/cos 是线性的，所以内层仍是最小二乘
    best = float("inf")
    span = float(t[-1] - t[0]) or 1.0
    for period in np.linspace(span / 4, span * 1.5, 24):
        w = 2 * np.pi / period
        B = np.stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)], 1)
        k, *_ = np.linalg.lstsq(B, c, rcond=None)
        best = min(best, rmse(B @ k))
    out["oscillate"] = best
    # 绕竖轴匀速转：质心绕某个中心转圈，所以也是「中心 + 半径 + 角速度」
    best_spin = float("inf")
    centre = c.mean(0)
    for omega in np.linspace(0.1, 6.0, 24):
        B = np.stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)], 1)
        k, *_ = np.linalg.lstsq(B, c - centre, rcond=None)
        best_spin = min(best_spin, rmse(centre + B @ k))
    out["spin"] = best_spin
    pick = min(out, key=out.get)
    return {"residuals": {k: round(v, 5) for k, v in out.items()},
            "pick": pick,
            # **零假设**：static 比最好的那个差多少。差得少就该答 static——
            # 今天反复吃亏的地方就是没有并排摆基线。
            "gain_over_static": round((out["static"] - out[pick]) / max(out["static"], 1e-9), 4)}

def _common_visible_rotation_deg(clouds: list, frames: list, cams: list,
                                 centres: dict) -> np.ndarray:
    """**前向模型在真实数据上的预言机**：相邻两帧都裁到「两个相机都看得见」的面再配。

    合成实验（`probe_icp_visibility.py`）里这条是今天唯一三条判据全过的：
    静止 12.5°→0.0°、纯平移 26.9°→0.8°、而真转 90° 保住 90.4°。但那是**合成上限**
    ——可见性精确可算、深度无噪声。模板路线就是死在这一关（合成 5.2°、落地 81°），
    所以不在真实点云上验，前面那个结果什么都不说明。

    这里要回答的是一个更窄的问题：**即使前向模型完美，真实深度的噪声会不会把收益
    吃掉。** 所以可见性仍然用真值（物体中心来自 gt_states，法向由点云邻域估出再朝
    这个中心的外侧定向），而点云是真实深度 + 真实掩码得到的。

    法向定向必须用**物体中心**，不能用点云自己的质心——部分可见的点云质心落在表面
    上，「朝外」没有定义。这个坑今天踩过两次（柱面轴心、可见子集配准）。
    """
    from gwm.perception.registration import estimate_normals, icp
    if len(clouds) < 2 or len(cams) != len(clouds):
        return np.array([])

    def facing(cloud, centre, eye):
        if len(cloud) < 16:
            return np.ones(len(cloud), bool)
        n = estimate_normals(cloud, k=min(12, len(cloud) - 1))
        outward = cloud - centre                       # ← 物体中心，不是点云质心
        flip = np.sign(np.einsum("ij,ij->i", n, outward))
        flip[flip == 0] = 1.0
        n = n * flip[:, None]
        view = eye - cloud
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
        return np.einsum("ij,ij->i", n, view) > 0.05

    R_acc = np.eye(3)
    out = [0.0]
    for i in range(1, len(clouds)):
        c_prev = np.asarray(centres.get(frames[i - 1], clouds[i - 1].mean(0)), float)
        c_cur = np.asarray(centres.get(frames[i], clouds[i].mean(0)), float)
        # 两帧各自裁到「**对方**相机也看得见」的那部分——公共可见区
        keep_prev = facing(clouds[i - 1], c_prev, cams[i - 1]) & facing(clouds[i - 1], c_prev, cams[i])
        keep_cur = facing(clouds[i], c_cur, cams[i]) & facing(clouds[i], c_cur, cams[i - 1])
        src, dst = clouds[i][keep_cur], clouds[i - 1][keep_prev]
        if len(src) < 30 or len(dst) < 30:
            out.append(out[-1]); continue
        step = icp(src, dst)
        if step.get("ok"):
            R_acc = np.asarray(step["rotation"], float) @ R_acc
        out.append(float(np.degrees(R.from_matrix(R_acc).magnitude())))
    return np.asarray(out)

def _primitive_template_rotation_deg(clouds: list) -> np.ndarray:
    """和资产模板同一套流程，只把模板换成拟合出来的长方体。

    两者唯一的差别就是模板从哪来，所以并排比能干净地回答「问题出在模板的来源
    （检索不到对的资产）还是模板这个思路本身」。
    """
    from gwm.perception.registration import icp
    model = _fitted_primitive_points(clouds)
    if model is None:
        return np.array([])
    warm_R, warm_t = np.eye(3), np.zeros(3)
    out = []
    for cloud in clouds:
        moved = cloud @ warm_R.T + warm_t
        step = icp(moved, model)
        rot = np.asarray(step["rotation"], float)
        warm_R, warm_t = rot @ warm_R, rot @ warm_t + np.asarray(step["translation"], float)
        out.append(float(np.degrees(R.from_matrix(warm_R).magnitude())))
    return np.asarray(out)


_IMG_INDEX: dict = {}


def _image_rank_of_asset(run: Path, evidence: dict, oid: str, asset_id: str) -> int | None:
    """这个物体自己的裁剪图，在全库 4812 个资产里按图像分排序，**它自己的那个真值
    资产**排第几。

    **为什么要重量一次。** 之前报的「正确资产中位第 365 名」用的是**集合判据**——
    本片段真值集合里排名最靠前的那个。那是放水的上界：一条片段有五六个资产，
    问「有没有任何一个排得靠前」比问「这个物体自己的那个排第几」宽得多。
    而端到端要的恰恰是后者：正确答案有没有进这个物体的候选集。

    我拿集合判据的 365 去算端到端的期望（约 5/76），实测 0/78，对不上。
    这是两处可能的原因之一（另一处是残差的数全在真值感知上测的）。

    真值感知下物体 id 就是真值节点 id，所以不需要配对——那个坑这里绕开了。
    """
    from PIL import Image
    from gwm.assets.annotate import bbox_of, best_observation, crop_object
    from gwm.assets.embed import AssetIndex, Embedder
    from gwm.assets.library import library_root
    from gwm.config import load_config
    from gwm.perception.run import load_masks

    if not asset_id:
        return None
    if "index" not in _IMG_INDEX:
        root = library_root()
        cfg = load_config([REPO / "configs/default.yaml",
                           REPO / f"configs/{site_name()}.yaml"])
        path = Path(root) / "dinov2_index.npz" if root else None
        _IMG_INDEX["index"] = (AssetIndex.load(path)
                               if path and path.is_file() else None)
        _IMG_INDEX["embedder"] = (Embedder(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])
                                  if _IMG_INDEX["index"] is not None else None)
    index, embedder = _IMG_INDEX["index"], _IMG_INDEX["embedder"]
    if index is None or embedder is None:
        return None
    obj = next((o for o in evidence.get("objects") or [] if str(o.get("id")) == oid), None)
    observation = best_observation(obj) if obj else None
    if observation is None:
        return None
    try:
        frames = json.loads((run / "frames" / "frames.json").read_text(encoding="utf-8"))["frames"]
    except (OSError, ValueError, KeyError):
        return None
    frame = next((f for f in frames if f.get("index") == observation.get("frame_index")), None)
    mask = (load_masks(run).get(oid) or {}).get(observation.get("frame_index"))
    if not frame or mask is None:
        return None
    try:
        rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
        box, box_size = bbox_of(observation)
        if box_size and box_size[0]:
            k = rgb.shape[1] / float(box_size[0])
            box = [v * k for v in box]
        crop = crop_object(rgb, np.asarray(mask), box)
        if crop is None:
            return None
        scores = index.scores(embedder.embed_images([Image.fromarray(crop)])[0])
    except (OSError, ValueError, KeyError):
        return None
    own = float(scores.get(asset_id, -1.0))
    if own < 0:
        return None
    return 1 + sum(1 for v in scores.values() if v > own)


def _d2_signature(pts: np.ndarray, bins: int = 32, samples: int = 4000,
                  seed: int = 0) -> np.ndarray | None:
    """D2 形状分布：随机点对的距离直方图，按自身尺度归一化。

    **为什么要它。** 残差比的是**点到点距离**，深度噪声直接打在上面——实测在真实
    VGGT 深度下残差挑资产从 0.297 掉到 0.041（随机 0.030），而把相机位姿换成真值
    只回到 0.054，说明主因是深度不是相机。所以要一个**不依赖逐点表面细节**的判据。

    D2 是全局统计量：逐点噪声在几千个点对里被平均掉，而形状信息留着。它同时比
    包围盒边长比例丰富得多——后者今天已经证明没有预测力（边长比例差 0.002 的网格
    当模板就把假转动翻倍），因为边长比例相同的两个资产可以是完全不同的东西。

    按点对距离的**中位数**归一化，而不是最大值：最大值由最远的那对点决定，
    一个离群点就把整条直方图压扁。
    """
    a = np.asarray(pts, float)
    if len(a) < 32:
        return None
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(a), samples)
    j = rng.integers(0, len(a), samples)
    d = np.linalg.norm(a[i] - a[j], axis=1)
    d = d[d > 1e-9]
    if len(d) < 100:
        return None
    med = float(np.median(d))
    if med <= 1e-9:
        return None
    hist, _ = np.histogram(d / med, bins=bins, range=(0.0, 3.0), density=False)
    total = hist.sum()
    return hist.astype(float) / total if total else None


def _d2_pick_rate(clouds: list, gt_ref: str, n_distract: int,
                  n_views: int = 7) -> float | None:
    """和 `_residual_pick_rate` 同一套协议，只把判据从 ICP 残差换成 D2 距离。

    干扰项同样用**形状最接近**的那批（密度匹配），候选数、视角数都对齐，所以两者
    可以直接并排比——唯一的变量是判据。
    """
    from gwm.assets.library import library_root, load_catalog
    if not gt_ref or not clouds:
        return None
    if library_root() is None:
        return None
    _ASSET_CACHE.setdefault("__catalog__", load_catalog(library_root()))
    picked = _hard_distractors(gt_ref, n_distract)
    if len(picked) < n_distract:
        return None
    if 0 < n_views < len(clouds):
        stride = len(clouds) / float(n_views)
        views = [clouds[min(len(clouds) - 1, int(stride * (i + 0.5)))] for i in range(n_views)]
    else:
        views = clouds
    sigs = [g for g in (_d2_signature(np.asarray(c, float)) for c in views) if g is not None]
    if not sigs:
        return None
    query = np.mean(np.asarray(sigs, float), axis=0)
    best_id, best_d = None, float("inf")
    for aid in [gt_ref] + picked:
        tpl = _asset_points_by_id(aid)
        if tpl is None:
            continue
        sig = _d2_signature(tpl, seed=1)
        if sig is None:
            continue
        # 直方图之间用 L1：对单个分箱的扰动不像 L2 那样被平方放大。
        dist = float(np.abs(sig - query).sum())
        if dist < best_d:
            best_d, best_id = dist, aid
    if best_id is None:
        return None
    return 1.0 if best_id == gt_ref else 0.0


_SIL_CACHE: dict = {}
THUMB_DIR = Path("/scratch/jwj/gwm/asset_thumbs")


def _hu_moments(mask: np.ndarray) -> np.ndarray | None:
    """二值图的 Hu 矩（取对数压幅），外加紧实度和长宽比。

    Hu 矩对平移、尺度、面内旋转都不变，所以观测的朝向任意也能比。
    自己算不依赖 cv2：中心矩 → 归一化中心矩 → Hu 的七个组合。
    """
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None
    x0, y0 = xs.mean(), ys.mean()
    dx, dy = xs - x0, ys - y0
    m00 = float(len(xs))

    def mu(i, j):
        return float((dx ** i * dy ** j).sum())

    def nu(i, j):
        return mu(i, j) / (m00 ** (1.0 + (i + j) / 2.0))

    n20, n02, n11 = nu(2, 0), nu(0, 2), nu(1, 1)
    n30, n03, n21, n12 = nu(3, 0), nu(0, 3), nu(2, 1), nu(1, 2)
    h = np.empty(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h[5] = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4 * n11 * (n30 + n12) * (n21 + n03))
    h[6] = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    hu = np.sign(h) * np.log10(np.abs(h) + 1e-30)
    # 再补两个直观量：紧实度（占外接框的比例）和长宽比。它们不是 Hu 的冗余，
    # Hu 对这两样不敏感，而低多边形资产之间差别恰恰常在这里。
    w = float(xs.max() - xs.min() + 1)
    hgt = float(ys.max() - ys.min() + 1)
    solidity = m00 / max(w * hgt, 1.0)
    aspect = max(w, hgt) / max(min(w, hgt), 1.0)
    return np.concatenate([hu, [solidity * 10.0, np.log10(aspect) * 10.0]])


def _asset_silhouettes(asset_id: str) -> list:
    """资产的轮廓描述子（从缩略图的 alpha 通道来，每个资产 4 个视角）。

    **为什么轮廓。** 根因量到深度误差是**沿相机视线**的（物体被压扁到最短轴的
    0.526），而轮廓是**二维**的，完全不碰深度——沿视线的误差投影到像平面上是零。
    所以这是唯一一个绕开已坐实根因的判据。

    每个候选严格 4 个视角，预算完全相同，所以「取最好」在这里不触犯
    [[best-of-n-init-biases-selection]]（那条针对的是预算不等）。
    """
    if asset_id in _SIL_CACHE:
        return _SIL_CACHE[asset_id]
    from PIL import Image
    stem = asset_id.replace("/", "-")
    out = []
    for k in range(4):
        f = THUMB_DIR / f"{stem}_{k}.png"
        if not f.is_file():
            continue
        try:
            rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.int16)
        except (OSError, ValueError):
            continue
        # **不能用 alpha 通道抠。** 第一版用 `convert("RGBA")[:,:,3] > 16`，而这批
        # 缩略图的 alpha **全是 255**（有 alpha 通道不等于用了它）——于是每个资产的
        # 「轮廓」都是整个 160×160 方块，Hu 矩完全相同，所有候选距离相等。
        # 第一版靠 `if d < best_d` 严格小于保留先遍历到的那个，而列表是
        # `[gt_ref] + picked`，真值永远排首位，于是「74/74 全对」——纯平局顺序造的。
        # 自检（margin 恒为 0.000、改成排序后挑对率掉到 0.000~0.125）把它抓了出来。
        #
        # 背景是均匀的浅灰 (242,242,242)，按离背景色的距离抠前景。
        bg = np.median(np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]),
                       axis=0)
        a = np.abs(rgb - bg[None, None, :]).max(axis=2) > 24
        d = _hu_moments(a)
        if d is not None:
            out.append(d)
    _SIL_CACHE[asset_id] = out
    return out


def _silhouette_pick_rate(per_frame: dict, frames: list, gt_ref: str,
                          n_distract: int, n_views: int = 7,
                          stats: dict | None = None) -> float | None:
    """**用轮廓挑资产**：观测掩码的 Hu 矩对候选资产缩略图轮廓的 Hu 矩。

    协议和残差那条对齐：同样的形状最接近干扰项、同样的候选数和视角数，
    唯一的变量是判据——一个比三维表面（被深度误差毁掉），一个比二维轮廓（不碰深度）。
    """
    if stats is None:
        stats = {}
    if not gt_ref or not per_frame:
        return None
    picked = _hard_distractors(gt_ref, n_distract)
    if len(picked) < n_distract:
        return None
    avail = [f for f in frames if f in per_frame]
    if not avail:
        return None
    if 0 < n_views < len(avail):
        stride = len(avail) / float(n_views)
        avail = [avail[min(len(avail) - 1, int(stride * (i + 0.5)))] for i in range(n_views)]
    obs = [d for d in (_hu_moments(np.asarray(per_frame[f]).astype(bool)) for f in avail)
           if d is not None]
    if not obs:
        return None
    scored = []
    for aid in [gt_ref] + picked:
        sils = _asset_silhouettes(aid)
        if not sils:
            continue
        # 每一帧对该资产的 4 个视角取最近的那个，再对帧取中位——和残差那条
        # 同样的聚合方式（中位不取均值，单帧掩码漏一块不会拖垮）。
        per = [min(float(np.abs(o - t).sum()) for t in sils) for o in obs]
        scored.append((float(np.median(per)), aid))
    # **自检。** 第一版给出 74/74 全对，而残差在同一批候选上（而且是干净几何）
    # 才 0.297——这种整齐的数先查是不是某道门把候选全挡了。
    # 实际参与打分的候选数该是 n_distract+1；少于这个说明有候选被静默跳过。
    if len(scored) < 2:
        return None
    scored.sort()
    stats["n_scored"] = len(scored)
    stats["margin"] = (scored[1][0] - scored[0][0]) / max(scored[0][0], 1e-9)
    return 1.0 if scored[0][1] == gt_ref else 0.0


def _gt_other_ref(oid: str) -> str:
    """同一片段里**另一个**物体的真值资产 id，用来做错配零假设。

    拿 A 的掩码去配「A 的真值 + 它的 32 个形状最近邻」这个候选集，挑对率该掉到
    随机；掉不下去就说明判据里有泄漏，而不是真的在比形状。
    """
    refs = _gt_asset_refs()
    others = sorted(v for k, v in refs.items() if k != oid)
    if not others:
        return ""
    return others[hash(oid) % len(others)]


def _project_silhouette(pts: np.ndarray, R_wc: np.ndarray, yaw: float,
                        grid: int = 96) -> np.ndarray | None:
    """把候选网格按偏航转一下、用**相机朝向**转进相机系、正交投影成二值轮廓。

    **这条路一点不碰深度。** 根因是深度误差沿相机视线（物体被压扁到最短轴的
    0.526），而沿视线的误差投影到像平面上恰好是零。所以只用相机朝向 + 掩码。

    正交投影而不是透视：物体在画面里只占几百到几千像素，透视形变可以忽略，
    而正交投影不需要知道物体离相机多远——那个距离正是深度给的、而深度是坏的。

    投影后按**二维包围盒**归一化到固定格子，所以和物体的像素大小无关，
    比的是纯形状。

    **格子不能太粗。** 第一版用 48×48 = 2304 格，而网格点有 3000 个——平均每格
    1.3 个点，几乎填满整个包围框，于是不同偏航的轮廓都成了差不多的实心块，
    IoU 对角度不敏感（实测 margin 只有 1~2%，解位姿时偏航在整个圆周上乱跳）。
    提到 96×96 = 9216 格，同时把采样点数提上去填实。
    """
    a = np.asarray(pts, float)
    c, sn = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
    cam = (a @ R_yaw.T) @ np.asarray(R_wc, float).T
    uv = cam[:, :2]
    lo, hi = uv.min(0), uv.max(0)
    span = np.maximum(hi - lo, 1e-9)
    # 等比缩放到格子里（保长宽比），居中
    k = (grid - 2) / span.max()
    ij = np.floor((uv - lo) * k).astype(int)
    ij[:, 0] += (grid - 1 - int(span[0] * k)) // 2
    ij[:, 1] += (grid - 1 - int(span[1] * k)) // 2
    ij = np.clip(ij, 0, grid - 1)
    out = np.zeros((grid, grid), bool)
    out[ij[:, 1], ij[:, 0]] = True
    return out


def _mask_to_grid(mask: np.ndarray, grid: int = 96) -> np.ndarray | None:
    """观测掩码按它的二维包围盒等比归一化到同一个格子。"""
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    k = (grid - 2) / float(max(w, h))
    ij = np.stack([((xs - x0) * k).astype(int), ((ys - y0) * k).astype(int)], axis=1)
    ij[:, 0] += (grid - 1 - int(w * k)) // 2
    ij[:, 1] += (grid - 1 - int(h * k)) // 2
    ij = np.clip(ij, 0, grid - 1)
    out = np.zeros((grid, grid), bool)
    out[ij[:, 1], ij[:, 0]] = True
    return out


def _degrade_mask(g: np.ndarray, px: int, seed: int = 0) -> np.ndarray:
    """把二值轮廓按 px 个格子膨胀或腐蚀，模拟神经掩码的边界误差。

    **为什么要它。** 端到端把轮廓挑接进管线之后仍是 0/75，而探针上同一配置该给到
    约 4/78。账目显示机制跑通了（93/122 条决策改由图像候选胜出），所以不是开关
    没点着。差别在输入：**探针用真值掩码，管线用神经掩码**，而轮廓判据整个建立
    在掩码上——掩码就是信号本身，边界一糊 IoU 比较就失效。深度残差对这个反而
    不敏感，因为它比的是三维点。

    这条扫出「掩码要准到什么程度轮廓挑才成立」，是任何分割后端要达到的指标——
    和深度噪声那条扫描同一个用法。

    正负各一半：膨胀和腐蚀对 IoU 的影响不对称，只扫一侧会读偏。
    """
    if px == 0:
        return g
    out = g.copy()
    grow = (seed % 2 == 0)
    for _ in range(abs(px)):
        a = out
        sh = np.zeros_like(a)
        sh[1:, :] |= a[:-1, :]; sh[:-1, :] |= a[1:, :]
        sh[:, 1:] |= a[:, :-1]; sh[:, :-1] |= a[:, 1:]
        out = (a | sh) if grow else (a & sh)
    return out


def _render_pick_rate(per_frame: dict, frames: list, cams: dict, gt_ref: str,
                      n_distract: int, n_views: int = 3, n_yaw: int = 8,
                      stats: dict | None = None, mask_px: int = 0) -> float | None:
    """**按观测位姿渲染候选轮廓**，和观测掩码比 IoU，挑最高的。

    上一版用资产缩略图的固定 4 视角 + Hu 矩，得 0.081（随机 0.030，2.7 倍）——
    方向对但太粗。这一版直接按**这一帧的相机朝向**渲染候选，视角就对上了。

    每个候选给同样的 `n_yaw` 个偏航，预算严格相同，所以「取最好」不触犯
    [[best-of-n-init-biases-selection]]（那条针对的是预算不等）。
    """
    if stats is None:
        stats = {}
    if not gt_ref or not per_frame:
        return None
    picked = _hard_distractors(gt_ref, n_distract)
    if len(picked) < n_distract:
        return None
    avail = [f for f in frames if f in per_frame and f in cams]
    if not avail:
        return None
    if 0 < n_views < len(avail):
        stride = len(avail) / float(n_views)
        avail = [avail[min(len(avail) - 1, int(stride * (i + 0.5)))] for i in range(n_views)]
    obs = [(f, _mask_to_grid(per_frame[f])) for f in avail]
    obs = [(f, g) for f, g in obs if g is not None]
    if not obs:
        return None
    if mask_px:
        obs = [(f, _degrade_mask(g, mask_px, seed=i)) for i, (f, g) in enumerate(obs)]
    yaws = [2.0 * np.pi * j / n_yaw for j in range(n_yaw)]
    scored = []
    for aid in [gt_ref] + picked:
        tpl = _asset_points_by_id(aid, samples=20000)
        if tpl is None or len(tpl) < 50:
            continue
        tpl = tpl - tpl.mean(0)
        best_per_frame = []
        for f, g in obs:
            R_wc = np.asarray(cams[f], float)[:3, :3].T      # 世界 -> 相机
            best = 0.0
            for y in yaws:
                sil = _project_silhouette(tpl, R_wc, y)
                if sil is None:
                    continue
                inter = float(np.logical_and(sil, g).sum())
                union = float(np.logical_or(sil, g).sum())
                if union > 0:
                    best = max(best, inter / union)
            best_per_frame.append(best)
        if best_per_frame:
            scored.append((-float(np.median(best_per_frame)), aid))
    if len(scored) < 2:
        return None
    scored.sort()
    stats["n_scored"] = len(scored)
    stats["margin"] = abs(scored[1][0] - scored[0][0]) / max(abs(scored[0][0]), 1e-9)
    return 1.0 if scored[0][1] == gt_ref else 0.0


def _silhouette_yaw_series(per_frame: dict, frames: list, cams: dict,
                           asset_id: str, n_yaw: int = 36) -> np.ndarray:
    """**用轮廓解位姿**：每一帧扫一圈偏航，取和观测掩码 IoU 最高的那个。

    今天所有位姿实验都建在**深度点云的 ICP** 上，而深度被证明是坏的（VGGT 沿相机
    视线把物体压扁到最短轴的 0.526），所以那条路上静止物体的假转动一直压不下去
    （真实几何下基线 30.72°）。

    而判据那一侧已经量到：不碰深度的判据比用深度的强 2.6 倍
    （按观测位姿渲染 + IoU 0.108 对 ICP 残差 0.041，随机 0.030）。
    既然二维轮廓携带的可用形状信息更多，**位姿也该从轮廓解**。这条一次都没试过。

    返回每帧相对第 0 帧的转角（度）。偏航的分辨率是 360/n_yaw，所以 n_yaw=36
    时量化噪声约 ±5°——判「静止物体解出几十度假转动」够用了。
    """
    if not asset_id:
        return np.array([])
    tpl = _asset_points_by_id(asset_id, samples=20000)
    if tpl is None or len(tpl) < 50:
        return np.array([])
    tpl = tpl - tpl.mean(0)
    yaws = np.array([2.0 * np.pi * j / n_yaw for j in range(n_yaw)])
    out = []
    for f in frames:
        if f not in per_frame or f not in cams:
            out.append(np.nan); continue
        g = _mask_to_grid(per_frame[f])
        if g is None:
            out.append(np.nan); continue
        R_wc = np.asarray(cams[f], float)[:3, :3].T
        best, best_y = -1.0, np.nan
        for y in yaws:
            sil = _project_silhouette(tpl, R_wc, float(y))
            if sil is None:
                continue
            union = float(np.logical_or(sil, g).sum())
            if union <= 0:
                continue
            iou = float(np.logical_and(sil, g).sum()) / union
            if iou > best:
                best, best_y = iou, float(np.degrees(y))
        out.append(best_y)
    a = np.asarray(out, float)
    if not np.isfinite(a).any():
        return np.array([])
    # 相对第一个有效帧，并把角度绕回 [-180, 180]
    base = a[np.isfinite(a)][0]
    d = (a - base + 180.0) % 360.0 - 180.0
    return np.abs(np.where(np.isfinite(d), d, 0.0))


def _image_top_candidates(run: Path, evidence: dict, oid: str, n: int) -> list:
    """这个物体的裁剪图在全库里按图像分排序的前 n 个资产 id。

    **为什么要它。** 到目前为止所有「挑资产」的实验，候选集用的都是**形状最近邻**
    （按包围盒边长比例），而管线真正面对的是**图像检索的前 N 名**。两者难度未必
    一样，而后者才是这套方法的真实形态。

    图像那一侧的数已经有了：静止物体自己的真值资产中位排第 **36** 名，
    前 64 名覆盖率 0.625、前 256 名 0.750。所以「图像提候选 + 轮廓挑」这个组合
    是有希望的，而它一次都没被整体量过。
    """
    from PIL import Image
    from gwm.assets.annotate import bbox_of, best_observation, crop_object
    from gwm.perception.run import load_masks

    if "index" not in _IMG_INDEX:
        _image_rank_of_asset(run, evidence, oid, "__warmup__")
    index, embedder = _IMG_INDEX.get("index"), _IMG_INDEX.get("embedder")
    if index is None or embedder is None:
        return []
    obj = next((o for o in evidence.get("objects") or [] if str(o.get("id")) == oid), None)
    observation = best_observation(obj) if obj else None
    if observation is None:
        return []
    try:
        frames_j = json.loads((run / "frames" / "frames.json").read_text(encoding="utf-8"))["frames"]
    except (OSError, ValueError, KeyError):
        return []
    frame = next((f for f in frames_j if f.get("index") == observation.get("frame_index")), None)
    mask = (load_masks(run).get(oid) or {}).get(observation.get("frame_index"))
    if not frame or mask is None:
        return []
    try:
        rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
        box, box_size = bbox_of(observation)
        if box_size and box_size[0]:
            k = rgb.shape[1] / float(box_size[0])
            box = [v * k for v in box]
        crop = crop_object(rgb, np.asarray(mask), box)
        if crop is None:
            return []
        scores = index.scores(embedder.embed_images([Image.fromarray(crop)])[0])
    except (OSError, ValueError, KeyError):
        return []
    return [aid for aid, _v in sorted(scores.items(), key=lambda kv: -kv[1])[:n]]


def _combined_pick_rate(run: Path, evidence: dict, per_frame: dict, frames: list,
                        cams: dict, oid: str, gt_ref: str, n_cand: int = 64,
                        stats: dict | None = None) -> float | None:
    """**这套方法的真实形态**：图像提候选（前 n_cand 名），轮廓 IoU 挑。

    和之前所有挑资产的实验不同，候选集不是人造的「真值 + 形状最近邻」，而是管线
    真正会拿到的那一批。所以这个数**包含了「正确答案可能根本不在候选里」这一项**
    ——静止物体前 64 名的覆盖率是 0.625，所以上限就是它。

    返回 1/0；候选里压根没有真值时返回 0（不是 None），因为那正是真实失败的一部分。
    """
    if stats is None:
        stats = {}
    cands = _image_top_candidates(run, evidence, oid, n_cand)
    if not cands:
        return None
    stats["n_cand"] = len(cands)
    stats["gt_in_cands"] = 1.0 if gt_ref in cands else 0.0
    if gt_ref not in cands:
        return 0.0                      # 真实失败的一部分，不能当缺失值丢掉
    avail = [f for f in frames if f in per_frame and f in cams]
    if not avail:
        return None
    if len(avail) > 3:
        stride = len(avail) / 3.0
        avail = [avail[min(len(avail) - 1, int(stride * (i + 0.5)))] for i in range(3)]
    obs = [(f, _mask_to_grid(per_frame[f])) for f in avail]
    obs = [(f, g) for f, g in obs if g is not None]
    if not obs:
        return None
    yaws = [2.0 * np.pi * j / 8 for j in range(8)]
    scored = []
    for aid in cands:
        tpl = _asset_points_by_id(aid, samples=20000)
        if tpl is None or len(tpl) < 50:
            continue
        tpl = tpl - tpl.mean(0)
        per = []
        for f, g in obs:
            R_wc = np.asarray(cams[f], float)[:3, :3].T
            best = 0.0
            for y in yaws:
                sil = _project_silhouette(tpl, R_wc, float(y))
                if sil is None:
                    continue
                union = float(np.logical_or(sil, g).sum())
                if union > 0:
                    best = max(best, float(np.logical_and(sil, g).sum()) / union)
            per.append(best)
        if per:
            scored.append((-float(np.median(per)), aid))
    if len(scored) < 2:
        return None
    scored.sort()
    stats["n_scored"] = len(scored)
    return 1.0 if scored[0][1] == gt_ref else 0.0


def _shape_distance(a_id: str, b_id: str) -> float | None:
    """两个资产的边长比例平均绝对差。和 probe_asset_retrieval 报的「形状距离」同一个量。"""
    shapes = _ASSET_CACHE.get("__shapes__") or {}
    a, b = shapes.get(a_id), shapes.get(b_id)
    if not a or not b:
        return None
    try:
        return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())
    except (TypeError, ValueError):
        return None


def _hard_distractors(gt_ref: str, n: int) -> list[str]:
    """**形状最接近真值资产**的那 n 个，当难干扰项用。

    随机抽的干扰项太好认——随机资产长得和目标完全不像。而真实方法面对的候选集是
    检索出来的前 N 名，**个个都像**。拿随机干扰项量出来的挑对率是偏乐观的，这正是
    「零假设要密度匹配」那条。

    用边长比例（`asset_shapes.json` 里预算好的）找最近的 n 个。它和被测的判据
    （配准残差）不是同一个量，所以不构成「配对条件含被测量」那类泄漏。
    """
    shapes = _ASSET_CACHE.get("__shapes__")
    if shapes is None:
        import os
        path = os.environ.get("GWM_ASSET_SHAPES", "/scratch/jwj/gwm/asset_shapes.json")
        try:
            shapes = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            shapes = {}
        _ASSET_CACHE["__shapes__"] = shapes
    target = shapes.get(gt_ref)
    if not target:
        return []
    t = np.asarray(target, float)
    scored = []
    for aid, ratio in shapes.items():
        if aid == gt_ref:
            continue
        try:
            scored.append((float(np.abs(np.asarray(ratio, float) - t).mean()), aid))
        except (TypeError, ValueError):
            continue
    scored.sort()
    return [aid for _d, aid in scored[:n]]


def _residual_pick_rate(clouds: list, gt_ref: str, n_distract: int,
                        seed: int = 0, hard: bool = False,
                        n_views: int = 1, n_yaw: int = 1,
                        noise: float = 0.0, unflatten: float = 1.0) -> float | None:
    """**用配准残差挑资产**：真值资产 + N 个随机干扰项里，残差最小的是不是真值。

    这是创新点一的字面实现——用连续稠密几何去监督一个离散的符号选择。梯度传不过去，
    但「哪个网格配得上这团点」这个判据不需要梯度。

    **为什么这样设计而不是直接测管线的命中率。** 管线那条链有两段：提候选、在候选里挑。
    现在卡在第一段（文字查询词来自检测器，命中率 0/71），而第二段从来没被单独测过。
    把两段拆开才答得了「挑这件事本身成不成立」。

    **为什么不用类名取候选。** assetgt 的类名就是资产文件名拆出来的，拿它检索等于
    拿资产自己的名字查自己——那条「上限 top-1 0.689」就是这么来的，是循环的。
    改成「真值 + 随机干扰项」就没有这个问题，而且 N 可调，直接给出「能从多少个里挑对」。

    配准前把每个候选缩到**逐帧包围盒的中位对角线**上。并帧对角线比物体大 1.89 倍，
    而受控扫描说模板大 1.2 倍就越过对照、1.3 倍崩掉——这个坑踩过一次。

    随机干扰项要**按 id 排序后再抽**，否则目录遍历顺序会让结果不可复现。
    """
    from gwm.assets.library import library_root, load_catalog
    from gwm.perception.registration import icp
    if not gt_ref or not clouds:
        return None
    root = library_root()
    if root is None:
        return None
    catalog = _ASSET_CACHE.setdefault("__catalog__", load_catalog(root))
    others = sorted((a.id for a in catalog if a.id != gt_ref))
    if len(others) < n_distract:
        return None
    if hard:
        picked = _hard_distractors(gt_ref, n_distract)
        if len(picked) < n_distract:
            return None
    else:
        rng = random.Random(f"{seed}:{gt_ref}:{n_distract}")
        picked = rng.sample(others, n_distract)
    ids = [gt_ref] + picked
    per_frame = [np.linalg.norm(np.asarray(c, float).max(0) - np.asarray(c, float).min(0))
                 for c in clouds if len(np.asarray(c, float)) >= 10]
    if not per_frame:
        return None
    obs_diag = float(np.median(per_frame))
    # **用几帧，不是一帧。** 第一版只拿中间那一帧当查询，而一帧只是一个局部视角——
    # 同一个包围盒比例下完全不同的物体，在单个视角上可能都配得差不多。多取几帧
    # （均匀抽），每帧各配一次取残差中位数，不同视角的证据就叠起来了。
    # 代价是线性增加 ICP 次数，而 ICP 在三千点上很便宜。
    # **别从第 0 帧起手。** 首帧的可见面最偏，实测只用首帧挑对 0.149、只用中间帧
    # 0.243——差 63%。均匀抽的时候把起点放到半个间隔处，避开两端。
    picks = clouds
    if 0 < n_views < len(clouds):
        step = len(clouds) / float(n_views)
        picks = [clouds[min(len(clouds) - 1, int(step * (i + 0.5)))] for i in range(n_views)]
    queries = [np.asarray(c, float) for c in picks if len(np.asarray(c, float)) >= 30]
    if not queries:
        return None
    # **注入可控深度噪声。** 真实 VGGT 深度下这条判据掉到随机水平，而相机、判据、
    # 物体大小三条都排除了。所以问题是「深度要准到什么程度这条才成立」——这个数
    # 比「再试一个后端」有用得多：它是任何未来几何后端要达到的**指标**。
    # 噪声按物体自身尺度给（包围盒对角线的比例），各向同性高斯。
    if noise > 0:
        nrng = np.random.default_rng(abs(hash((gt_ref, n_distract, noise))) % (2**32))
        scale = float(np.median([np.linalg.norm(q.max(0) - q.min(0)) for q in queries]))
        queries = [q + nrng.normal(0.0, noise * scale, q.shape) for q in queries]
    # **把已知的系统性畸变补偿掉。** 实测 VGGT 把物体沿最短轴压扁到正确比例的
    # 0.526（中位，n=67，中间轴 0.924 基本对），而噪声扫描已经排除了「这是噪声」
    # （5% 各向同性噪声仍有 0.108，真实 VGGT 只有 0.041）。
    #
    # 所以直接沿**最短主轴**拉回来：1/0.526 ≈ 1.9。用 PCA 定轴而不是坐标轴，
    # 因为物体朝向是任意的。回得来就同时坐实诊断并拿到一个可用的修正；
    # 回不来就说明压扁不是起作用的那个因素。
    if unflatten and abs(unflatten - 1.0) > 1e-6:
        fixed = []
        for q in queries:
            c = q.mean(0)
            centred = q - c
            try:
                _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
            except np.linalg.LinAlgError:
                fixed.append(q); continue
            local = centred @ vt.T          # 主轴坐标系，第三根是最短的
            local[:, 2] *= float(unflatten)
            fixed.append(local @ vt + c)
        queries = fixed
    best_id, best_rmse = None, float("inf")
    for aid in ids:
        tpl = _asset_points_by_id(aid)
        if tpl is None:
            continue
        tpl_diag = float(np.linalg.norm(tpl.max(0) - tpl.min(0)))
        if tpl_diag <= 1e-9:
            continue
        model = (tpl - tpl.mean(0)) * (obs_diag / tpl_diag)
        # **多个初始偏航各配一次取最好。** 资产网格有自己的规范朝向，而观测到的
        # 物体偏航是任意的；ICP 从质心对齐起手只做局部搜索，朝向差太多就停在局部
        # 极小。视角数那条杠杆已经到顶（7 帧之后不动），这是下一个最可能的瓶颈，
        # 而且它正好解释了为什么转动类全军覆没（periodic_rotate 挑对 0.000）。
        #
        # 只绕竖轴转：场景里的物体基本都立在地面上，俯仰和翻滚的自由度实际用不上，
        # 而每多一个初值就多一次 ICP。
        variants = [model]
        for j in range(1, max(1, n_yaw)):
            ang = 2.0 * np.pi * j / n_yaw
            c, sn = np.cos(ang), np.sin(ang)
            rot = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
            variants.append(model @ rot.T)
        rs = []
        for q in queries:
            # 每一帧在所有初值里取**最小**残差：这一帧最好能解释成什么样。
            best_q = None
            for m in variants:
                step = icp(q, m)
                if step["ok"] and step["rmse"] is not None:
                    v = float(step["rmse"])
                    if best_q is None or v < best_q:
                        best_q = v
            if best_q is not None:
                rs.append(best_q)
        if not rs:
            continue
        # 取中位数不取均值：某一帧掩码漏了一块就会拉出一个离谱的残差，
        # 而中位数不受单帧拖累。
        score = float(np.median(rs))
        if score < best_rmse:
            best_rmse, best_id = score, aid
    if best_id is None:
        return None
    return 1.0 if best_id == gt_ref else 0.0


def _camera_total_deg(cam_list: list) -> float | None:
    """这一段相机一共转过多少度（绕物体的方位角总行程）。

    **为什么换这个量。** 逐步相关系数那条路被量具自己否了：它的分母是两列的标准差，
    而真值静止那一桶的 `cam_step_spread`（每步转角的四分位距/中位数）只有 **0.036**，
    也就是每步角步长近似常数——相关系数算出来是噪声。合成探针里我为此特意给相机
    加了非匀速，真实数据的相机档没有这个性质。所以 static 那个 -0.034 **不是**
    「机制不存在」的证据，那条路没被关闭，只是量错了。

    换一个不依赖「逐帧步长有方差」的形式：机制预测假转动的**幅度**随相机的**总
    角位移**增长。拿它和 `solved_range_deg` 配对，就能直接回归。

    自检同样要带：如果各物体的总角位移都差不多，这个回归也退化。所以把它本身也
    报出来，先看它散不散得开。
    """
    if len(cam_list) < 2:
        return None
    total = 0.0
    for a, b in zip(cam_list[:-1], cam_list[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        total += float(np.degrees(np.arccos(
            np.clip(float(a @ b) / (na * nb), -1.0, 1.0))))
    return total if total > 1e-9 else None


def _camera_step_spread(cam_list: list) -> float | None:
    """每步相机转角的**变异程度**（四分位距 / 中位数）。量具的自检。

    相关系数的分母是两列的标准差。如果每步角步长近似常数，这个相关系数算出来的是
    噪声，不是证据——合成探针里我为此特意给相机加了非匀速，注释写得很清楚：
    「均匀的话每步角步长是常数，和任何东西的相关系数都算不出来（分母是零方差）」。

    而真实数据的相机档里 `orbit` 的角度就是 u 的线性函数，角步长近似常数。
    所以报相关系数之前必须先报这个数：它太小的话，上面那一列不能当否定结论用。
    """
    if len(cam_list) < 4:
        return None
    steps = []
    for a, b in zip(cam_list[:-1], cam_list[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        steps.append(float(np.degrees(np.arccos(
            np.clip(float(a @ b) / (na * nb), -1.0, 1.0)))))
    if len(steps) < 3:
        return None
    med = float(np.median(steps))
    if med < 1e-6:
        return None
    return float((np.percentile(steps, 75) - np.percentile(steps, 25)) / med)


def _camera_step_correlation(tracked: dict, cam_list: list) -> float | None:
    """逐步解出的转角，和相机这一步转了多少，相关到什么程度。

    **为什么轮到它。** 伪造转动的机制已经坐实：采样格子钉在图像上 × 相机在动，
    相邻两帧采到的不是同一批物理点。十二条排除清单里「相机位姿」那一条指的是
    把相机位姿**用进配准**（每帧只对它看得见的那部分模型配），不是拿它**当判别子**。
    后者没试过，而它直接长在机制上：假转动由相机扫过物体表面造成，应该跟着相机的
    角步长走；物体自身的转动不该。合成实验里这个相关系数在出错的那几行是
    +0.333~+0.446，真转动那几行算不出来（分母零方差）或接近零。

    **这是个不需要真值的量**，所以如果两类分得开，可以逐物体逐帧当门用——而现在
    那道门用的是足迹长短边比，已经量过分不开（排除清单第十二条）。

    模板那条路今天走到头了：所有模板变体都是**均匀阻尼**，真假转动一起压，判别比
    从基线的 3.7 倍掉到 0.5 倍，倒挂。所以方向不是「换更好的模板」，是「更好的闸」。
    """
    steps = tracked.get("step_transforms") or []
    if len(steps) < 3 or len(cam_list) < len(steps) + 1:
        return None
    cam_step, obj_step = [], []
    for k, (rot, _t) in enumerate(steps):
        a = np.asarray(cam_list[k], float)
        b = np.asarray(cam_list[k + 1], float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        cam_step.append(float(np.degrees(np.arccos(
            np.clip(float(a @ b) / (na * nb), -1.0, 1.0)))))
        ang = (np.trace(np.asarray(rot, float)) - 1.0) / 2.0
        obj_step.append(float(np.degrees(np.arccos(np.clip(ang, -1.0, 1.0)))))
    if len(cam_step) < 3 or np.ptp(cam_step) < 1e-9 or np.ptp(obj_step) < 1e-12:
        return None
    return float(np.corrcoef(cam_step, obj_step)[0, 1])


def _per_frame_box_points(clouds: list, n: int = 4000,
                          seed: int = 0) -> tuple[np.ndarray | None, float]:
    """按**逐帧包围盒的中位边长**造模板，而不是按所有帧并起来的那团点云。

    `_fitted_primitive_points` 的结论是「拟合长方体模板不成立，比不用模板还差」
    （static 的 range 43.59° 对基线 19.85°），而它自己的注释把原因写清楚了：
    那个盒子是**所有帧并起来**的点云的包围盒，而那团点云随视角扫动，所以盒子比
    物体大一圈。

    今天的受控扫描给这个诊断配上了刻度——模板相对点云大多少时会崩：

        长方体   x1.0: 5.2°   x1.1: 9.7°   x1.2: 24.2°   x1.3: 88.5°（对照 12.5°）
        L 形     x1.0: 4.0°   x1.1: 4.6°   x1.2:  5.7°   x1.3: 15.4°（对照  7.6°）

    悬崖在 x1.1 到 x1.3 之间。所以如果并帧盒子的超大倍数落在这个区间，那次失败
    就是**模板偏大**造成的，而不是「模板这条路不成立」——两者的区别决定后面做不做。

    逐帧包围盒取中位边长系统性地小于并帧包围盒，而且管线里 `obbs` 本来就是逐帧的。
    一并返回超大倍数（并帧对角线 / 逐帧中位对角线），好直接验证那个诊断。
    """
    if not clouds:
        return None, float("nan")
    per_frame = []
    for c in clouds:
        a = np.asarray(c, float)
        if len(a) >= 10:
            per_frame.append(a.max(0) - a.min(0))
    if not per_frame:
        return None, float("nan")
    size = np.median(np.asarray(per_frame, float), axis=0)
    span = np.vstack(clouds)
    union = span.max(0) - span.min(0)
    oversize = (float(np.linalg.norm(union) / np.linalg.norm(size))
                if np.linalg.norm(size) > 1e-9 else float("nan"))
    if (size <= 1e-6).any():
        return None, oversize
    rng = np.random.default_rng(seed)
    per = max(n // 6, 1)
    faces = []
    lo = -size / 2.0
    for axis in range(3):
        for end in (0.0, 1.0):
            uv = rng.random((per, 3))
            uv[:, axis] = end
            faces.append(lo + uv * size)
    pts = np.vstack(faces)
    return pts - pts.mean(0), oversize


def _per_frame_box_rotation_deg(clouds: list) -> tuple[np.ndarray, float]:
    """和 `_primitive_template_rotation_deg` 同一套流程，只换模板的搭法。

    唯一的变量是「盒子按并帧点云搭还是按逐帧包围盒中位边长搭」，所以并排比能干净
    地回答：那次失败是模板偏大造成的，还是模板这条路本身不成立。
    """
    from gwm.perception.registration import icp
    model, oversize = _per_frame_box_points(clouds)
    if model is None:
        return np.array([]), oversize
    warm_R, warm_t = np.eye(3), np.zeros(3)
    out = []
    for cloud in clouds:
        moved = cloud @ warm_R.T + warm_t
        step = icp(moved, model)
        rot = np.asarray(step["rotation"], float)
        warm_R, warm_t = rot @ warm_R, rot @ warm_t + np.asarray(step["translation"], float)
        out.append(float(np.degrees(R.from_matrix(warm_R).magnitude())))
    return np.asarray(out), oversize


def _template_rotation_from_points(clouds: list, template) -> np.ndarray:
    """给定模板点集，逐帧把观测配上去，返回累积转角序列。

    尺度按**逐帧包围盒对角线的中位**对齐，不是并帧点云的对角线——后者随视角扫动，
    实测比前者大 **1.89 倍**，而受控扫描说模板大 1.2 倍就越过对照、1.3 倍崩掉。
    """
    from gwm.perception.registration import icp
    if template is None or not clouds:
        return np.array([])
    per_frame = [np.linalg.norm(np.asarray(c, float).max(0) - np.asarray(c, float).min(0))
                 for c in clouds if len(np.asarray(c, float)) >= 10]
    if not per_frame:
        return np.array([])
    obs_diag = float(np.median(per_frame))
    tpl_diag = float(np.linalg.norm(template.max(0) - template.min(0)))
    if tpl_diag <= 1e-9 or obs_diag <= 1e-9:
        return np.array([])
    model = (template - template.mean(0)) * (obs_diag / tpl_diag)
    warm_R, warm_t = np.eye(3), np.zeros(3)
    out = []
    for cloud in clouds:
        moved = cloud @ warm_R.T + warm_t
        step = icp(moved, model)
        rot = np.asarray(step["rotation"], float)
        warm_R, warm_t = rot @ warm_R, rot @ warm_t + np.asarray(step["translation"], float)
        out.append(float(np.degrees(R.from_matrix(warm_R).magnitude())))
    return np.asarray(out)


DECISIONS_ROOT: Path | None = None          # --decisions-from，见 _chosen_assets


def _chosen_assets(run: Path) -> dict:
    """管线**实际挑中**的资产和它的文字分，按物体 id 索引。

    读 `rebuild.json` 的 decisions。这是「二次配准」那条路要的输入：
    配准跑在感知里、在检索**之前**，所以要用检索结果当模板只能再跑一遍。

    **这个探针自己只跑感知，不跑检索**，所以它的 work 目录里没有 rebuild.json。
    用 `--decisions-from` 指到一次**已经跑过的管线产出**（里面按片段名分子目录），
    从那里取 decisions。两边的感知档位必须一致（都是 gtseg），否则物体 id 对不上。
    """
    if DECISIONS_ROOT is not None and CURRENT_CLIP_DIR is not None:
        rb = DECISIONS_ROOT / CURRENT_CLIP_DIR.name / "rebuild.json"
    else:
        rb = run.parent / "rebuild.json"
    if not rb.is_file():
        return {}
    try:
        recs = json.loads(rb.read_text(encoding="utf-8")).get("decisions") or []
    except (OSError, ValueError):
        return {}
    out = {}
    for r in recs:
        if not isinstance(r, dict) or not r.get("chosen"):
            continue
        key = str(r.get("evidence_id") or r.get("id") or "")
        if key:
            out[key] = (str(r["chosen"]), float(r.get("score") or 0.0))
    return out


def _asset_template_rotation_deg(clouds: list, cls: str) -> np.ndarray:
    """拿检索到的资产网格当模板，逐帧把观测配上去。

    资产的尺度和物体未必一致（检索只保证类别像），所以配准前先按两边的包围盒
    对角线把模板缩到同一尺度——ICP 解的是刚体变换，尺度不对会被旋转和平移吸收。

    **缩到哪个对角线上要紧，这里原来是错的。** 第一版用 `np.vstack(clouds)`，也就是
    所有帧并起来的那团点云的对角线。但那团点云随视角扫动，比物体大一圈——实测
    并帧对角线是逐帧包围盒中位对角线的 **1.89 倍**。于是资产模板被统一放大了 1.89 倍，
    而受控扫描说模板相对点云大 1.2 倍就越过对照、大 1.3 倍崩到 88.5°。

    也就是说那次「资产网格当模板，静止假转动 63.71°、RMSE 80.92°」量的不是资产
    模板，是一个被放大 1.89 倍的资产模板。拟合长方体那一臂换成逐帧中位之后
    RMSE 从 81.07° 掉到 12.42°，这一臂理应同样受益。
    """
    return _template_rotation_from_points(clouds, _asset_points(cls))


def _unflatten_along_ray(clouds: list, cam_positions: list, factor: float = 1.9) -> list:
    """**沿每一帧自己的相机视线**把点云拉回来，再去配准。

    这是「换判据」之外的另一条路：**动观测本身**。位姿这一级试过的判据
    （自由 ICP 互配、盒子模板、资产网格模板、真值位姿预言机、轮廓位姿、
    光流对应、可见性裁剪、运动模型拟合）全都失败于同一个根因——相机在动，
    逐帧采到的表面本身在变，于是「物体在转」永远比「没动」更能解释观测。
    换算法治不了这个，因为错在输入。

    而那个畸变是**量过的、有方向的**：VGGT 沿相机视线把物体压扁，最短轴只剩
    正确比例的 **0.526**（所以拉回来的系数 ≈ 1/0.526 ≈ 1.9）。关键是**视线方向
    逐帧不同**——相机在转，所以每一帧被压扁的方向不一样，拼起来就像物体在转。
    沿**各自的视线**各自拉回来，这个逐帧不一致就该被消掉。

    和既有的 `unflatten`（在 `_residual_pick_rate` 里）不是一回事：那条用 **PCA
    最短主轴**，而最短主轴会把「被 VGGT 压扁」和「物体本来就扁」（地砖、板子）
    混在一起。视线方向是外部已知量，不从被测数据里估，所以不会混。
    """
    if len(cam_positions) != len(clouds):
        return clouds
    out = []
    for cloud, cam in zip(clouds, cam_positions):
        q = np.asarray(cloud, float)
        if len(q) < 10:
            out.append(q); continue
        c = q.mean(0)
        ray = np.asarray(c, float) - np.asarray(cam, float)
        n = float(np.linalg.norm(ray))
        if n <= 1e-9:
            out.append(q); continue
        ray = ray / n
        centred = q - c
        along = centred @ ray                      # 沿视线的分量
        out.append(centred + np.outer(along * (factor - 1.0), ray) + c)
    return out


def _gt_poses(_run: Path) -> dict:
    """逐帧真值位姿（位置 + 朝向），按物体 id 再按帧序号索引。"""
    if CURRENT_CLIP_DIR is None:
        return {}
    states = CURRENT_CLIP_DIR / "gt_states.jsonl"
    if not states.is_file():
        return {}
    out: dict = {}
    for line in states.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        for obj in state.get("objects", []):
            out.setdefault(obj["id"], {})[int(state["i"])] = (list(obj["pos"]), list(obj["quat"]))
    return out


def _gt_quaternions(_run: Path) -> dict:
    """逐帧真值朝向，按物体 id 再按帧序号索引。"""
    if CURRENT_CLIP_DIR is None:
        print("  icp: 没拿到片段目录", flush=True)
        return {}
    states = CURRENT_CLIP_DIR / "gt_states.jsonl"
    if not states.is_file():
        print(f"  icp: {states} 不在", flush=True)
        return {}
    out: dict = {}
    for line in states.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        for obj in state.get("objects", []):
            out.setdefault(obj["id"], {})[int(state["i"])] = list(obj["quat"])
    return out


def mode_common(evidence: dict, truth: dict, kinds: dict, run: Path) -> dict:
    """拿背景估公共漂移，再从物体轨迹里减掉。

    漂移很可能是共模的：相机位姿误差会让画面里所有东西一起飘。背景（所有不属于任何
    检出物体的像素）按定义是不动的，所以逐帧配准背景量到的位移，就是公共漂移的直接
    估计。减掉它不需要先知道哪个物体静止——那正是现在做不到的事。

    只减平移分量。公共旋转也可能存在，但平移是漂移指标真正在量的东西，先看它够不够。
    """
    from gwm.perception.masks import resize_mask
    geom = load_geometry(run / "geometry.npz")
    masks = load_masks(run)
    frame_of = {f: i for i, f in enumerate(geom.frame_indices)}
    times = {f["frame_index"]: round(float(f["t"]), 3) for f in evidence.get("frames", [])}
    out: dict[str, list] = {"step/register": [], "step/common_removed": [],
                            "drift_static/register": [], "drift_static/common_removed": [],
                            "common_travel": []}

    # 背景点云：把所有物体掩码并起来再取反
    frames = sorted({fi for per in masks.values() for fi in per})
    bg_clouds, bg_stamps = [], []
    for fi in frames:
        k, t = frame_of.get(fi), times.get(fi)
        if k is None or t is None:
            continue
        shape = geom.depth[k].shape
        union = np.zeros(shape, bool)
        for per in masks.values():
            m = per.get(fi)
            if m is None:
                continue
            m = np.asarray(m).astype(bool)
            if m.shape != shape:
                m = resize_mask(m, (shape[1], shape[0]))
            union |= m
        pts = to_world(unproject(geom.depth[k], geom.intrinsics[k], ~union, stride=8),
                       _fix(geom.cam_to_world[k]))
        if len(pts) >= 200:
            bg_clouds.append(pts); bg_stamps.append(t)
    if len(bg_clouds) < 6:
        return out
    bg = track_rigid(bg_clouds)
    if not bg["ok"]:
        return out
    bg_pos = np.array(bg["positions"], float)
    common = {t: bg_pos[i] - bg_pos[0] for i, t in enumerate(bg_stamps)}
    out["common_travel"].append(float(np.linalg.norm(bg_pos.max(0) - bg_pos.min(0))))

    for oid, per_frame in masks.items():
        if oid not in truth:
            continue
        clouds, stamps = [], []
        for fi, mask in sorted(per_frame.items()):
            k, t = frame_of.get(fi), times.get(fi)
            if k is None or t is None or t not in truth[oid] or t not in common:
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
        registered = np.array(tracked["positions"], float)
        gt = np.stack([truth[oid][t] for t in stamps])
        gt_rel = gt - gt[0]
        reg_rel = registered - registered[0]
        corrected = reg_rel - np.stack([common[t] for t in stamps])
        out["step/register"].extend(np.linalg.norm(reg_rel - gt_rel, axis=1).tolist())
        out["step/common_removed"].extend(np.linalg.norm(corrected - gt_rel, axis=1).tolist())
        if kinds.get(oid) == "static":
            out["drift_static/register"].append(float(np.linalg.norm(reg_rel.max(0) - reg_rel.min(0))))
            out["drift_static/common_removed"].append(
                float(np.linalg.norm(corrected.max(0) - corrected.min(0))))
    return out


MODES = {"bias": mode_bias, "offset": mode_offset, "signal": mode_signal, "common": mode_common, "icp": mode_icp,
         "register": mode_register, "articulation": mode_articulation}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--decisions-from", default="",
                    help="已跑过的管线产出根（按片段名分子目录），二次配准从这里取 decisions")
    ap.add_argument("--perception", default="gt", choices=("gt", "gtcamseg", "gtseg", "neural"),
                    help="gt=全真值；gtseg=真值分割+真实几何（保住物体身份，只换深度）；"
                         "neural=全真实。残差挑资产的数都是全真值上测的，而管线端到端"
                         "是 0/78，必须单独验深度噪声这一条。")
    a = ap.parse_args()
    if a.decisions_from:
        globals()["DECISIONS_ROOT"] = Path(a.decisions_from)
        print(f"二次配准的 decisions 取自 {a.decisions_from}", flush=True)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in Path(a.rendered).iterdir() if (p / "clip.mp4").is_file())[:a.limit]
    merged: dict[str, list] = {}
    for clip_dir in clips:
        try:
            evidence = perceive(clip_dir, work, perception=a.perception)
        except Exception as error:                   # 一条挂了不该拖垮整轮
            print(f"{clip_dir.name}: 失败 {type(error).__name__}: {error}"[:200], flush=True)
            continue
        truth, kinds = gt_series(clip_dir)
        globals()["CURRENT_CLIP_DIR"] = clip_dir
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
        # 0/1 序列的中位数是退化的（非 0 即 1），要的是**均值**；而且必须连着
        # 「随机猜能猜到多少」一起读——从 N+1 个里随机挑是 1/(N+1)。这个坑记过。
        if "/own_in_top" in key:
            extra = f"  **覆盖率 {statistics.mean(values):.3f}**（候选集里有正确答案的比例）"
        if ("/residual_pick_of_" in key or "/residual_pick_hard_" in key
                or "/respick_cands" in key or "/d2pick_cands" in key
                or "/respick33" in key or "/respick_noise" in key
                or "/respick_unflat" in key or "/silpick_cands" in key
                or key.endswith("/silpick_mismatched")
                or key.endswith("/renderpick_33")
                or "/combined_top" in key or "/gt_in_top" in key
                or "/renderpick_mask" in key):
            n_plus = (33 if "_hard_33" in key
                      else 33 if (key.endswith("/respick33") or "/respick_noise" in key
                                  or "/respick_unflat" in key)
                      else int(key.rsplit("cands", 1)[-1])
                      if ("respick_cands" in key or "d2pick_cands" in key
                          or "silpick_cands" in key)
                      else 33 if (key.endswith("/silpick_mismatched")
                                  or key.endswith("/renderpick_33")
                                  or "/renderpick_mask" in key)
                      else int(key.rsplit("top", 1)[-1])
                      if ("/combined_top" in key or "/gt_in_top" in key)
                      else int(key.rsplit("_", 1)[-1]))
            extra = (f"  **挑对的比例 {statistics.mean(values):.3f}**"
                     f"（随机猜 {1.0 / n_plus:.3f}，即 {statistics.mean(values) * n_plus:.1f} 倍）")
        # 静止转角那几列，中位数对改动不敏感（样本少，中间那个不动就看不出变化），
        # 而决策真正要的是**有多少比例低于铰链门槛 8 度**——高于门槛的会产出假关节。
        if key.startswith(("rot_static/", "rot_rotating/", "rot_translating/")):
            # 门槛这条线对三个桶的含义是**相反的**：静止和纯平移的物体该低于门槛
            # （高于就是假关节），而真在转的物体不该低于（低于就是真转动被压没了）。
            # 所以这一列必须连着桶名一起读。
            below = sum(1 for v in values if v < 8.0) / len(values)
            want = "该高" if key.startswith("rot_rotating/") else "该低"
            extra += (f"  均值 {statistics.mean(values):>6.2f}"
                      f"  低于 8 度门槛 {below:.3f}（{want}）")
        # **四分位一并报。** 只给中位数会掩盖分布——这个项目里栽过三次（双峰、
        # 散开六倍都被一个中位数抹平）。而现在有两个结论直接取决于散度：
        # rot_per_cam_deg 要「稳定」才说明机制解释得了假转动，cam_total_deg 要
        # 「散得开」这个回归才不退化。
        q1, q3 = ((np.percentile(values, 25), np.percentile(values, 75))
                  if len(values) >= 4 else (float("nan"), float("nan")))
        print(f"  {key:<26} 中位数 {statistics.median(values):>8.3f}   样本 {len(values)}"
              f"   四分位 {q1:>7.3f}/{q3:<7.3f}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
