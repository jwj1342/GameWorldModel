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
        # 物体在画面里多大，用来分桶。对应关系方法在小物体上必然差（孔径问题 +
        # 邻域平滑），所以报误差之前先把尺寸摆出来——掩码中位只有 264 像素时，
        # 「方法不行」和「物体太小」是两件事。
        _areas = [int(np.asarray(per_frame[f]).astype(bool).sum())
                  for f in frames if f in per_frame]
        _med_area = float(np.median(_areas)) if _areas else 0.0
        _bucket = ("小 <200px" if _med_area < 200 else
                   "中 200~600px" if _med_area < 600 else "大 >600px")
        out.setdefault(f"area/{_bucket}", []).append(_med_area)
        if len(dense_deg) == len(true_deg):
            out.setdefault(f"bysize/{_bucket}/dense_rmse_deg", []).append(
                float(np.sqrt(np.mean((dense_deg - true_deg) ** 2))))
        out.setdefault(f"bysize/{_bucket}/icp_rmse_deg", []).append(
            float(np.sqrt(np.mean((solved_deg - true_deg) ** 2))))
        out.setdefault(f"{kind}/true_range_deg", []).append(float(np.ptp(true_deg)))
        out.setdefault(f"{kind}/solved_range_deg", []).append(float(np.ptp(solved_deg)))
        out.setdefault(f"{kind}/angle_rmse_deg", []).append(
            float(np.sqrt(np.mean((solved_deg - true_deg) ** 2))))
        if len(template_deg) == len(true_deg):
            out.setdefault(f"{kind}/template_range_deg", []).append(float(np.ptp(template_deg)))
            out.setdefault(f"{kind}/template_rmse_deg", []).append(
                float(np.sqrt(np.mean((template_deg - true_deg) ** 2))))
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


def _asset_template_rotation_deg(clouds: list, cls: str) -> np.ndarray:
    """拿检索到的资产网格当模板，逐帧把观测配上去。

    资产的尺度和物体未必一致（检索只保证类别像），所以配准前先按两边的包围盒
    对角线把模板缩到同一尺度——ICP 解的是刚体变换，尺度不对会被旋转和平移吸收。
    """
    from gwm.perception.registration import icp
    template = _asset_points(cls)
    if template is None or not clouds:
        return np.array([])
    span = np.vstack(clouds)
    obs_diag = float(np.linalg.norm(span.max(0) - span.min(0)))
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
        print(f"  {key:<26} 中位数 {statistics.median(values):>8.3f}   样本 {len(values)}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
