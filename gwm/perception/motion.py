"""Model-independent motion estimation in an explicit y-up world frame."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULTS = {
    "min_support_frames": 5,
    "min_temporal_span_s": 0.75,
    "max_missing_fraction": 0.50,
    "min_mean_visible_fraction": 0.50,
    "max_gap_factor": 4.0,
    "min_primary_confidence": 0.55,
    "min_candidate_confidence": 0.15,
    "min_hypothesis_margin": 0.10,
    "static_pos_thresh_m": 0.05,
    "static_pos_frac_of_size": 0.15,
    "static_rot_thresh_deg": 3.0,
    "translation_min_displacement_m": 0.08,
    "translation_residual_frac": 0.12,
    "rotation_min_angle_deg": 12.0,
    "rotation_residual_deg": 5.0,
    "revolute_min_radius_m": 0.10,
    "revolute_residual_frac": 0.12,
    "revolute_axis_stability": 0.85,
    "periodic_min_cycles": 1.5,
    "periodic_min_period_s": 0.25,
    "periodic_min_samples_per_cycle": 4.0,
    "periodic_min_r2": 0.70,
    "periodic_frequency_steps": 160,
    "smooth_window": 1,
}


def _settings(cfg: dict) -> dict:
    supplied = ((cfg.get("perception") or {}).get("motion") or {}) if "perception" in cfg else (cfg.get("motion") or cfg)
    out = {**DEFAULTS, **supplied}
    positive = (
        "min_support_frames", "min_temporal_span_s", "max_gap_factor", "static_pos_thresh_m",
        "static_rot_thresh_deg", "translation_min_displacement_m", "translation_residual_frac",
        "rotation_min_angle_deg", "rotation_residual_deg", "periodic_min_cycles",
        "periodic_min_period_s", "periodic_min_samples_per_cycle", "periodic_frequency_steps", "revolute_min_radius_m",
        "revolute_residual_frac",
    )
    if any(isinstance(out[key], bool) or not isinstance(out[key], (int, float)) or not math.isfinite(out[key]) or out[key] <= 0 for key in positive):
        raise ValueError("motion evidence thresholds must be finite and positive")
    ratios = ("max_missing_fraction", "min_mean_visible_fraction", "min_primary_confidence", "min_candidate_confidence", "min_hypothesis_margin", "periodic_min_r2", "revolute_axis_stability")
    if any(isinstance(out[key], bool) or not isinstance(out[key], (int, float)) or not 0 <= out[key] <= 1 for key in ratios):
        raise ValueError("motion confidence and coverage thresholds must be in [0, 1]")
    return out


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values
    window = min(int(window) | 1, len(values) if len(values) % 2 else len(values) - 1)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])], axis=1)


def _validate_inputs(times, positions, quaternions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.asarray(times, dtype=float)
    pos = np.asarray(positions, dtype=float)
    quat = np.asarray(quaternions, dtype=float)
    if ts.ndim != 1 or pos.shape != (len(ts), 3) or quat.shape != (len(ts), 4):
        raise ValueError("motion observations require times[N], positions[N,3], quaternions[N,4]")
    if not np.isfinite(ts).all() or not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("motion observations must be finite")
    if len(ts) and np.any(np.diff(ts) <= 0):
        raise ValueError("motion observation times must be strictly increasing")
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms <= 1e-9):
        raise ValueError("motion observation quaternions must be non-zero")
    return ts, pos, quat / norms[:, None]


def to_world_observations(
    positions: np.ndarray,
    quaternions: np.ndarray,
    coordinate_space: str,
    camera_positions: np.ndarray | None = None,
    camera_quaternions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return y-up world poses; camera inputs use three.js local axes and xyzw quaternions."""
    if coordinate_space == "world":
        return positions.copy(), quaternions.copy()
    if coordinate_space != "camera":
        raise ValueError("coordinate_space must be 'world' or 'camera'")
    if camera_positions is None or camera_quaternions is None:
        raise ValueError("camera-space observations require camera positions and quaternions")
    cam_pos = np.asarray(camera_positions, dtype=float)
    cam_quat = np.asarray(camera_quaternions, dtype=float)
    if cam_pos.shape != positions.shape or cam_quat.shape != quaternions.shape or not np.isfinite(cam_pos).all() or not np.isfinite(cam_quat).all():
        raise ValueError("camera poses must match motion observations and be finite")
    cam_norm = np.linalg.norm(cam_quat, axis=1)
    if np.any(cam_norm <= 1e-9):
        raise ValueError("camera quaternions must be non-zero")
    cam_rot = R.from_quat(cam_quat / cam_norm[:, None])
    world_pos = cam_rot.apply(positions) + cam_pos
    world_quat = (cam_rot * R.from_quat(quaternions)).as_quat()
    return world_pos, world_quat


def _candidate(kind: str, confidence: float, residual: float, **parameters: Any) -> dict:
    return {"type": kind, "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 6),
            "residual": round(float(max(residual, 0.0)), 6), **parameters}


def _orientation_signal(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rotations = R.from_quat(quaternions)
    relative = rotations[0].inv() * rotations
    vectors = relative.as_rotvec()
    if np.linalg.norm(vectors) < 1e-12:
        return np.zeros(len(vectors)), np.array([0.0, 1.0, 0.0]), 1.0
    _u, singular, vt = np.linalg.svd(vectors - vectors.mean(axis=0), full_matrices=False)
    axis = vt[0]
    angles = vectors @ axis
    if len(angles) > 1 and angles[-1] < angles[0]:
        axis, angles = -axis, -angles
    stability = float(singular[0] ** 2 / max(float((singular ** 2).sum()), 1e-12))
    return angles, axis, stability


def _sinusoid(times: np.ndarray, signal: np.ndarray, settings: dict) -> dict | None:
    span = float(times[-1] - times[0])
    if span <= 0 or np.ptp(signal) <= 1e-9:
        return None
    min_period = max(float(settings["periodic_min_period_s"]),
                     float(settings["periodic_min_samples_per_cycle"]) * float(np.median(np.diff(times))))
    max_period = span / float(settings["periodic_min_cycles"])
    if max_period < min_period:
        return None
    periods = np.linspace(min_period, max_period, int(settings["periodic_frequency_steps"]))
    centered = signal - signal.mean()
    total = float(centered @ centered)
    best = None
    for period in periods:
        omega = 2 * math.pi / period
        design = np.column_stack([np.sin(omega * times), np.cos(omega * times), np.ones(len(times))])
        coeff, *_ = np.linalg.lstsq(design, signal, rcond=None)
        residuals = signal - design @ coeff
        mse = float(np.mean(residuals ** 2))
        r2 = 1.0 - float(residuals @ residuals) / max(total, 1e-12)
        if best is None or mse < best["mse"]:
            best = {"period": float(period), "amp": float(math.hypot(coeff[0], coeff[1])),
                    "phase": float(math.atan2(coeff[1], coeff[0])), "mse": mse,
                    "rmse": math.sqrt(max(mse, 0.0)), "r2": float(np.clip(r2, 0.0, 1.0))}
    return best


def _hypothesis(candidates: list[dict]) -> dict | None:
    if len(candidates) < 2:
        return None
    return {"kind": "motion", "candidates": [
        {"value": candidate["type"], "confidence": candidate["confidence"], "source": "multi_frame_motion_estimator"}
        for candidate in candidates
    ]}


def _unknown(reason: str, support: dict, candidates: list[dict] | None = None) -> dict:
    ranked = sorted(candidates or [], key=lambda candidate: (-candidate["confidence"], candidate["type"]))
    margin = ranked[0]["confidence"] - ranked[1]["confidence"] if len(ranked) > 1 else "unknown"
    return {
        "motion_guess": {"type": "unknown", "conf": "unknown", "support_frames": support["frames"],
                         "temporal_span_s": support["span"], "residual": "unknown",
                         "candidate_margin": margin, "coordinate_space": "world", "notes": reason,
                         "missing_fraction": support["missing_fraction"], "max_gap_s": support["max_gap_s"],
                         "mean_visible_fraction": support["mean_visible_fraction"],
                         "candidate_details": ranked},
        "hypothesis": _hypothesis(ranked),
    }


def estimate_motion(
    times,
    positions,
    quaternions,
    cfg: dict,
    *,
    size: np.ndarray | None = None,
    coordinate_space: str = "world",
    camera_positions: np.ndarray | None = None,
    camera_quaternions: np.ndarray | None = None,
    visible_fractions: np.ndarray | None = None,
    camera_pose_available: bool = True,
    orientation_reliable: bool = True,
) -> dict:
    """Fit competing existing motion types and return a primary guess plus alternatives."""
    settings = _settings(cfg)
    ts, local_pos, local_quat = _validate_inputs(times, positions, quaternions)
    support = {"frames": int(len(ts)), "span": round(float(ts[-1] - ts[0]), 6) if len(ts) > 1 else 0.0,
               "missing_fraction": 0.0, "max_gap_s": 0.0, "mean_visible_fraction": "unknown"}
    if visible_fractions is not None:
        visibility = np.asarray(visible_fractions, dtype=float)
        if visibility.shape != ts.shape or not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
            raise ValueError("visible_fractions must match observations and lie in [0, 1]")
        support["mean_visible_fraction"] = round(float(visibility.mean()), 6) if len(visibility) else "unknown"
    if not camera_pose_available or (coordinate_space == "camera" and (camera_positions is None or camera_quaternions is None)):
        return _unknown("camera pose missing; camera and object motion cannot be separated", support)
    world_pos, world_quat = to_world_observations(local_pos, local_quat, coordinate_space, camera_positions, camera_quaternions)
    if len(ts) < int(settings["min_support_frames"]) or support["span"] < float(settings["min_temporal_span_s"]):
        return _unknown("insufficient frames or temporal span", support)

    gaps = np.diff(ts)
    median_gap = float(np.median(gaps))
    expected = max(len(ts), int(round(support["span"] / max(median_gap, 1e-9))) + 1)
    support["missing_fraction"] = round(max(0.0, 1.0 - len(ts) / expected), 6)
    support["max_gap_s"] = round(float(gaps.max()), 6)
    severe_gap = gaps.max() > float(settings["max_gap_factor"]) * median_gap
    severe_missing = support["missing_fraction"] > float(settings["max_missing_fraction"])
    severe_occlusion = isinstance(support["mean_visible_fraction"], float) and support["mean_visible_fraction"] < float(settings["min_mean_visible_fraction"])

    pos = _smooth(world_pos, int(settings["smooth_window"]))
    size_diag = float(np.linalg.norm(size)) if size is not None else 0.0
    static_scale = max(float(settings["static_pos_thresh_m"]), float(settings["static_pos_frac_of_size"]) * size_diag, 1e-6)
    centered = pos - pos.mean(axis=0)
    static_pos_rmse = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    angles, rotation_axis, axis_stability = _orientation_signal(world_quat)
    rotation_deg = np.degrees(angles)
    static_rot_rmse = float(np.sqrt(np.mean((rotation_deg - rotation_deg.mean()) ** 2)))
    static_score = math.exp(-0.5 * (static_pos_rmse / static_scale) ** 2 - 0.5 * (static_rot_rmse / float(settings["static_rot_thresh_deg"])) ** 2)
    if not orientation_reliable:
        static_score = 0.0  # position alone cannot prove absence of rotation
    candidates = [_candidate("static", static_score, static_pos_rmse, residual_unit="m",
                             position_residual_m=round(static_pos_rmse, 6), rotation_residual_deg=round(static_rot_rmse, 6))]

    design = np.column_stack([ts - ts[0], np.ones(len(ts))])
    coeff, *_ = np.linalg.lstsq(design, pos, rcond=None)
    velocity, origin = coeff[0], coeff[1]
    linear_fit = design @ coeff
    linear_rmse = float(np.sqrt(np.mean(np.sum((pos - linear_fit) ** 2, axis=1))))
    displacement = float(np.linalg.norm(linear_fit[-1] - linear_fit[0]))
    linear_scale = max(float(settings["translation_residual_frac"]) * max(displacement, static_scale), static_scale * 0.25, 1e-6)
    motion_strength = 1.0 - math.exp(-displacement / max(float(settings["translation_min_displacement_m"]), 1e-6))
    linear_score = motion_strength * math.exp(-0.5 * (linear_rmse / linear_scale) ** 2)
    if orientation_reliable:
        linear_score *= math.exp(-0.5 * (static_rot_rmse / float(settings["static_rot_thresh_deg"])) ** 2)
    speed = float(np.linalg.norm(velocity))
    axis = velocity / speed if speed > 1e-9 else np.array([1.0, 0.0, 0.0])
    candidates.append(_candidate("prismatic", linear_score, linear_rmse, residual_unit="m", axis=[float(value) for value in axis], rate=speed,
                                 range=[0.0, displacement]))

    angle_design = np.column_stack([ts - ts[0], np.ones(len(ts))])
    angle_coeff, *_ = np.linalg.lstsq(angle_design, rotation_deg, rcond=None)
    angle_fit = angle_design @ angle_coeff
    angle_rmse = float(np.sqrt(np.mean((rotation_deg - angle_fit) ** 2)))
    angle_range = float(np.ptp(rotation_deg))
    rotation_strength = 1.0 - math.exp(-angle_range / max(float(settings["rotation_min_angle_deg"]), 1e-6))
    rotation_score = rotation_strength * axis_stability * math.exp(-0.5 * (angle_rmse / float(settings["rotation_residual_deg"])) ** 2)
    rotation_score *= math.exp(-0.5 * (static_pos_rmse / static_scale) ** 2)
    if orientation_reliable:
        candidates.append(_candidate("spin", rotation_score, angle_rmse, residual_unit="deg", axis=[float(value) for value in rotation_axis],
                                     rate=float(angle_coeff[0]), range_deg=[float(rotation_deg.min()), float(rotation_deg.max())]))

    if orientation_reliable and axis_stability >= float(settings["revolute_axis_stability"]) and angle_range >= float(settings["rotation_min_angle_deg"]):
        basis_u = np.cross(rotation_axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(basis_u) < 1e-6:
            basis_u = np.cross(rotation_axis, np.array([0.0, 1.0, 0.0]))
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(rotation_axis, basis_u)
        projected_positions = np.column_stack([pos @ basis_u, pos @ basis_v])
        circle_design = np.column_stack([2 * projected_positions, np.ones(len(ts))])
        circle_target = np.sum(projected_positions ** 2, axis=1)
        fitted, *_ = np.linalg.lstsq(circle_design, circle_target, rcond=None)
        center_2d = fitted[:2]
        radius = math.sqrt(max(float(fitted[2] + center_2d @ center_2d), 0.0))
        arc_extent = float(np.max(np.linalg.norm(projected_positions - projected_positions[0], axis=1)))
        radial_error = float(np.sqrt(np.mean((np.linalg.norm(projected_positions - center_2d, axis=1) - radius) ** 2)))
        axial_values = pos @ rotation_axis
        axial_error = float(np.std(axial_values))
        arc_residual = math.hypot(radial_error, axial_error)
        if radius >= float(settings["revolute_min_radius_m"]) and arc_extent >= float(settings["revolute_min_radius_m"]):
            pivot = center_2d[0] * basis_u + center_2d[1] * basis_v + axial_values.mean() * rotation_axis
            revolute_score = rotation_strength * axis_stability
            revolute_score *= math.exp(-0.5 * (arc_residual / max(radius * float(settings["revolute_residual_frac"]), 1e-6)) ** 2)
            revolute_score *= math.exp(-0.5 * (angle_rmse / float(settings["rotation_residual_deg"])) ** 2)
            candidates.append(_candidate("revolute", revolute_score, arc_residual, residual_unit="m",
                                         axis=[float(value) for value in rotation_axis], pivot=[float(value) for value in pivot],
                                         range=[float(rotation_deg.min()), float(rotation_deg.max())]))

    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    projection_axis = vt[0] if len(vt) else np.array([1.0, 0.0, 0.0])
    projected = centered @ projection_axis
    periodic = _sinusoid(ts, projected, settings)
    if periodic is not None and periodic["r2"] >= float(settings["periodic_min_r2"]):
        periodic_score = periodic["r2"] * min(1.0, support["span"] / max(periodic["period"] * float(settings["periodic_min_cycles"]), 1e-9))
        candidates.append(_candidate("periodic_translate", periodic_score, periodic["rmse"], residual_unit="m",
                                     axis=[float(value) for value in projection_axis], period=periodic["period"],
                                     amp=periodic["amp"], phase=periodic["phase"], cycles=support["span"] / periodic["period"]))
    periodic_rotation = _sinusoid(ts, rotation_deg, settings)
    if orientation_reliable and periodic_rotation is not None and periodic_rotation["r2"] >= float(settings["periodic_min_r2"]):
        periodic_rotation_score = periodic_rotation["r2"] * axis_stability * min(1.0, support["span"] / max(periodic_rotation["period"] * float(settings["periodic_min_cycles"]), 1e-9))
        candidates.append(_candidate("periodic_rotate", periodic_rotation_score, periodic_rotation["rmse"], residual_unit="deg",
                                     axis=[float(value) for value in rotation_axis], period=periodic_rotation["period"],
                                     amp_deg=periodic_rotation["amp"], phase=periodic_rotation["phase"],
                                     cycles=support["span"] / periodic_rotation["period"]))

    ordered = sorted(candidates, key=lambda candidate: (-candidate["confidence"], candidate["type"]))
    ranked = [candidate for candidate in ordered if candidate["confidence"] >= float(settings["min_candidate_confidence"])]
    if len(ranked) < 2:
        ranked = ordered[:2]
    margin = ranked[0]["confidence"] - ranked[1]["confidence"] if len(ranked) > 1 else 1.0
    if severe_gap or severe_missing or severe_occlusion:
        reason = "severe track gaps" if severe_gap else ("too many missing observations" if severe_missing else "severe occlusion")
        return _unknown(reason, support, ranked)
    if ranked[0]["confidence"] < float(settings["min_primary_confidence"]) or margin < float(settings["min_hypothesis_margin"]):
        reason = "object orientation is unreliable" if not orientation_reliable and static_pos_rmse < static_scale else "motion hypotheses are not sufficiently separated"
        return _unknown(reason, support, ranked)

    best = ranked[0]
    guess = {key: value for key, value in best.items() if key != "confidence"}
    guess.update({"conf": best["confidence"], "support_frames": support["frames"], "temporal_span_s": support["span"],
                  "candidate_margin": round(float(margin), 6), "coordinate_space": "world",
                  "missing_fraction": support["missing_fraction"], "max_gap_s": support["max_gap_s"],
                  "mean_visible_fraction": support["mean_visible_fraction"],
                  "candidate_details": ranked, "notes": "competing hypotheses fitted in aligned world coordinates"})
    return {"motion_guess": guess, "hypothesis": _hypothesis(ranked)}


def classify_motion_legacy(times, centers, yaws, cfg: dict, size=None, class_name: str = "", cam_pos=None) -> dict:
    """Compatibility adapter for the original yaw-based classify_motion API."""
    del class_name, cam_pos  # legacy hints no longer override measured motion evidence
    yaw = np.asarray(yaws, dtype=float)
    quaternions = R.from_euler("y", yaw).as_quat() if yaw.ndim == 1 else np.asarray(yaws, dtype=float)
    return estimate_motion(times, centers, quaternions, cfg, size=size, coordinate_space="world")["motion_guess"]
