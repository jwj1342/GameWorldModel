"""Model-independent Evidence quality metrics and configurable gate."""
from __future__ import annotations

import math
from typing import Any

from .contract import UNKNOWN, validate_evidence


_THRESHOLD_KEYS = {
    "min_object_observation_coverage",
    "min_bbox_coverage",
    "min_mask_coverage",
    "min_depth_coverage",
    "min_visibility_coverage",
    "max_track_gap_frames",
    "min_object_temporal_span_ratio",
    "min_camera_frame_coverage",
    "min_camera_temporal_span_ratio",
    "max_attribute_unknown_ratio",
    "max_identity_ambiguity",
    "min_motion_support",
    "severe_occlusion_visible_fraction",
    "max_severe_occlusion_fraction",
}


def _diagnostic(path: str, code: str, message: str, suggestion: str = "", severity: str = "warning") -> dict:
    return {
        "path": path,
        "code": code,
        "message": message,
        "suggestion": suggestion,
        "stage": "evidence_quality",
        "severity": severity,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _thresholds(config: dict | None) -> tuple[dict, list[dict]]:
    if not isinstance(config, dict):
        return {}, [_diagnostic(
            "/quality/thresholds", "quality_thresholds_missing",
            "Evidence quality thresholds were not supplied; only structural blocking is active",
            "pass the project config or an evidence_quality threshold mapping",
        )]
    values = config.get("evidence_quality") if "evidence_quality" in config else config
    if not isinstance(values, dict):
        values = {}
    missing = sorted(_THRESHOLD_KEYS - set(values))
    diagnostics = []
    if missing:
        diagnostics.append(_diagnostic(
            "/quality/thresholds", "quality_thresholds_incomplete",
            "quality thresholds are missing: " + ", ".join(missing),
            "define every threshold in evidence_quality; missing checks are skipped",
        ))
    valid = {}
    for key in sorted(_THRESHOLD_KEYS & set(values)):
        number = _number(values[key])
        in_range = number is not None and (number >= 0 if key == "max_track_gap_frames" else 0 <= number <= 1)
        if key == "max_track_gap_frames":
            in_range = in_range and float(number).is_integer()
        if not in_range:
            diagnostics.append(_diagnostic(
                f"/quality/thresholds/{key}", "invalid_quality_threshold",
                f"quality threshold '{key}' has invalid value {values[key]!r}",
                "use an integer >= 0 for max_track_gap_frames and ratios in [0, 1] for other thresholds",
            ))
            continue
        valid[key] = int(number) if key == "max_track_gap_frames" else number
    return valid, diagnostics


def _identity_ambiguity(hypotheses: list[dict]) -> dict:
    identity = [hypothesis for hypothesis in hypotheses if hypothesis.get("kind") == "identity"]
    if not identity:
        return {"hypothesis_count": 0, "max_candidate_count": 0, "normalized_entropy": 0.0}
    entropies: list[float] = []
    max_candidates = 0
    unknown_confidence = False
    for hypothesis in identity:
        candidates = hypothesis.get("candidates") or []
        max_candidates = max(max_candidates, len(candidates))
        scores = [_number(candidate.get("confidence")) for candidate in candidates]
        if any(score is None for score in scores):
            unknown_confidence = True
            continue
        total = sum(scores)
        if len(scores) < 2 or total <= 0:
            unknown_confidence = True
            continue
        probabilities = [score / total for score in scores]
        entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
        entropies.append(entropy / math.log(len(probabilities)))
    return {
        "hypothesis_count": len(identity),
        "max_candidate_count": max_candidates,
        "normalized_entropy": round(max(entropies), 6) if entropies else UNKNOWN,
        "has_unknown_confidence": unknown_confidence,
    }


def _motion_support(obj: dict) -> dict:
    motion_guess = obj.get("motion_guess") or {}
    primary = _number((obj.get("attribute_confidence") or {}).get("motion"))
    if primary is None:
        primary = _number(motion_guess.get("conf"))
    motion_hypotheses = [hypothesis for hypothesis in obj.get("hypotheses", []) if hypothesis.get("kind") == "motion"]
    candidate_scores = [
        score
        for hypothesis in motion_hypotheses
        for candidate in hypothesis.get("candidates", [])
        if (score := _number(candidate.get("confidence"))) is not None
    ]
    ordered = sorted(candidate_scores, reverse=True)
    return {
        "primary_confidence": round(primary, 6) if primary is not None else UNKNOWN,
        "support_frames": int(motion_guess["support_frames"]) if isinstance(motion_guess.get("support_frames"), int) else UNKNOWN,
        "temporal_span_s": round(float(motion_guess["temporal_span_s"]), 6) if _number(motion_guess.get("temporal_span_s")) is not None else UNKNOWN,
        "residual": round(float(motion_guess["residual"]), 6) if _number(motion_guess.get("residual")) is not None else UNKNOWN,
        "residual_unit": motion_guess.get("residual_unit", UNKNOWN),
        "hypothesis_count": len(motion_hypotheses),
        "best_candidate_confidence": round(ordered[0], 6) if ordered else UNKNOWN,
        "candidate_margin": (round(float(motion_guess["candidate_margin"]), 6)
                             if _number(motion_guess.get("candidate_margin")) is not None
                             else (round(ordered[0] - ordered[1], 6) if len(ordered) > 1 else UNKNOWN)),
    }


def _object_metrics(obj: dict, frames: list[dict], frame_order: dict[int, int], duration: float, thresholds: dict) -> dict:
    observations = obj.get("observations") or []
    known_observations = [observation for observation in observations if isinstance(observation.get("frame_index"), int)]
    unique_frames = sorted({observation["frame_index"] for observation in known_observations if observation["frame_index"] in frame_order}, key=frame_order.get)
    positions = [frame_order[frame_index] for frame_index in unique_frames]
    gaps = [b - a - 1 for a, b in zip(positions, positions[1:]) if b - a > 1]
    times = sorted(float(observation["t"]) for observation in observations)
    span = times[-1] - times[0] if len(times) > 1 else 0.0

    measured = {
        field: _ratio(sum(observation.get(field) != UNKNOWN for observation in observations), len(observations))
        for field in ("bbox", "mask_ref", "depth", "visible_fraction")
    }
    visibility_values = [
        value for observation in observations
        if (value := _number(observation.get("visible_fraction"))) is not None
    ]
    occlusion_cutoff = _number(thresholds.get("severe_occlusion_visible_fraction"))
    severe_count = sum(value <= occlusion_cutoff for value in visibility_values) if occlusion_cutoff is not None else None

    attributes = obj.get("attribute_confidence") or {}
    unknown_attributes = sum(value == UNKNOWN for value in attributes.values())
    return {
        "object_id": obj.get("id"),
        "track_id": obj.get("track_id"),
        "observation_count": len(observations),
        "known_observation_frame_count": len(unique_frames),
        "registered_frame_count": len(frames),
        "observation_coverage": _ratio(len(unique_frames), len(frames)),
        "measured_coverage": measured,
        "visibility": {
            "mean_visible_fraction": round(sum(visibility_values) / len(visibility_values), 6) if visibility_values else UNKNOWN,
            "severe_occlusion_fraction": _ratio(severe_count, len(visibility_values)) if visibility_values and severe_count is not None else UNKNOWN,
        },
        "track": {
            "gap_count": len(gaps),
            "max_gap_frames": max(gaps) if gaps else 0,
            "temporal_span_s": round(span, 6),
            "temporal_span_ratio": round(min(1.0, span / duration), 6) if duration > 0 else 0.0,
        },
        "attribute_unknown_ratio": _ratio(unknown_attributes, len(attributes)),
        "attribute_unknown_count": unknown_attributes,
        "attribute_count": len(attributes),
        "identity_ambiguity": _identity_ambiguity(obj.get("hypotheses") or []),
        "motion_support": _motion_support(obj),
    }


def _camera_metrics(evidence: dict) -> dict:
    frames = evidence.get("frames") or []
    poses = (evidence.get("camera") or {}).get("poses") or []
    duration = _number((evidence.get("meta") or {}).get("duration")) or 0.0
    times = sorted(float(pose["t"]) for pose in poses)
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    return {
        "pose_count": len(poses),
        "frame_coverage": round(min(1.0, _ratio(len(poses), len(frames))), 6),
        "temporal_span_s": round(span, 6),
        "temporal_span_ratio": round(min(1.0, span / duration), 6) if duration > 0 else 0.0,
    }


def _aggregate_metrics(objects: list[dict]) -> dict:
    if not objects:
        return {
            "object_count": 0,
            "mean_observation_coverage": 0.0,
            "measured_coverage": {field: 0.0 for field in ("bbox", "mask_ref", "depth", "visible_fraction")},
            "mean_attribute_unknown_ratio": 0.0,
        }
    return {
        "object_count": len(objects),
        "mean_observation_coverage": round(sum(obj["observation_coverage"] for obj in objects) / len(objects), 6),
        "measured_coverage": {
            field: round(sum(obj["measured_coverage"][field] for obj in objects) / len(objects), 6)
            for field in ("bbox", "mask_ref", "depth", "visible_fraction")
        },
        "mean_attribute_unknown_ratio": round(sum(obj["attribute_unknown_ratio"] for obj in objects) / len(objects), 6),
    }


def _association_metrics(evidence: dict, objects: list[dict]) -> dict:
    diagnostics = (evidence.get("meta") or {}).get("association_diagnostics") or []
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        code = str(diagnostic.get("code") or "unknown")
        counts[code] = counts.get(code, 0) + 1
    return {
        "track_count": len(objects),
        "tracks_with_identity_hypotheses": sum(obj["identity_ambiguity"]["hypothesis_count"] > 0 for obj in objects),
        "diagnostic_counts": counts,
    }


def _below(diagnostics: list[dict], path: str, code: str, value: float, threshold: Any, label: str) -> None:
    limit = _number(threshold)
    if limit is not None and value < limit:
        diagnostics.append(_diagnostic(
            path, code, f"{label} {value:.3f} is below configured minimum {limit:.3f}",
            "preserve the limitation for downstream synthesis or improve evidence acquisition",
        ))


def _above(diagnostics: list[dict], path: str, code: str, value: float, threshold: Any, label: str) -> None:
    limit = _number(threshold)
    if limit is not None and value > limit:
        diagnostics.append(_diagnostic(
            path, code, f"{label} {value:.3f} exceeds configured maximum {limit:.3f}",
            "preserve alternative hypotheses or request stronger evidence",
        ))


def _quality_diagnostics(metrics: dict, thresholds: dict) -> list[dict]:
    diagnostics: list[dict] = []
    if metrics["aggregate"]["object_count"] == 0:
        diagnostics.append(_diagnostic(
            "/quality/objects", "no_evidence_objects",
            "Evidence contains no object tracks", "proceed only with camera/static synthesis or acquire object evidence",
        ))
    camera = metrics["camera"]
    _below(diagnostics, "/quality/camera/frame_coverage", "low_camera_frame_coverage", camera["frame_coverage"], thresholds.get("min_camera_frame_coverage"), "camera frame coverage")
    _below(diagnostics, "/quality/camera/temporal_span_ratio", "short_camera_span", camera["temporal_span_ratio"], thresholds.get("min_camera_temporal_span_ratio"), "camera temporal span")

    coverage_thresholds = {
        "bbox": "min_bbox_coverage",
        "mask_ref": "min_mask_coverage",
        "depth": "min_depth_coverage",
        "visible_fraction": "min_visibility_coverage",
    }
    for index, obj in enumerate(metrics["objects"]):
        base = f"/quality/objects/{index}"
        _below(diagnostics, f"{base}/observation_coverage", "low_object_coverage", obj["observation_coverage"], thresholds.get("min_object_observation_coverage"), "object observation coverage")
        for field, threshold_key in coverage_thresholds.items():
            _below(diagnostics, f"{base}/measured_coverage/{field}", f"low_{field}_coverage", obj["measured_coverage"][field], thresholds.get(threshold_key), f"{field} measured coverage")

        max_gap = thresholds.get("max_track_gap_frames")
        if _number(max_gap) is not None and obj["track"]["max_gap_frames"] > float(max_gap):
            diagnostics.append(_diagnostic(
                f"{base}/track/max_gap_frames", "track_break",
                f"track has a gap of {obj['track']['max_gap_frames']} frames; configured maximum is {int(max_gap)}",
                "retain the discontinuity and avoid assuming continuous identity or motion",
            ))
        _below(diagnostics, f"{base}/track/temporal_span_ratio", "short_track_span", obj["track"]["temporal_span_ratio"], thresholds.get("min_object_temporal_span_ratio"), "track temporal span")
        _above(diagnostics, f"{base}/attribute_unknown_ratio", "high_attribute_unknown_ratio", obj["attribute_unknown_ratio"], thresholds.get("max_attribute_unknown_ratio"), "attribute unknown ratio")

        ambiguity = obj["identity_ambiguity"]["normalized_entropy"]
        if ambiguity == UNKNOWN and obj["identity_ambiguity"]["hypothesis_count"]:
            diagnostics.append(_diagnostic(
                f"{base}/identity_ambiguity", "identity_ambiguity_unknown",
                "identity alternatives exist but their ambiguity cannot be quantified from known confidences",
                "preserve all candidates for downstream reasoning",
            ))
        elif ambiguity != UNKNOWN:
            _above(diagnostics, f"{base}/identity_ambiguity/normalized_entropy", "high_identity_ambiguity", ambiguity, thresholds.get("max_identity_ambiguity"), "identity ambiguity")

        motion_support = obj["motion_support"]["primary_confidence"]
        if motion_support == UNKNOWN:
            diagnostics.append(_diagnostic(
                f"{base}/motion_support", "motion_support_unknown",
                "motion hypothesis support is unknown", "do not commit to a precise motion program without additional evidence",
            ))
        else:
            _below(diagnostics, f"{base}/motion_support/primary_confidence", "low_motion_support", motion_support, thresholds.get("min_motion_support"), "motion support")

        occlusion = obj["visibility"]["severe_occlusion_fraction"]
        if occlusion != UNKNOWN:
            _above(diagnostics, f"{base}/visibility/severe_occlusion_fraction", "severe_occlusion", occlusion, thresholds.get("max_severe_occlusion_fraction"), "severe occlusion fraction")
    return diagnostics


def assess_evidence_quality(evidence: Any, config: dict | None = None) -> dict:
    """Return component metrics, diagnostics, and a proceed/warn/block decision.

    Structural Evidence errors block.  Missing or weak measurements produce
    warnings and remain available to downstream stages without imputation.
    """
    validation = validate_evidence(evidence)
    diagnostics = list(validation["findings"])
    thresholds, threshold_diagnostics = _thresholds(config)
    diagnostics.extend(threshold_diagnostics)
    normalized = validation["evidence"]

    if not validation["ok"]:
        return {
            "version": "1.0",
            "decision": "block",
            "evidence": normalized,
            "validation": validation,
            "thresholds": thresholds,
            "metrics": {"objects": [], "camera": {}, "aggregate": {}, "association": {}},
            "diagnostics": diagnostics,
        }

    frames = normalized.get("frames") or []
    frame_order = {frame["frame_index"]: index for index, frame in enumerate(frames)}
    duration = _number((normalized.get("meta") or {}).get("duration")) or 0.0
    object_metrics = [_object_metrics(obj, frames, frame_order, duration, thresholds) for obj in normalized.get("objects") or []]
    metrics = {
        "objects": object_metrics,
        "camera": _camera_metrics(normalized),
        "aggregate": _aggregate_metrics(object_metrics),
        "association": _association_metrics(normalized, object_metrics),
    }
    diagnostics.extend(_quality_diagnostics(metrics, thresholds))
    decision = "warn" if any(item["severity"] == "warning" for item in diagnostics) else "proceed"
    return {
        "version": "1.0",
        "decision": decision,
        "evidence": normalized,
        "validation": validation,
        "thresholds": thresholds,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
