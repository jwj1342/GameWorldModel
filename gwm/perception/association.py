"""Deterministic, model-independent association of multi-frame detections."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np

from .base import DetectionRecord, TrackedObject, Tracks


@dataclass
class AssociatedTrack:
    track_id: str
    class_label: str
    detections: list[DetectionRecord] = field(default_factory=list)
    identity_hypotheses: list[dict] = field(default_factory=list)

    @property
    def last(self) -> DetectionRecord:
        return self.detections[-1]


@dataclass
class AssociationResult:
    tracks: list[AssociatedTrack]
    diagnostics: list[dict]


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (value or "object")[:24]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    aa, bb = a.astype(bool), b.astype(bool)
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 0.0


def _appearance_similarity(a, b) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return None
    return float(np.clip((float(av @ bv) / denom + 1.0) / 2.0, 0.0, 1.0))


def _class_similarity(a: str, b: str) -> float:
    aa = set(re.findall(r"[a-z0-9]+", a.lower()))
    bb = set(re.findall(r"[a-z0-9]+", b.lower()))
    union = aa | bb
    return len(aa & bb) / len(union) if union else 0.0


def _center_distance_ratio(a, b) -> float:
    ac = np.asarray([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2], dtype=float)
    bc = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=float)
    scale = max(math.hypot(a[2] - a[0], a[3] - a[1]), math.hypot(b[2] - b[0], b[3] - b[1]), 1e-6)
    return float(np.linalg.norm(ac - bc) / scale)


def _config(config: dict) -> dict:
    if "perception" in config:
        config = (config.get("perception") or {}).get("association") or {}
    elif "association" in config:
        config = config["association"]
    required = {
        "max_frame_gap", "max_bbox_center_distance_ratio", "max_position_3d_distance", "min_class_similarity",
        "min_match_score", "min_ambiguous_score", "ambiguity_margin", "max_identity_candidates",
        "allow_class_mismatch", "time_decay_frames", "weights",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("missing association config: " + ", ".join(missing))
    required_weights = {"class", "bbox_iou", "mask_iou", "position_3d", "appearance", "time"}
    weights = config.get("weights")
    if not isinstance(weights, dict) or required_weights - set(weights):
        raise ValueError("association weights must define: " + ", ".join(sorted(required_weights)))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("association weights must be finite, non-negative, and not all zero")
    ratio_keys = {"min_match_score", "min_ambiguous_score", "ambiguity_margin", "min_class_similarity"}
    if any(not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or not 0 <= config[key] <= 1 for key in ratio_keys):
        raise ValueError("association score thresholds must be in [0, 1]")
    if config["min_ambiguous_score"] > config["min_match_score"]:
        raise ValueError("min_ambiguous_score must not exceed min_match_score")
    if not isinstance(config["max_frame_gap"], int) or isinstance(config["max_frame_gap"], bool) or config["max_frame_gap"] < 1:
        raise ValueError("max_frame_gap must be an integer >= 1")
    if not isinstance(config["max_identity_candidates"], int) or isinstance(config["max_identity_candidates"], bool) or config["max_identity_candidates"] < 2:
        raise ValueError("max_identity_candidates must be an integer >= 2")
    if any(not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or not math.isfinite(config[key]) or config[key] <= 0
           for key in ("max_bbox_center_distance_ratio", "max_position_3d_distance", "time_decay_frames")):
        raise ValueError("association distance and time thresholds must be finite and positive")
    if not isinstance(config["allow_class_mismatch"], bool):
        raise ValueError("allow_class_mismatch must be boolean")
    return config


def _pair_score(track: AssociatedTrack, detection: DetectionRecord, config: dict) -> tuple[float, dict] | None:
    previous = track.last
    gap = detection.frame_index - previous.frame_index
    if gap <= 0 or gap > int(config["max_frame_gap"]):
        return None
    class_similarity = _class_similarity(track.class_label, detection.class_label)
    if class_similarity < float(config["min_class_similarity"]) and not config["allow_class_mismatch"]:
        return None
    center_ratio = _center_distance_ratio(previous.bbox, detection.bbox)
    if center_ratio > float(config["max_bbox_center_distance_ratio"]):
        return None

    components: dict[str, float] = {
        "class": class_similarity,
        "bbox_iou": _iou(previous.bbox, detection.bbox),
        "time": math.exp(-gap / max(float(config["time_decay_frames"]), 1e-6)),
    }
    mask_iou = _mask_iou(previous.mask, detection.mask)
    if mask_iou is not None:
        components["mask_iou"] = mask_iou
    appearance = _appearance_similarity(previous.appearance, detection.appearance)
    if appearance is not None:
        components["appearance"] = appearance
    if previous.position_3d is not None and detection.position_3d is not None:
        distance = float(np.linalg.norm(np.asarray(previous.position_3d) - np.asarray(detection.position_3d)))
        maximum = float(config["max_position_3d_distance"])
        if distance > maximum:
            return None
        components["position_3d"] = max(0.0, 1.0 - distance / max(maximum, 1e-6))

    weights = config["weights"]
    active = [(name, value, float(weights.get(name, 0.0))) for name, value in components.items() if float(weights.get(name, 0.0)) > 0]
    total_weight = sum(weight for _, _, weight in active)
    if total_weight <= 0:
        return None
    score = sum(value * weight for _, value, weight in active) / total_weight
    return float(np.clip(score, 0.0, 1.0)), {"gap": gap, "center_distance_ratio": center_ratio, **components}


def _identity_hypothesis(candidates: list[tuple[float, AssociatedTrack]], limit: int) -> dict:
    selected = candidates[:max(2, limit)]
    total = sum(score for score, _ in selected)
    items = []
    for score, track in selected:
        confidence = score / total if total > 0 else 1.0 / len(selected)
        items.append({
            "value": f"continuation_of_{track.track_id}",
            "confidence": round(float(confidence), 6),
            "source": "multi_frame_association",
            "ref": track.track_id,
        })
    return {"kind": "identity", "candidates": items}


def associate_detections(detections: list[DetectionRecord], config: dict) -> AssociationResult:
    """Associate detections without mutating them or forcing ambiguous merges."""
    config = _config(config)
    grouped: dict[int, list[DetectionRecord]] = {}
    seen_ids: set[str] = set()
    for detection in detections:
        if detection.detection_id in seen_ids:
            raise ValueError(f"duplicate detection_id: {detection.detection_id}")
        seen_ids.add(detection.detection_id)
        if detection.frame_index < 0:
            raise ValueError("detection frame_index must be non-negative")
        if isinstance(detection.score, bool) or not isinstance(detection.score, (int, float)) or not math.isfinite(detection.score) or not 0 <= detection.score <= 1:
            raise ValueError(f"invalid score for detection {detection.detection_id}")
        if detection.bbox[2] < detection.bbox[0] or detection.bbox[3] < detection.bbox[1]:
            raise ValueError(f"invalid bbox for detection {detection.detection_id}")
        grouped.setdefault(int(detection.frame_index), []).append(detection)

    tracks: list[AssociatedTrack] = []
    counters: dict[str, int] = {}
    diagnostics: list[dict] = []

    def new_track(detection: DetectionRecord, hypotheses: list[dict] | None = None) -> AssociatedTrack:
        prefix = _slug(detection.class_label)
        counters[prefix] = counters.get(prefix, 0) + 1
        track = AssociatedTrack(f"{prefix}_{counters[prefix]}", detection.class_label, [detection], hypotheses or [])
        tracks.append(track)
        return track

    for frame_index in sorted(grouped):
        frame_detections = sorted(grouped[frame_index], key=lambda item: (item.detection_id, item.class_label, item.bbox))
        active = [track for track in tracks if 0 < frame_index - track.last.frame_index <= int(config["max_frame_gap"])]
        candidates_by_detection: dict[str, list[tuple[float, AssociatedTrack, dict]]] = {}
        ambiguous: set[str] = set()
        for detection in frame_detections:
            candidates = []
            for track in active:
                scored = _pair_score(track, detection, config)
                if scored is not None and scored[0] >= float(config["min_ambiguous_score"]):
                    candidates.append((scored[0], track, scored[1]))
            candidates.sort(key=lambda item: (-item[0], item[1].track_id))
            candidates_by_detection[detection.detection_id] = candidates
            if len(candidates) >= 2 and candidates[0][0] >= float(config["min_match_score"]) and candidates[0][0] - candidates[1][0] <= float(config["ambiguity_margin"]):
                ambiguous.add(detection.detection_id)

        assigned_detections: set[str] = set()
        assigned_tracks: set[str] = set()
        proposals = []
        for detection in frame_detections:
            if detection.detection_id in ambiguous:
                continue
            candidates = candidates_by_detection[detection.detection_id]
            if candidates and candidates[0][0] >= float(config["min_match_score"]):
                score, track, details = candidates[0]
                proposals.append((score, detection.detection_id, track.track_id, detection, track, details))
        proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
        for score, detection_id, track_id, detection, track, details in proposals:
            if detection_id in assigned_detections or track_id in assigned_tracks:
                continue
            track.detections.append(detection)
            assigned_detections.add(detection_id)
            assigned_tracks.add(track_id)
            diagnostics.append({"code": "associated", "frame_index": frame_index, "detection_id": detection_id,
                                "track_id": track_id, "score": round(score, 6), "components": details})

        for detection in frame_detections:
            if detection.detection_id in assigned_detections:
                continue
            candidates = candidates_by_detection[detection.detection_id]
            hypotheses = []
            if detection.detection_id in ambiguous:
                ranked = [(score, track) for score, track, _ in candidates]
                hypotheses = [_identity_hypothesis(ranked, int(config["max_identity_candidates"]))]
            track = new_track(detection, hypotheses)
            diagnostics.append({"code": "ambiguous_split" if hypotheses else "new_track", "frame_index": frame_index,
                                "detection_id": detection.detection_id, "track_id": track.track_id,
                                "candidate_tracks": [candidate.track_id for _, candidate, _ in candidates]})

    return AssociationResult(tracks=tracks, diagnostics=diagnostics)


def to_tracks(result: AssociationResult, frame_size: tuple[int, int], backend: str = "association") -> Tracks:
    """Convert associated records to the existing Tracks container."""
    objects = []
    for track in result.tracks:
        scores = [detection.score for detection in track.detections]
        obj = TrackedObject(
            id=track.track_id,
            phrase=track.class_label,
            score=float(sum(scores) / len(scores)) if scores else 0.0,
            identity_hypotheses=track.identity_hypotheses,
            source_detection_ids=[detection.detection_id for detection in track.detections],
        )
        for detection in track.detections:
            obj.boxes[detection.frame_index] = tuple(detection.bbox)
            if detection.mask is not None:
                obj.masks[detection.frame_index] = detection.mask.astype(bool)
        objects.append(obj)
    return Tracks(objects=objects, frame_size=frame_size, backend=backend, association_diagnostics=result.diagnostics)
