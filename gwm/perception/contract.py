"""Evidence v2 schema, validation, and loss-aware v1 migration."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import jsonschema


SCHEMA_PATH = Path(__file__).parent / "schema" / "evidence-v2.schema.json"
_SCHEMA: dict | None = None
UNKNOWN = "unknown"


def evidence_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


def _diag(path: str, code: str, message: str, suggestion: str, stage: str, severity: str) -> dict:
    return {
        "path": path,
        "code": code,
        "message": message,
        "suggestion": suggestion,
        "stage": stage,
        "severity": severity,
    }


def _migration_warning(path: str, code: str, message: str, suggestion: str = "") -> dict:
    return _diag(path, code, message, suggestion, "evidence_migration", "warning")


def _known_confidence(value: Any) -> float | str:
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else UNKNOWN


def _source(meta: dict) -> dict:
    return {
        "geometry": str(meta.get("geometry_backend") or UNKNOWN),
        "segmentation": str(meta.get("segmentation_backend") or UNKNOWN),
        "tracking": str(meta.get("tracks_backend") or UNKNOWN),
    }


def adapt_evidence_v1(evidence: dict) -> tuple[dict, list[dict]]:
    """Return a v2 copy plus migration warnings; never invent observations.

    Existing OBB entries are mapped one-for-one to observations.  Fields not
    carried by v1 are represented by the literal ``"unknown"``.
    """
    if evidence.get("schema_version") == "2.0":
        return copy.deepcopy(evidence), []

    out = copy.deepcopy(evidence)
    warnings = [_migration_warning(
        "/schema_version", "legacy_evidence_adapted",
        "Evidence without schema_version=2.0 was adapted in memory",
        "persist v2 evidence at the perception boundary for future runs",
    )]
    out["schema_version"] = "2.0"
    meta = out.setdefault("meta", {})
    if "scale" not in meta:
        meta["scale"] = UNKNOWN
        warnings.append(_migration_warning("/meta/scale", "legacy_unknown", "v1 evidence did not record scale"))

    frame_map: dict[int, dict] = {}
    geometry_frames = meta.get("geometry_frames") or []
    camera_poses = (out.get("camera") or {}).get("poses") or []
    for frame_index, pose in zip(geometry_frames, camera_poses):
        if isinstance(frame_index, int) and isinstance(pose.get("t"), (int, float)):
            frame_map[frame_index] = {
                "frame_index": frame_index,
                "t": float(pose["t"]),
                "source": {"kind": "video_frame", "ref": UNKNOWN},
            }

    for i, obj in enumerate(out.get("objects") or []):
        track_id = str(obj.get("track_id") or obj.get("id") or UNKNOWN)
        if "track_id" not in obj:
            warnings.append(_migration_warning(
                f"/objects/{i}/track_id", "legacy_track_mapping",
                "v1 object id was retained as its track id",
            ))
        obj["track_id"] = track_id
        obj.setdefault("class_guess", UNKNOWN)
        obj["confidence"] = _known_confidence(obj.get("confidence"))
        obj.setdefault("is_dynamic", UNKNOWN)
        obj.setdefault("motion_guess", {"type": UNKNOWN, "conf": UNKNOWN})
        obj.setdefault("contacts", [])
        obj.setdefault("hypotheses", [])

        motion_conf = _known_confidence((obj.get("motion_guess") or {}).get("conf"))
        obj.setdefault("attribute_confidence", {
            "class": obj["confidence"],
            "identity": UNKNOWN,
            "geometry": UNKNOWN,
            "motion": motion_conf,
        })

        if "observations" in obj:
            continue
        observations = []
        for j, obb in enumerate(obj.get("obb") or []):
            t = obb.get("t")
            if not isinstance(t, (int, float)) or not math.isfinite(t):
                warnings.append(_migration_warning(
                    f"/objects/{i}/obb/{j}", "legacy_observation_skipped",
                    "OBB without a finite timestamp cannot be represented as a v2 observation",
                ))
                continue
            frame_index = obb.get("frame") if isinstance(obb.get("frame"), int) else UNKNOWN
            if frame_index != UNKNOWN:
                frame_map.setdefault(frame_index, {
                    "frame_index": frame_index,
                    "t": float(t),
                    "source": {"kind": "video_frame", "ref": UNKNOWN},
                })
            observations.append({
                "frame_index": frame_index,
                "track_id": track_id,
                "t": float(t),
                "bbox": copy.deepcopy(obb.get("bbox", UNKNOWN)),
                "mask_ref": obb.get("mask_ref", UNKNOWN),
                "visible_fraction": _known_confidence(obb.get("visible_fraction")),
                "depth": copy.deepcopy(obb.get("depth", UNKNOWN)),
                "source": _source(meta),
                "confidence": {
                    "geometry": _known_confidence(obb.get("conf")),
                    "segmentation": obj["confidence"],
                    "tracking": UNKNOWN,
                },
            })
        obj["observations"] = observations

    out["frames"] = sorted(frame_map.values(), key=lambda frame: (frame["t"], frame["frame_index"]))
    out.setdefault("keyframes", [])
    return out, warnings


def _schema_findings(evidence: Any) -> list[dict]:
    validator = jsonschema.Draft202012Validator(evidence_schema())
    findings = []
    for error in sorted(validator.iter_errors(evidence), key=lambda item: list(item.absolute_path)):
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        findings.append(_diag(
            path, "schema", error.message[:300],
            "fix the field to match Evidence v2 schema",
            "evidence_schema", "error",
        ))
    return findings


def _semantic_findings(evidence: dict) -> list[dict]:
    findings: list[dict] = []

    if evidence["meta"].get("scale") == UNKNOWN:
        findings.append(_diag(
            "/meta/scale", "unknown_value", "scene scale is unknown",
            "preserve relative scale or record metric scale only when measured",
            "evidence_semantic", "warning",
        ))

    frames = evidence.get("frames") or []
    frame_by_id: dict[int, dict] = {}
    if not frames:
        findings.append(_diag(
            "/frames", "no_frame_registry", "frame registry is empty",
            "record source frames when they are available; do not synthesize frame ids",
            "evidence_semantic", "warning",
        ))
    for i, frame in enumerate(frames):
        frame_index = frame["frame_index"]
        if frame_index in frame_by_id:
            findings.append(_diag(
                f"/frames/{i}/frame_index", "duplicate_frame",
                f"frame_index {frame_index} is duplicated", "keep one registry entry per frame",
                "evidence_semantic", "error",
            ))
        frame_by_id[frame_index] = frame
        if frame["source"]["ref"] == UNKNOWN:
            findings.append(_diag(
                f"/frames/{i}/source/ref", "unknown_value",
                "frame source reference is unknown", "record a stable source frame reference when available",
                "evidence_semantic", "warning",
            ))
    frame_times = [frame["t"] for frame in frames]
    if any(b <= a for a, b in zip(frame_times, frame_times[1:])):
        findings.append(_diag(
            "/frames", "frame_time_order", "frame times must be strictly increasing",
            "sort frames by time and remove duplicate timestamps", "evidence_semantic", "error",
        ))

    for i, keyframe in enumerate(evidence.get("keyframes") or []):
        frame_index = keyframe.get("frame_index", keyframe.get("index")) if isinstance(keyframe, dict) else None
        if frame_index is None:
            findings.append(_diag(
                f"/keyframes/{i}", "unknown_value", "keyframe has no frame reference",
                "record frame_index when the keyframe can be traced", "evidence_semantic", "warning",
            ))
        elif frame_index not in frame_by_id:
            findings.append(_diag(
                f"/keyframes/{i}", "unknown_frame_ref", f"keyframe references unknown frame {frame_index}",
                "add the frame registry entry or correct the keyframe reference", "evidence_semantic", "error",
            ))
        elif isinstance(keyframe.get("t"), (int, float)) and abs(float(keyframe["t"]) - float(frame_by_id[frame_index]["t"])) > 1e-4:
            findings.append(_diag(
                f"/keyframes/{i}/t", "frame_time_mismatch", "keyframe time does not match its registered frame",
                "use the registered frame timestamp", "evidence_semantic", "error",
            ))

    camera_times = [pose["t"] for pose in (evidence.get("camera") or {}).get("poses") or []]
    if any(b < a for a, b in zip(camera_times, camera_times[1:])):
        findings.append(_diag(
            "/camera/poses", "camera_time_order", "camera pose times must be non-decreasing",
            "sort camera poses by time", "evidence_semantic", "error",
        ))

    objects = evidence.get("objects") or []
    object_ids: set[str] = set()
    track_ids: set[str] = set()
    for i, obj in enumerate(objects):
        oid, track_id = obj["id"], obj["track_id"]
        if oid in object_ids:
            findings.append(_diag(
                f"/objects/{i}/id", "duplicate_object", f"object id '{oid}' is duplicated",
                "use a stable unique object id", "evidence_semantic", "error",
            ))
        object_ids.add(oid)
        if track_id in track_ids:
            findings.append(_diag(
                f"/objects/{i}/track_id", "duplicate_track", f"track id '{track_id}' is assigned to multiple objects",
                "represent uncertain identity with hypotheses instead of duplicating a track",
                "evidence_semantic", "error",
            ))
        track_ids.add(track_id)

    known_refs = object_ids | track_ids | {"ground", "player"}
    for static_node in (evidence.get("static") or {}).get("nodes", []):
        if static_node.get("id"):
            known_refs.add(static_node["id"])

    for i, obj in enumerate(objects):
        base = f"/objects/{i}"
        if obj["class_guess"] == UNKNOWN:
            findings.append(_diag(
                f"{base}/class_guess", "unknown_value", "object class is unknown",
                "retain unknown until a traceable classifier or annotation provides a class",
                "evidence_semantic", "warning",
            ))
        motion_guess = obj.get("motion_guess") or {}
        if motion_guess.get("type") == UNKNOWN:
            findings.append(_diag(
                f"{base}/motion_guess/type", "unknown_value", "motion type is unknown",
                "use motion hypotheses when multiple supported interpretations exist",
                "evidence_semantic", "warning",
            ))
        if motion_guess.get("conf") == UNKNOWN:
            findings.append(_diag(
                f"{base}/motion_guess/conf", "unknown_value", "motion confidence is unknown",
                "do not replace unknown with an arbitrary default", "evidence_semantic", "warning",
            ))
        for field, value in obj["attribute_confidence"].items():
            if value == UNKNOWN:
                findings.append(_diag(
                    f"{base}/attribute_confidence/{field}", "unknown_value",
                    f"attribute confidence '{field}' is unknown", "record it only when supported by the producing backend",
                    "evidence_semantic", "warning",
                ))
        if obj["confidence"] == UNKNOWN:
            findings.append(_diag(
                f"{base}/confidence", "unknown_value", "legacy object confidence is unknown",
                "leave it unknown unless the producer supplies a calibrated score", "evidence_semantic", "warning",
            ))
        if obj["is_dynamic"] == UNKNOWN:
            findings.append(_diag(
                f"{base}/is_dynamic", "unknown_value", "dynamic/static state is unknown",
                "represent alternative motion interpretations in hypotheses", "evidence_semantic", "warning",
            ))

        observations = obj["observations"]
        times = [observation["t"] for observation in observations]
        if any(b < a for a, b in zip(times, times[1:])):
            findings.append(_diag(
                f"{base}/observations", "observation_time_order",
                "observation times must be non-decreasing", "sort observations by time",
                "evidence_semantic", "error",
            ))
        seen_frames: set[int] = set()
        for j, observation in enumerate(observations):
            opath = f"{base}/observations/{j}"
            frame_index = observation["frame_index"]
            if observation["track_id"] != obj["track_id"]:
                findings.append(_diag(
                    f"{opath}/track_id", "track_mismatch",
                    f"observation track '{observation['track_id']}' does not match object track '{obj['track_id']}'",
                    "move the observation to the matching object or correct its track id",
                    "evidence_semantic", "error",
                ))
            if frame_index == UNKNOWN:
                findings.append(_diag(
                    f"{opath}/frame_index", "unknown_value", "observation frame reference is unknown",
                    "record frame_index when the source observation can be traced", "evidence_semantic", "warning",
                ))
            else:
                if frame_index in seen_frames:
                    findings.append(_diag(
                        f"{opath}/frame_index", "duplicate_observation",
                        f"track has multiple observations for frame {frame_index}", "keep one observation per track and frame",
                        "evidence_semantic", "error",
                    ))
                seen_frames.add(frame_index)
                frame = frame_by_id.get(frame_index)
                if frame is None:
                    findings.append(_diag(
                        f"{opath}/frame_index", "unknown_frame_ref",
                        f"frame {frame_index} is not present in the frame registry", "add the frame registry entry or correct the reference",
                        "evidence_semantic", "error",
                    ))
                elif abs(float(frame["t"]) - float(observation["t"])) > 1e-4:
                    findings.append(_diag(
                        f"{opath}/t", "frame_time_mismatch",
                        f"observation time {observation['t']} does not match frame time {frame['t']}",
                        "use the registered frame timestamp", "evidence_semantic", "error",
                    ))

            for field in ("bbox", "mask_ref", "visible_fraction", "depth"):
                if observation[field] == UNKNOWN:
                    findings.append(_diag(
                        f"{opath}/{field}", "unknown_value", f"observation {field} is unknown",
                        "populate it only from a traceable measurement", "evidence_semantic", "warning",
                    ))
            for field, value in observation["source"].items():
                if value == UNKNOWN:
                    findings.append(_diag(
                        f"{opath}/source/{field}", "unknown_value", f"observation source '{field}' is unknown",
                        "record the producing backend when known", "evidence_semantic", "warning",
                    ))
            for field, value in observation["confidence"].items():
                if value == UNKNOWN:
                    findings.append(_diag(
                        f"{opath}/confidence/{field}", "unknown_value", f"observation confidence '{field}' is unknown",
                        "do not replace unknown with an arbitrary default", "evidence_semantic", "warning",
                    ))

            bbox = observation["bbox"]
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox["values"]
                width, height = bbox["image_size"]
                limit_x, limit_y = (1.0, 1.0) if bbox["space"] == "normalized" else (width, height)
                if x2 < x1 or y2 < y1 or x2 > limit_x or y2 > limit_y:
                    findings.append(_diag(
                        f"{opath}/bbox", "invalid_bbox", "bbox must be ordered and contained in its declared image space",
                        "correct xyxy ordering or image dimensions", "evidence_semantic", "error",
                    ))
            depth = observation["depth"]
            if isinstance(depth, dict) and not depth["min"] <= depth["median"] <= depth["max"]:
                findings.append(_diag(
                    f"{opath}/depth", "invalid_depth_order", "depth must satisfy min <= median <= max",
                    "recompute depth statistics from valid mask pixels", "evidence_semantic", "error",
                ))

        for j, contact in enumerate(obj.get("contacts") or []):
            ref = contact.get("with_id")
            if ref and ref not in known_refs:
                findings.append(_diag(
                    f"{base}/contacts/{j}/with_id", "unknown_object_ref",
                    f"contact target '{ref}' is not a known object, track, or static id",
                    "correct the reference or add the referenced entity", "evidence_semantic", "error",
                ))

        for j, hypothesis in enumerate(obj["hypotheses"]):
            values: set[str] = set()
            for k, candidate in enumerate(hypothesis["candidates"]):
                if candidate["value"] in values:
                    findings.append(_diag(
                        f"{base}/hypotheses/{j}/candidates/{k}/value", "duplicate_hypothesis",
                        f"hypothesis candidate '{candidate['value']}' is duplicated", "keep unique alternatives",
                        "evidence_semantic", "error",
                    ))
                values.add(candidate["value"])
                ref = candidate.get("ref")
                if candidate["confidence"] == UNKNOWN:
                    findings.append(_diag(
                        f"{base}/hypotheses/{j}/candidates/{k}/confidence", "unknown_value",
                        "hypothesis confidence is unknown", "retain unknown until the producer supplies a score",
                        "evidence_semantic", "warning",
                    ))
                if hypothesis["kind"] == "identity" and ref and ref not in known_refs:
                    findings.append(_diag(
                        f"{base}/hypotheses/{j}/candidates/{k}/ref", "unknown_object_ref",
                        f"identity candidate ref '{ref}' is unknown", "reference a known object or track id",
                        "evidence_semantic", "error",
                    ))
    return findings


def validate_evidence(evidence: Any, adapt_v1: bool = True) -> dict:
    """Validate Evidence and return the normalized v2 copy with diagnostics."""
    migration: list[dict] = []
    normalized = copy.deepcopy(evidence)
    adapted = False
    version = normalized.get("schema_version") if isinstance(normalized, dict) else None
    legacy_versions = {None, "1", "1.0", 1}
    if isinstance(normalized, dict) and version in legacy_versions and adapt_v1:
        normalized, migration = adapt_evidence_v1(normalized)
        adapted = True

    findings = list(migration)
    schema_findings = _schema_findings(normalized)
    findings.extend(schema_findings)
    if not schema_findings and isinstance(normalized, dict):
        findings.extend(_semantic_findings(normalized))
    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]
    return {
        "version": "1.0",
        "ok": not errors,
        "adapted": adapted,
        "evidence": normalized,
        "findings": findings,
        "errors": errors,
        "warnings": warnings,
    }


def format_evidence_errors(report: dict, limit: int = 30) -> str:
    lines = [
        f"- {item['path']}: [{item['code']}] {item['message']}"
        + (f" -> {item['suggestion']}" if item.get("suggestion") else "")
        for item in report["errors"][:limit]
    ]
    return "\n".join(lines) if lines else "(no errors)"
