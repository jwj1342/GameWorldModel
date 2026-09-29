"""Auditable, reversible updates to source-bound Evidence v2 observations."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .contract import validate_evidence
from .quality import assess_evidence_quality


_OBSERVATION_FIELDS = frozenset({
    "bbox", "mask_ref", "visible_fraction", "visibility_state", "depth", "source", "confidence",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def evidence_digest(evidence: dict) -> str:
    """Stable SHA-256 over the JSON content, independent of formatting."""
    payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_hash(evidence: dict) -> str:
    value = ((evidence.get("meta") or {}).get("source_video") or {}).get("sha256")
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("Evidence must contain meta.source_video.sha256 before an auditable update")
    return value


def _validate_provenance(provenance: dict, source_hash: str) -> dict:
    if not isinstance(provenance, dict):
        raise ValueError("update provenance must be a mapping")
    required = ("source_kind", "source_ref", "video_sha256", "trigger_type")
    if any(not isinstance(provenance.get(key), str) or not provenance[key].strip() for key in required):
        raise ValueError("update provenance requires non-empty source and trigger fields")
    if provenance["video_sha256"] != source_hash:
        raise ValueError("update video SHA-256 does not match Evidence source")
    if provenance["trigger_type"] not in {"manual_review", "verification_request"}:
        raise ValueError("trigger_type must be manual_review or verification_request")
    if provenance["trigger_type"] == "verification_request" and not provenance.get("request_ref"):
        raise ValueError("verification_request update requires request_ref")
    if provenance["source_kind"] == "perception_provider":
        decision = provenance.get("identity_decision")
        if (not isinstance(decision, dict) or decision.get("status") != "confirmed"
                or not isinstance(decision.get("reviewer"), str) or not decision["reviewer"].strip()):
            raise ValueError("provider observation requires confirmed identity_decision with reviewer")
    return copy.deepcopy(provenance)


def _object_by_id(evidence: dict, object_id: str) -> dict:
    matches = [obj for obj in evidence["objects"] if obj["id"] == object_id]
    if len(matches) != 1:
        raise ValueError(f"update target object '{object_id}' is not unique in Evidence")
    return matches[0]


def _observation_index(obj: dict, frame_index: int) -> int | None:
    matches = [i for i, observation in enumerate(obj["observations"]) if observation["frame_index"] == frame_index]
    if len(matches) > 1:
        raise ValueError(f"track '{obj['track_id']}' has duplicate frame {frame_index}")
    return matches[0] if matches else None


def _quality_summary(report: dict) -> dict:
    return {
        "decision": report["decision"],
        "metrics": copy.deepcopy(report["metrics"]),
        "diagnostic_codes": sorted({item["code"] for item in report["diagnostics"]}),
        "diagnostic_count": len(report["diagnostics"]),
    }


def apply_evidence_update(base: dict, updates: list[dict], provenance: dict, config: dict | None = None) -> tuple[dict, dict]:
    """Apply reviewed observation upserts without mutating the base Evidence.

    An update identifies an existing object and frame, with either a ``fields``
    patch or a complete ``observation`` for a new frame observation. Identity
    changes are intentionally unsupported: provider candidates must be mapped
    to an existing track by a reviewer before entering this function.
    """
    validation_before = validate_evidence(base, adapt_v1=False)
    if not validation_before["ok"]:
        raise ValueError("base Evidence is structurally invalid")
    source_hash = _source_hash(base)
    source = _validate_provenance(provenance, source_hash)
    if not isinstance(updates, list) or not updates:
        raise ValueError("updates must be a non-empty list")
    updated = copy.deepcopy(base)
    known_frames = {frame["frame_index"] for frame in updated["frames"]}
    seen: set[tuple[str, int]] = set()
    changes = []
    for entry in updates:
        if not isinstance(entry, dict):
            raise ValueError("each update must be a mapping")
        object_id, frame_index = entry.get("object_id"), entry.get("frame_index")
        if not isinstance(object_id, str) or not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ValueError("update requires object_id and integer frame_index")
        key = (object_id, frame_index)
        if key in seen:
            raise ValueError(f"duplicate update for {object_id} frame {frame_index}")
        seen.add(key)
        if frame_index not in known_frames:
            raise ValueError(f"update references unregistered frame {frame_index}")
        obj = _object_by_id(updated, object_id)
        if entry.get("track_id") != obj["track_id"]:
            raise ValueError(f"update track does not match object {object_id}")
        index = _observation_index(obj, frame_index)
        if index is None:
            if set(entry) != {"object_id", "track_id", "frame_index", "observation"}:
                raise ValueError("new observation requires a complete observation, not a field patch")
            after = copy.deepcopy(entry["observation"])
            if not isinstance(after, dict) or after.get("frame_index") != frame_index or after.get("track_id") != obj["track_id"]:
                raise ValueError("new observation has inconsistent frame or track reference")
            before = None
            obj["observations"].append(after)
            obj["observations"].sort(key=lambda observation: (observation["t"], observation["frame_index"]))
            operation = "add"
            fields = sorted(after)
        else:
            if set(entry) != {"object_id", "track_id", "frame_index", "fields"}:
                raise ValueError("existing observation requires a fields patch")
            fields_patch = entry["fields"]
            if not isinstance(fields_patch, dict) or not fields_patch or set(fields_patch) - _OBSERVATION_FIELDS:
                raise ValueError("observation patch contains no supported measurement fields")
            before = copy.deepcopy(obj["observations"][index])
            after = copy.deepcopy(before)
            after.update(copy.deepcopy(fields_patch))
            if after == before:
                raise ValueError(f"update for {object_id} frame {frame_index} has no change")
            obj["observations"][index] = after
            operation = "update"
            fields = sorted(field for field in fields_patch if after[field] != before.get(field))
        changes.append({
            "operation": operation, "object_id": object_id, "track_id": obj["track_id"],
            "frame_index": frame_index, "changed_fields": fields,
            "before": before, "after": copy.deepcopy(after),
        })
    validation_after = validate_evidence(updated, adapt_v1=False)
    if not validation_after["ok"]:
        codes = sorted({item["code"] for item in validation_after["errors"]})
        raise ValueError("updated Evidence failed validation: " + ", ".join(codes))
    quality_before = assess_evidence_quality(base, config)
    quality_after = assess_evidence_quality(updated, config)
    before_coverage = quality_before["metrics"]["aggregate"]["measured_coverage"]
    after_coverage = quality_after["metrics"]["aggregate"]["measured_coverage"]
    event = {
        "version": "1.0", "kind": "evidence_observation_update",
        "provenance": source, "video_sha256": source_hash,
        "base_digest": evidence_digest(base), "updated_digest": evidence_digest(updated),
        "changes": changes,
        "counts": {
            "observations_added": sum(change["operation"] == "add" for change in changes),
            "observations_updated": sum(change["operation"] == "update" for change in changes),
            "field_changes": sum(len(change["changed_fields"]) for change in changes),
        },
        "validation": {
            "before": {"ok": validation_before["ok"], "errors": len(validation_before["errors"]), "warnings": len(validation_before["warnings"])},
            "after": {"ok": validation_after["ok"], "errors": len(validation_after["errors"]), "warnings": len(validation_after["warnings"])},
        },
        "quality": {
            "before": _quality_summary(quality_before), "after": _quality_summary(quality_after),
            "measured_coverage_delta": {key: round(after_coverage[key] - before_coverage[key], 6) for key in before_coverage},
        },
    }
    return updated, event


def rollback_evidence_update(updated: dict, event: dict) -> dict:
    """Undo only the recorded update, rejecting modified or unrelated input."""
    if event.get("kind") != "evidence_observation_update" or evidence_digest(updated) != event.get("updated_digest"):
        raise ValueError("updated Evidence does not match the recorded update event")
    restored = copy.deepcopy(updated)
    for change in reversed(event["changes"]):
        obj = _object_by_id(restored, change["object_id"])
        index = _observation_index(obj, change["frame_index"])
        if index is None or obj["observations"][index] != change["after"]:
            raise ValueError("update event and current observation disagree")
        if change["operation"] == "add":
            del obj["observations"][index]
        elif change["operation"] == "update":
            obj["observations"][index] = copy.deepcopy(change["before"])
        else:
            raise ValueError("unknown update operation")
    if evidence_digest(restored) != event.get("base_digest") or not validate_evidence(restored, adapt_v1=False)["ok"]:
        raise ValueError("rollback did not restore the validated base Evidence")
    return restored
