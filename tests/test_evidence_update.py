import copy

import pytest

from gwm.perception.update import apply_evidence_update, evidence_digest, rollback_evidence_update


HASH = "a" * 64


def _observation(frame_index):
    return {
        "frame_index": frame_index, "track_id": "track_1", "t": float(frame_index),
        "bbox": "unknown", "mask_ref": "unknown", "visibility_state": "unknown",
        "visible_fraction": "unknown", "depth": "unknown",
        "source": {"geometry": "unknown", "segmentation": "unknown", "tracking": "manual_review"},
        "confidence": {"geometry": "unknown", "segmentation": "unknown", "tracking": "unknown"},
    }


def _evidence(observe_second=True):
    return {
        "schema_version": "2.0",
        "meta": {"clip": "synthetic", "duration": 2.0, "scale": "unknown", "source_video": {"sha256": HASH}},
        "frames": [
            {"frame_index": i, "t": float(i), "source": {"kind": "video_frame", "ref": f"sha256:{HASH}#frame={i}"}}
            for i in range(2)
        ],
        "camera": {"intrinsics": {}, "poses": []}, "static": {},
        "objects": [{
            "id": "object_1", "track_id": "track_1", "class_guess": "unknown", "confidence": "unknown",
            "is_dynamic": "unknown", "obb": [], "motion_guess": {"type": "unknown", "conf": "unknown"},
            "contacts": [], "observations": [_observation(0)] + ([_observation(1)] if observe_second else []),
            "attribute_confidence": {key: "unknown" for key in ("class", "identity", "geometry", "motion")},
            "hypotheses": [],
        }],
        "keyframes": [],
    }


def _provenance(**changes):
    result = {
        "source_kind": "manual_bbox_review", "source_ref": "synthetic_review_v1",
        "video_sha256": HASH, "trigger_type": "manual_review",
    }
    result.update(changes)
    return result


def _bbox():
    return {"format": "xyxy", "values": [10, 12, 30, 40], "space": "pixel", "image_size": [100, 80]}


def test_reviewed_update_is_auditable_and_exactly_reversible():
    base = _evidence(); original = copy.deepcopy(base)
    updates = [
        {"object_id": "object_1", "track_id": "track_1", "frame_index": i,
         "fields": {"bbox": _bbox(), "visibility_state": "visible"}}
        for i in range(2)
    ]
    updated, event = apply_evidence_update(base, updates, _provenance())
    assert base == original and updated is not base
    assert event["counts"] == {"observations_added": 0, "observations_updated": 2, "field_changes": 4}
    assert event["base_digest"] == evidence_digest(base)
    assert event["updated_digest"] == evidence_digest(updated)
    assert event["validation"]["before"]["ok"] and event["validation"]["after"]["ok"]
    assert event["quality"]["measured_coverage_delta"]["bbox"] == 1.0
    assert event["quality"]["measured_coverage_delta"]["visibility_state"] == 1.0
    assert event["quality"]["measured_coverage_delta"]["visible_fraction"] == 0.0
    assert rollback_evidence_update(updated, event) == original


def test_provider_observation_can_be_added_only_to_reviewed_existing_identity():
    base = _evidence(observe_second=False)
    new = _observation(1)
    new["bbox"] = _bbox()
    new["visibility_state"] = "visible"
    update = {"object_id": "object_1", "track_id": "track_1", "frame_index": 1, "observation": new}
    provenance = _provenance(source_kind="perception_provider", source_ref="provider_candidate_1",
                             trigger_type="verification_request", request_ref="request_1",
                             identity_decision={"status": "confirmed", "reviewer": "human_reviewer"})
    updated, event = apply_evidence_update(base, [update], provenance)
    assert event["counts"]["observations_added"] == 1
    assert len(updated["objects"][0]["observations"]) == 2
    assert rollback_evidence_update(updated, event) == base


def test_mismatched_source_or_unreviewed_provider_is_rejected():
    base = _evidence()
    update = {"object_id": "object_1", "track_id": "track_1", "frame_index": 0,
              "fields": {"bbox": _bbox()}}
    with pytest.raises(ValueError, match="SHA-256"):
        apply_evidence_update(base, [update], _provenance(video_sha256="b" * 64))
    with pytest.raises(ValueError, match="identity_decision"):
        apply_evidence_update(base, [update], _provenance(source_kind="perception_provider"))
    with pytest.raises(ValueError, match="request_ref"):
        apply_evidence_update(base, [update], _provenance(trigger_type="verification_request"))


def test_bad_reference_or_invalid_measurement_never_changes_base():
    base = _evidence(); original = copy.deepcopy(base)
    bad_track = {"object_id": "object_1", "track_id": "other", "frame_index": 0,
                 "fields": {"bbox": _bbox()}}
    with pytest.raises(ValueError, match="track"):
        apply_evidence_update(base, [bad_track], _provenance())
    bad_bbox = {"object_id": "object_1", "track_id": "track_1", "frame_index": 0,
                "fields": {"bbox": {"format": "xyxy", "values": [30, 20, 10, 40],
                                    "space": "pixel", "image_size": [100, 80]}}}
    with pytest.raises(ValueError, match="failed validation"):
        apply_evidence_update(base, [bad_bbox], _provenance())
    assert base == original


def test_rollback_refuses_tampered_evidence():
    base = _evidence()
    update = {"object_id": "object_1", "track_id": "track_1", "frame_index": 0,
              "fields": {"bbox": _bbox()}}
    updated, event = apply_evidence_update(base, [update], _provenance())
    updated["objects"][0]["observations"][0]["bbox"]["values"][0] = 11
    with pytest.raises(ValueError, match="does not match"):
        rollback_evidence_update(updated, event)
