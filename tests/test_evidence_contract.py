import copy

import pytest

from gwm.perception.contract import adapt_evidence_v1, validate_evidence
from gwm.synthesis.direct import evidence_to_program


def _observation(frame_index=0, track_id="track_1", t=0.0):
    return {
        "frame_index": frame_index,
        "track_id": track_id,
        "t": t,
        "bbox": {"format": "xyxy", "values": [10, 20, 30, 40], "space": "pixel", "image_size": [100, 80]},
        "mask_ref": f"masks.npz#track_1__{frame_index}",
        "visible_fraction": 0.8,
        "depth": {"min": 1.0, "median": 1.5, "max": 2.0, "unit": "relative"},
        "source": {"geometry": "test_geometry", "segmentation": "test_segmentation", "tracking": "test_tracker"},
        "confidence": {"geometry": 0.9, "segmentation": 0.8, "tracking": 0.7},
    }


def _object(object_id="object_1", track_id="track_1"):
    return {
        "id": object_id,
        "track_id": track_id,
        "class_guess": "box",
        "confidence": 0.8,
        "is_dynamic": False,
        "obb": [],
        "motion_guess": {"type": "static", "conf": 0.9},
        "contacts": [],
        "observations": [_observation(track_id=track_id)],
        "attribute_confidence": {"class": 0.8, "identity": 0.7, "geometry": 0.9, "motion": 0.9},
        "hypotheses": [],
    }


def _v2():
    return {
        "schema_version": "2.0",
        "meta": {"clip": "synthetic", "duration": 1.0, "scale": "relative"},
        "frames": [
            {"frame_index": 0, "t": 0.0, "source": {"kind": "video_frame", "ref": "frame_000.png"}},
            {"frame_index": 1, "t": 1.0, "source": {"kind": "video_frame", "ref": "frame_001.png"}},
        ],
        "camera": {
            "intrinsics": {"width": 100, "height": 80, "fov_deg": 60},
            "poses": [
                {"t": 0.0, "pos": [0, 1, 5], "quat": [0, 0, 0, 1], "conf": 1.0},
                {"t": 1.0, "pos": [0, 1, 4], "quat": [0, 0, 0, 1], "conf": 1.0},
            ],
        },
        "static": {"planes": []},
        "objects": [_object()],
        "keyframes": [],
    }


def _v1():
    return {
        "meta": {"clip": "legacy", "duration": 1.0, "scale": "relative", "geometry_frames": [0, 1],
                 "geometry_backend": "legacy_geometry", "segmentation_backend": "legacy_segmentation", "tracks_backend": "legacy_tracker"},
        "camera": {"intrinsics": {"width": 100, "height": 80, "fov_deg": 60},
                   "poses": [{"t": 0.0, "pos": [0, 1, 5], "quat": [0, 0, 0, 1]},
                             {"t": 1.0, "pos": [0, 1, 4], "quat": [0, 0, 0, 1]}]},
        "static": {"planes": []},
        "objects": [{"id": "legacy_1", "class_guess": "box", "confidence": 0.75, "is_dynamic": False,
                     "obb": [{"frame": 0, "t": 0.0, "center": [0, 0.5, 0], "size": [1, 1, 1], "quat": [0, 0, 0, 1], "conf": 0.8},
                             {"frame": 1, "t": 1.0, "center": [0, 0.5, 0], "size": [1, 1, 1], "quat": [0, 0, 0, 1], "conf": 0.8}],
                     "motion_guess": {"type": "static", "conf": 0.6}, "contacts": []}],
        "keyframes": [],
    }


def test_valid_evidence_v2():
    report = validate_evidence(_v2())
    assert report["ok"] and not report["adapted"]
    assert report["errors"] == [] and report["warnings"] == []


def test_unknown_information_is_explicit_and_warned():
    evidence = _v2()
    observation = evidence["objects"][0]["observations"][0]
    for field in ("bbox", "mask_ref", "visible_fraction", "depth"):
        observation[field] = "unknown"
    evidence["objects"][0]["attribute_confidence"]["identity"] = "unknown"
    report = validate_evidence(evidence)
    assert report["ok"]
    warning_paths = {warning["path"] for warning in report["warnings"]}
    assert "/objects/0/observations/0/bbox" in warning_paths
    assert "/objects/0/attribute_confidence/identity" in warning_paths


def test_bad_frame_reference_is_rejected():
    evidence = _v2()
    evidence["objects"][0]["observations"][0]["frame_index"] = 99
    report = validate_evidence(evidence)
    assert not report["ok"] and any(error["code"] == "unknown_frame_ref" for error in report["errors"])


def test_invalid_confidence_is_rejected_by_schema():
    evidence = _v2()
    evidence["objects"][0]["attribute_confidence"]["class"] = 1.2
    report = validate_evidence(evidence)
    assert not report["ok"] and any(error["code"] == "schema" for error in report["errors"])


def test_unknown_schema_version_is_not_silently_migrated():
    evidence = _v2(); evidence["schema_version"] = "3.0"
    report = validate_evidence(evidence)
    assert not report["ok"] and not report["adapted"]
    assert any(error["code"] == "schema" for error in report["errors"])


def test_identity_ambiguity_has_explicit_candidates():
    evidence = _v2()
    second = _object("object_2", "track_2")
    second["observations"][0]["mask_ref"] = "masks.npz#track_2__0"
    evidence["objects"].append(second)
    evidence["objects"][0]["hypotheses"] = [{
        "kind": "identity",
        "candidates": [
            {"value": "same as track_1", "confidence": 0.55, "source": "association", "ref": "track_1"},
            {"value": "same as track_2", "confidence": 0.45, "source": "association", "ref": "track_2"},
        ],
    }]
    report = validate_evidence(evidence)
    assert report["ok"], report["errors"]


def test_order_and_track_consistency_are_checked():
    evidence = _v2()
    evidence["objects"][0]["observations"] = [
        _observation(frame_index=1, t=1.0),
        _observation(frame_index=0, track_id="other_track", t=0.0),
    ]
    report = validate_evidence(evidence)
    codes = {error["code"] for error in report["errors"]}
    assert {"observation_time_order", "track_mismatch"} <= codes


def test_v1_migration_is_loss_aware_and_does_not_mutate_input():
    legacy = _v1(); original = copy.deepcopy(legacy)
    migrated, migration_warnings = adapt_evidence_v1(legacy)
    report = validate_evidence(legacy)
    assert legacy == original
    assert migrated["schema_version"] == "2.0" and report["ok"] and report["adapted"]
    observation = report["evidence"]["objects"][0]["observations"][0]
    assert observation["bbox"] == "unknown" and observation["mask_ref"] == "unknown"
    assert observation["visible_fraction"] == "unknown" and observation["depth"] == "unknown"
    assert report["evidence"]["objects"][0]["attribute_confidence"]["identity"] == "unknown"
    assert migration_warnings and any(warning["code"] == "unknown_value" for warning in report["warnings"])


def test_program_generation_gate_rejects_invalid_evidence():
    evidence = _v2()
    evidence["objects"][0]["observations"][0]["frame_index"] = 99
    with pytest.raises(ValueError, match="invalid evidence"):
        evidence_to_program(evidence)
