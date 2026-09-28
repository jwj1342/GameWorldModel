"""Planning tests use synthetic contracts and never run browser or perception models."""
from copy import deepcopy
import hashlib

import pytest

from gwm.feedback.active_perception import plan_active_perception_requests


def _observation(index, track_id):
    return {"frame_index": index, "track_id": track_id, "t": float(index),
            "bbox": {"format": "xyxy", "values": [1, 1, 4, 4], "space": "pixel", "image_size": [16, 12]},
            "mask_ref": f"masks.npz#{track_id}_{index}", "visible_fraction": 0.8,
            "depth": {"min": 1.0, "median": 1.5, "max": 2.0, "unit": "relative"},
            "source": {"geometry": "synthetic", "segmentation": "synthetic", "tracking": "synthetic"},
            "confidence": {"geometry": 0.8, "segmentation": 0.8, "tracking": 0.8}}


def _evidence(names=("object_a",), observed=(0, 1, 2, 3, 4)):
    return {"schema_version": "2.0", "meta": {"clip": "synthetic", "duration": 4.0, "scale": "relative"},
            "frames": [{"frame_index": i, "t": float(i), "source": {"kind": "video_frame", "ref": f"f_{i}.png"}}
                       for i in range(5)],
            "camera": {"intrinsics": {}, "poses": []}, "static": {}, "keyframes": [],
            "objects": [{"id": name, "track_id": f"track_{i}", "class_guess": "object", "confidence": 0.8,
                         "is_dynamic": True, "obb": [], "motion_guess": {"type": "prismatic", "conf": 0.8},
                         "contacts": [], "observations": [_observation(k, f"track_{i}") for k in observed],
                         "attribute_confidence": {"class": 0.8, "identity": 0.8, "geometry": 0.8, "motion": 0.8},
                         "hypotheses": []} for i, name in enumerate(names)]}


def _program(names=("object_a",)):
    return {"objects": [{"id": name, "motion": {"type": "prismatic"}} for name in names]}


def _verification(*diagnostics, times=(0.0, 1.0, 2.0, 3.0, 4.0)):
    return {"version": "1.0", "diagnostics": list(diagnostics),
            "metrics": {"frames": [{"index": i, "t": t} for i, t in enumerate(times)], "objects": {}}}


def _signal(code, path="/objects/0", severity="warning"):
    return {"stage": "visibility", "severity": severity, "code": code, "path": path,
            "message": f"synthetic {code}", "hint": "inspect"}


def _plan(verification, evidence=None, program=None, config=None):
    return plan_active_perception_requests(verification, evidence or _evidence(), program or _program(), config)


def test_no_findings_produces_no_requests_and_preserves_inputs():
    verification, evidence, program = _verification(), _evidence(), _program()
    before = deepcopy((verification, evidence, program))
    report = _plan(verification, evidence, program)
    assert report["status"] == "ready" and report["mode"] == "review_only"
    assert report["requests"] == []
    assert (verification, evidence, program) == before


def test_planned_request_ids_are_stable_for_human_selection():
    verification = _verification(_signal("object_never_visible"))
    first = _plan(verification)["requests"][0]["request_id"]
    second = _plan(verification)["requests"][0]["request_id"]
    assert first == second and first.startswith("apr-")


def test_planner_binds_evidence_to_video_hash_and_rejects_mismatch(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"first source")
    evidence = _evidence()
    evidence["meta"]["source_video_sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    verification = _verification(_signal("object_never_visible"))
    matched = plan_active_perception_requests(verification, evidence, _program(), source_video=str(video))
    assert matched["source_binding"]["status"] == "verified"
    assert matched["source_binding"]["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    video.write_bytes(b"different source")
    rejected = plan_active_perception_requests(verification, evidence, _program(), source_video=str(video))
    assert rejected["status"] == "invalid_input" and rejected["requests"] == []
    assert rejected["diagnostics"][0]["code"] == "evidence_video_hash_mismatch"


def test_never_visible_and_low_coverage_merge_with_uncertainty_and_priority():
    verification = _verification(_signal("object_never_visible"))
    report = _plan(verification, _evidence(observed=(1,)))
    assert len(report["requests"]) == 1
    request = report["requests"][0]
    assert request["target"] == {"program_object_id": "object_a", "evidence_object_id": "object_a",
                                 "track_id": "track_0", "class_guess": "object"}
    assert {signal["code"] for signal in request["verification_signals"]} == {
        "object_never_visible", "insufficient_object_observations"}
    assert request["priority"] == "high"
    assert "occlusion_vs_out_of_view_unknown" in request["uncertainties"]
    assert {"bbox", "mask", "visible_fraction", "camera_pose", "depth"} <= set(request["needed_evidence"])
    assert request["suggested_video_range"]["start_s"] >= 0
    assert request["suggested_video_range"]["end_s"] <= 4
    assert request["suggested_keyframes"] and all("source_ref" in item for item in request["suggested_keyframes"])


def test_motion_and_identity_ambiguity_merge_without_choosing_candidate():
    evidence = _evidence()
    evidence["objects"][0]["hypotheses"] = [{"kind": "identity", "candidates": [
        {"value": "track_0", "confidence": 0.5, "source": "association"},
        {"value": "track_other", "confidence": 0.5, "source": "association"}]}]
    report = _plan(_verification(_signal("dynamic_object_not_moving", "/objects/0/motion", "error")), evidence)
    assert len(report["requests"]) == 1
    request = report["requests"][0]
    assert request["priority"] == "high"
    assert {signal["code"] for signal in request["verification_signals"]} == {
        "dynamic_object_not_moving", "identity_candidates_unresolved"}
    assert "track_identity" in request["needed_evidence"] and "world_position" in request["needed_evidence"]
    assert set(request["uncertainties"]) == {"source_motion_vs_runtime_unknown", "identity_unresolved"}


@pytest.mark.parametrize("fault", ["browser_pageerror", "missing_pass_file", "undecodable_pass", "state_id_mismatch"])
def test_execution_faults_take_precedence_over_reperception(fault):
    evidence = _evidence(observed=(0,))
    evidence["objects"][0]["hypotheses"] = [{"kind": "identity", "candidates": [
        {"value": "track_0", "confidence": "unknown", "source": "association"},
        {"value": "track_other", "confidence": "unknown", "source": "association"}]}]
    report = _plan(_verification(_signal(fault, "/frames/0/files/rgb", "error"),
                                 _signal("object_never_visible")), evidence)
    assert report["status"] == "execution_blocked"
    assert report["requests"] == []
    assert any(item["code"] == fault for item in report["execution_issues"])


def test_temporary_absence_does_not_assert_occlusion():
    verification = _verification(_signal("object_temporarily_not_visible"))
    verification["metrics"]["objects"] = {"object_a": {"id_absent_frames": [1, 2]}}
    report = _plan(verification)
    request = report["requests"][0]
    assert "occlusion_vs_out_of_view_unknown" in request["uncertainties"]
    assert request["suggested_video_range"] == {"start_s": 0.5, "end_s": 2.5}
    assert request["priority"] == "medium"


def test_count_range_and_keyframe_limits_are_configurable():
    names = ("object_a", "object_b")
    verification = _verification(_signal("object_never_visible", "/objects/0"),
                                 _signal("object_never_visible", "/objects/1"))
    report = _plan(verification, _evidence(names), _program(names), {"active_perception": {
        "max_requests": 1, "max_time_range_s": 1.0, "time_padding_s": 0.0,
        "max_keyframes_per_request": 1}})
    assert len(report["requests"]) == 1 and report["truncated"] == 1
    request = report["requests"][0]
    span = request["suggested_video_range"]["end_s"] - request["suggested_video_range"]["start_s"]
    assert span <= 1.0 and len(request["suggested_keyframes"]) == 1


def test_request_limit_keeps_high_priority_before_medium():
    names = ("object_a", "object_b")
    verification = _verification(_signal("object_temporarily_not_visible", "/objects/0"),
                                 _signal("object_never_visible", "/objects/1"))
    report = _plan(verification, _evidence(names), _program(names), {"max_requests": 1})
    assert report["requests"][0]["target"]["program_object_id"] == "object_b"
    assert report["truncated"] == 1


def test_merge_gap_controls_duplicate_requests():
    verification = _verification(_signal("object_never_visible"),
                                 _signal("object_temporarily_not_visible"), times=(0, 0.5, 2, 3, 4))
    verification["metrics"]["objects"] = {"object_a": {"id_absent_frames": [4]}}
    strict = _plan(verification, config={"time_padding_s": 0.0, "max_time_range_s": 0.5, "merge_gap_s": 0.0})
    loose = _plan(verification, config={"time_padding_s": 0.0, "max_time_range_s": 4.0, "merge_gap_s": 4.0})
    assert len(strict["requests"]) == 2
    assert len(loose["requests"]) == 1


def test_unmatched_evidence_identity_is_not_guessed():
    report = _plan(_verification(_signal("object_never_visible")), _evidence(names=("different",)), _program())
    assert report["requests"][0]["target"]["evidence_object_id"] is None
    assert any(item["code"] == "unmatched_evidence_object" for item in report["diagnostics"])


def test_bad_input_is_reported_not_transformed_into_a_request():
    report = _plan({"diagnostics": "invalid"})
    assert report["status"] == "execution_blocked" and report["requests"] == []
    evidence = _evidence()
    evidence["objects"][0]["observations"][0]["track_id"] = "incorrect"
    report = _plan(_verification(), evidence)
    assert report["status"] == "invalid_input" and report["requests"] == []
