"""Synthetic Program and Evidence records test local repair decisions only."""
import copy
import math

from gwm.compiler.repair import repair_candidate
from gwm.compiler.validate import validate_candidate


QUALITY = {
    "min_object_observation_coverage": 0.2, "min_bbox_coverage": 0.7,
    "min_mask_coverage": 0.7, "min_depth_coverage": 0.5,
    "min_visibility_coverage": 0.25, "max_track_gap_frames": 3,
    "min_object_temporal_span_ratio": 0.2, "min_camera_frame_coverage": 0.5,
    "min_camera_temporal_span_ratio": 0.7, "max_attribute_unknown_ratio": 0.5,
    "max_identity_ambiguity": 0.8, "min_motion_support": 0.4,
    "severe_occlusion_visible_fraction": 0.2, "max_severe_occlusion_fraction": 0.5,
}


def _config(mode="apply", **overrides):
    return {"candidate_repair": {"mode": mode, **overrides}, "evidence_quality": QUALITY}


def _program(y=0.5, support="ground"):
    obj = {"id": "object_1", "class": "box", "geom": {"kind": "asset", "query": "plain box", "extent": [1, 1, 1]},
           "pose": {"pos": [0.0, y, 0.0], "quat": [0, 0, 0, 1]}}
    if support is not None:
        obj["support"] = support
    return {"meta": {"clip": "synthetic", "units": "m", "up": "y", "duration": 9.0},
            "camera": {"keyframes": [{"t": 0.0, "pos": [0, 1, 5], "quat": [0, 0, 0, 1]}]},
            "static": [{"id": "ground", "class": "ground", "geom": {"kind": "primitive", "shape": "box", "extent": [10, 0.2, 10]},
                        "pose": {"pos": [0, -0.1, 0]}}], "objects": [obj]}


def _evidence():
    frames = [{"frame_index": i, "t": float(i), "source": {"kind": "video_frame", "ref": f"f{i}.png"}} for i in range(10)]
    observations = [{"frame_index": i, "track_id": "object_1", "t": float(i),
                     "bbox": {"format": "xyxy", "values": [10, 10, 30, 30], "space": "pixel", "image_size": [100, 80]},
                     "mask_ref": f"mask.npz#object_1__{i}", "visible_fraction": 0.9,
                     "depth": {"min": 1.0, "median": 1.5, "max": 2.0, "unit": "m"},
                     "source": {"geometry": "synthetic", "segmentation": "synthetic", "tracking": "synthetic"},
                     "confidence": {"geometry": 0.95, "segmentation": 0.95, "tracking": 0.95}} for i in range(10)]
    return {"schema_version": "2.0", "meta": {"clip": "synthetic", "duration": 9.0, "scale": "metric"},
            "frames": frames,
            "camera": {"intrinsics": {"width": 100, "height": 80, "fov_deg": 60},
                       "poses": [{"t": float(i), "pos": [0, 1, 5], "quat": [0, 0, 0, 1], "conf": 1.0} for i in range(10)]},
            "static": {"planes": []},
            "objects": [{"id": "object_1", "track_id": "object_1", "class_guess": "box", "confidence": 0.95,
                         "is_dynamic": False, "obb": [{"frame": i, "t": float(i), "center": [0, 0.5, 0],
                                                        "size": [1, 1, 1], "quat": [0, 0, 0, 1], "conf": 0.95} for i in range(10)],
                         "motion_guess": {"type": "static", "conf": 0.95}, "contacts": [{"with_id": "ground", "t_start": 0.0, "t_end": 9.0, "conf": 0.95}],
                         "observations": observations,
                         "attribute_confidence": {"class": 0.95, "identity": 0.95, "geometry": 0.95, "motion": 0.95},
                         "hypotheses": []}], "keyframes": []}


def _codes(items):
    return {item["code"] for item in items}


def test_near_unit_quaternion_repairs_without_mutating_input_and_is_idempotent():
    program, evidence = _program(), _evidence()
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 1.01]
    before = copy.deepcopy(program)
    fixed, report, unresolved = repair_candidate(program, evidence, validate_candidate(program), _config())
    assert program == before and fixed is not program
    assert fixed["objects"][0]["pose"]["quat"] == [0, 0, 0, 1.0]
    assert report["final_validation"]["ok"] and not unresolved
    edit = report["repairs"][0]
    assert {"code", "path", "before", "after", "rule", "evidence_refs", "confidence", "reason"} <= set(edit)
    assert edit["path"] == "/objects/0/pose/quat" and edit["code"] == "non_unit_quaternion"
    assert fixed["objects"][0]["id"] == before["objects"][0]["id"]
    again, second_report, _ = repair_candidate(fixed, evidence, None, _config())
    assert again == fixed and second_report["repairs"] == []


def test_small_penetration_and_gap_snap_only_with_unique_evidence_backed_support():
    for y, code in ((0.43, "support_penetration"), (0.57, "support_gap")):
        program, evidence = _program(y), _evidence()
        fixed, report, unresolved = repair_candidate(program, evidence, None, _config())
        assert fixed["objects"][0]["pose"]["pos"] == [0.0, 0.5, 0.0]
        assert validate_candidate(fixed)["ok"] and not unresolved
        assert report["repairs"][0]["code"] == code
        assert any(ref["role"] == "support_contact" for ref in report["repairs"][0]["evidence_refs"])
        assert program["objects"][0]["pose"]["pos"][1] == y
        repeated, repeated_report, _ = repair_candidate(fixed, evidence, None, _config())
        assert repeated == fixed and repeated_report["repairs"] == []


def test_quaternion_then_support_uses_bounded_revalidation():
    program = _program(0.43)
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 1.01]
    fixed, report, unresolved = repair_candidate(program, _evidence(), None, _config(max_passes=3))
    assert report["passes"] == 2 and len(report["repairs"]) == 2
    assert report["final_validation"]["ok"] and not unresolved
    assert fixed["objects"][0]["pose"]["pos"][1] == 0.5


def test_ambiguous_support_and_conflicting_contact_remain_unresolved():
    program, evidence = _program(0.43), _evidence()
    duplicate = copy.deepcopy(program["static"][0])
    duplicate["id"] = "second_floor"
    program["static"].append(duplicate)
    fixed, report, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and not report["repairs"]
    assert "support_penetration" in _codes(unresolved)
    program["static"].pop()
    evidence["objects"][0]["contacts"].append({"with_id": "other", "conf": 0.9})
    fixed, _, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and "support_penetration" in _codes(unresolved)


def test_dynamic_and_intentionally_floating_objects_are_not_snapped():
    program, evidence = _program(0.57), _evidence()
    program["objects"][0]["motion"] = {"type": "prismatic", "axis": [0, 1, 0], "rate": 0.2}
    fixed, _, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and "support_gap" in _codes(unresolved)
    floating = _program(0.57, support=None)
    fixed, report, unresolved = repair_candidate(floating, evidence, None, _config())
    assert fixed == floating and report["repairs"] == [] and not unresolved
    penetrating_without_support = _program(0.43, support=None)
    fixed, _, unresolved = repair_candidate(penetrating_without_support, evidence, None, _config())
    assert fixed == penetrating_without_support and "ground_penetration" in _codes(unresolved)


def test_large_error_missing_geometry_and_low_quality_evidence_are_refused():
    program, evidence = _program(0.20), _evidence()
    fixed, _, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and "support_penetration" in _codes(unresolved)
    missing_geom = _program(0.43)
    missing_geom["objects"][0]["geom"].pop("extent")
    fixed, _, unresolved = repair_candidate(missing_geom, evidence, None, _config())
    assert fixed == missing_geom and "schema" in _codes(unresolved)
    near_quat = _program()
    near_quat["objects"][0]["pose"]["quat"] = [0, 0, 0, 1.01]
    evidence["objects"][0]["attribute_confidence"]["geometry"] = "unknown"
    fixed, _, unresolved = repair_candidate(near_quat, evidence, None, _config())
    assert fixed == near_quat and "evidence_quality_not_proceed" in _codes(unresolved)


def test_relative_scale_and_conflicting_world_position_refuse_contact_shift():
    program, evidence = _program(0.43), _evidence()
    evidence["meta"]["scale"] = "relative"
    fixed, _, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and "support_penetration" in _codes(unresolved)
    evidence["meta"]["scale"] = "metric"
    for obb in evidence["objects"][0]["obb"]:
        obb["center"][0] = 0.4
    fixed, _, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and "support_penetration" in _codes(unresolved)


def test_nonfinite_position_zero_and_far_quaternions_are_not_guessed():
    program = _program()
    program["objects"][0]["pose"]["pos"][1] = math.inf
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 0]
    fixed, report, unresolved = repair_candidate(program, _evidence(), None, _config())
    assert math.isinf(fixed["objects"][0]["pose"]["pos"][1])
    assert fixed["objects"][0]["pose"]["quat"] == [0, 0, 0, 0]
    assert not report["repairs"] and {"zero_quaternion", "non_finite_transform"} <= _codes(unresolved)
    far = _program()
    far["objects"][0]["pose"]["quat"] = [0, 0, 0, 1.5]
    fixed, _, unresolved = repair_candidate(far, _evidence(), None, _config())
    assert fixed == far and "non_unit_quaternion" in _codes(unresolved)


def test_off_and_report_modes_preserve_candidate_and_stale_diagnostics_refuse():
    program, evidence = _program(), _evidence()
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 1.01]
    off, off_report, _ = repair_candidate(program, evidence, None, _config("off"))
    assert off == program and off is not program and off_report["repairs"] == []
    dry, dry_report, unresolved = repair_candidate(program, evidence, None, _config("report"))
    assert dry == program and dry_report["repairs"] == []
    assert len(dry_report["proposed_repairs"]) == 1 and dry_report["simulated_validation"]["ok"]
    assert "non_unit_quaternion" in _codes(unresolved)
    stale, stale_report, unresolved = repair_candidate(program, evidence, [], _config())
    assert stale == program and not stale_report["diagnostics_verified"]
    assert "stale_diagnostics" in _codes(unresolved)


def test_semantic_conflict_and_wrong_clip_block_even_independent_quaternion_edit():
    program, evidence = _program(), _evidence()
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 1.01]
    evidence["meta"]["clip"] = "different_clip"
    fixed, _, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and "evidence_clip_mismatch" in _codes(unresolved)
    evidence["meta"]["clip"] = "synthetic"
    program["objects"][0]["id"] = "ground"
    fixed, _, unresolved = repair_candidate(program, evidence, None, _config())
    assert fixed == program and "duplicate_id" in _codes(unresolved)


def test_iteration_limit_leaves_contact_unresolved_after_quaternion_pass():
    program = _program(0.43)
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 1.01]
    fixed, report, unresolved = repair_candidate(program, _evidence(), None, _config(max_passes=1))
    assert report["passes"] == 1
    assert fixed["objects"][0]["pose"]["quat"] == [0, 0, 0, 1.0]
    assert fixed["objects"][0]["pose"]["pos"][1] == 0.43
    assert "support_penetration" in _codes(unresolved)
