import copy

from gwm.perception.quality import assess_evidence_quality


QUALITY = {
    "min_object_observation_coverage": 0.20,
    "min_bbox_coverage": 0.70,
    "min_mask_coverage": 0.70,
    "min_depth_coverage": 0.50,
    "min_visibility_coverage": 0.25,
    "max_track_gap_frames": 3,
    "min_object_temporal_span_ratio": 0.20,
    "min_camera_frame_coverage": 0.50,
    "min_camera_temporal_span_ratio": 0.70,
    "max_attribute_unknown_ratio": 0.50,
    "max_identity_ambiguity": 0.80,
    "min_motion_support": 0.40,
    "severe_occlusion_visible_fraction": 0.20,
    "max_severe_occlusion_fraction": 0.50,
}


def _observation(frame_index, visible=0.9):
    return {
        "frame_index": frame_index,
        "track_id": "track_1",
        "t": float(frame_index),
        "bbox": {"format": "xyxy", "values": [10, 20, 30, 40], "space": "pixel", "image_size": [100, 80]},
        "mask_ref": f"masks.npz#track_1__{frame_index}",
        "visible_fraction": visible,
        "depth": {"min": 1.0, "median": 1.5, "max": 2.0, "unit": "relative"},
        "source": {"geometry": "test_geometry", "segmentation": "test_segmentation", "tracking": "test_tracker"},
        "confidence": {"geometry": 0.9, "segmentation": 0.8, "tracking": 0.7},
    }


def _evidence():
    frames = [{"frame_index": index, "t": float(index), "source": {"kind": "video_frame", "ref": f"f_{index:03d}.png"}} for index in range(10)]
    return {
        "schema_version": "2.0",
        "meta": {"clip": "quality_fixture", "duration": 9.0, "scale": "relative"},
        "frames": frames,
        "camera": {"intrinsics": {"width": 100, "height": 80, "fov_deg": 60},
                   "poses": [{"t": float(index), "pos": [0, 1, 5 - index * 0.1], "quat": [0, 0, 0, 1], "conf": 1.0} for index in range(10)]},
        "static": {"planes": []},
        "objects": [{
            "id": "object_1", "track_id": "track_1", "class_guess": "box", "confidence": 0.8, "is_dynamic": True,
            "obb": [], "motion_guess": {"type": "trajectory", "conf": 0.9}, "contacts": [],
            "observations": [_observation(index) for index in range(10)],
            "attribute_confidence": {"class": 0.8, "identity": 0.9, "geometry": 0.9, "motion": 0.9},
            "hypotheses": [],
        }],
        "keyframes": [],
    }


def _codes(report):
    return {diagnostic["code"] for diagnostic in report["diagnostics"]}


def test_high_quality_evidence_proceeds_without_total_score():
    report = assess_evidence_quality(_evidence(), {"evidence_quality": QUALITY})
    assert report["decision"] == "proceed"
    assert "score" not in report and "score" not in report["metrics"]
    obj = report["metrics"]["objects"][0]
    assert obj["observation_coverage"] == 1.0
    assert obj["measured_coverage"] == {"bbox": 1.0, "mask_ref": 1.0, "depth": 1.0, "visible_fraction": 1.0}
    assert report["metrics"]["camera"]["frame_coverage"] == 1.0


def test_low_observation_coverage_warns_but_does_not_block():
    evidence = _evidence(); evidence["objects"][0]["observations"] = [_observation(0)]
    report = assess_evidence_quality(evidence, QUALITY)
    assert report["decision"] == "warn"
    assert {"low_object_coverage", "short_track_span"} <= _codes(report)


def test_severe_occlusion_is_reported_separately():
    evidence = _evidence()
    for observation in evidence["objects"][0]["observations"]:
        observation["visible_fraction"] = 0.1
    report = assess_evidence_quality(evidence, QUALITY)
    assert report["decision"] == "warn" and "severe_occlusion" in _codes(report)
    assert report["metrics"]["objects"][0]["visibility"]["severe_occlusion_fraction"] == 1.0


def test_track_break_is_measured_without_assuming_continuity():
    evidence = _evidence(); evidence["objects"][0]["observations"] = [_observation(i) for i in (0, 1, 8, 9)]
    report = assess_evidence_quality(evidence, QUALITY)
    track = report["metrics"]["objects"][0]["track"]
    assert report["decision"] == "warn" and "track_break" in _codes(report)
    assert track["gap_count"] == 1 and track["max_gap_frames"] == 6 and track["temporal_span_ratio"] == 1.0


def test_identity_ambiguity_uses_normalized_entropy():
    evidence = _evidence()
    evidence["meta"]["association_diagnostics"] = [
        {"code": "new_track"}, {"code": "associated"}, {"code": "ambiguous_split"},
    ]
    evidence["objects"][0]["hypotheses"] = [{
        "kind": "identity",
        "candidates": [
            {"value": "track_1", "confidence": 0.5, "source": "association"},
            {"value": "track_2", "confidence": 0.5, "source": "association"},
        ],
    }]
    report = assess_evidence_quality(evidence, QUALITY)
    ambiguity = report["metrics"]["objects"][0]["identity_ambiguity"]
    assert report["decision"] == "warn" and "high_identity_ambiguity" in _codes(report)
    assert ambiguity["normalized_entropy"] == 1.0
    assert report["metrics"]["association"]["tracks_with_identity_hypotheses"] == 1
    assert report["metrics"]["association"]["diagnostic_counts"]["ambiguous_split"] == 1


def test_weak_motion_support_warns():
    evidence = _evidence()
    evidence["objects"][0]["attribute_confidence"]["motion"] = 0.2
    evidence["objects"][0]["motion_guess"]["conf"] = 0.2
    report = assess_evidence_quality(evidence, QUALITY)
    assert report["decision"] == "warn" and "low_motion_support" in _codes(report)
    assert report["metrics"]["objects"][0]["motion_support"]["primary_confidence"] == 0.2


def test_motion_fit_evidence_reaches_quality_report():
    evidence = _evidence()
    obj = evidence["objects"][0]
    obj["motion_guess"].update({"support_frames": 10, "temporal_span_s": 9.0, "residual": 0.012, "residual_unit": "m",
                                "candidate_margin": 0.42})
    obj["hypotheses"].append({"kind": "motion", "candidates": [
        {"value": "static", "confidence": 0.81, "source": "multi_frame_motion_estimator"},
        {"value": "prismatic", "confidence": 0.39, "source": "multi_frame_motion_estimator"}]})
    report = assess_evidence_quality(evidence, QUALITY)
    support = report["metrics"]["objects"][0]["motion_support"]
    assert support["support_frames"] == 10
    assert support["temporal_span_s"] == 9.0
    assert support["residual"] == 0.012
    assert support["residual_unit"] == "m"
    assert support["candidate_margin"] == 0.42


def test_configured_threshold_changes_decision():
    evidence = _evidence(); evidence["objects"][0]["observations"] = [_observation(i) for i in range(4)]
    permissive = copy.deepcopy(QUALITY); permissive["min_object_observation_coverage"] = 0.3
    strict = copy.deepcopy(QUALITY); strict["min_object_observation_coverage"] = 0.5
    assert assess_evidence_quality(evidence, permissive)["decision"] == "proceed"
    strict_report = assess_evidence_quality(evidence, strict)
    assert strict_report["decision"] == "warn" and "low_object_coverage" in _codes(strict_report)


def test_structural_reference_error_blocks():
    evidence = _evidence(); evidence["objects"][0]["observations"][0]["frame_index"] = 99
    report = assess_evidence_quality(evidence, QUALITY)
    assert report["decision"] == "block" and "unknown_frame_ref" in _codes(report)


def test_legacy_evidence_is_migrated_and_warned_not_blocked():
    evidence = _evidence()
    legacy = {key: copy.deepcopy(value) for key, value in evidence.items() if key not in ("schema_version", "frames")}
    for obj in legacy["objects"]:
        obj.pop("track_id"); obj.pop("observations"); obj.pop("attribute_confidence"); obj.pop("hypotheses")
        obj["obb"] = [{"frame": index, "t": float(index), "center": [0, 0.5, 0], "size": [1, 1, 1], "quat": [0, 0, 0, 1], "conf": 0.8} for index in range(10)]
    legacy["meta"]["geometry_frames"] = list(range(10))
    report = assess_evidence_quality(legacy, QUALITY)
    assert report["validation"]["adapted"] and report["decision"] == "warn"
    assert "legacy_evidence_adapted" in _codes(report)
