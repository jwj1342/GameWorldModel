"""Synthetic wiring test, not a measurement of real-model perception accuracy."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from PIL import Image
import pytest

from gwm.feedback.active_perception import plan_active_perception_requests
from gwm.perception.base import DetectionRecord
from gwm.perception.contract import validate_evidence
from gwm.perception import targeted
from gwm.perception.reviewed_update import apply_reviewed_targeted_candidate
from gwm.perception.provenance import declared_video_sha256
from gwm.perception.update import rollback_evidence_update


def _base(video_hash):
    return {
        "schema_version": "2.0",
        "meta": {"clip": "synthetic_link_test", "duration": 2.0, "scale": "unknown",
                 "source_video": {"sha256": video_hash}},
        "frames": [{"frame_index": i, "t": float(i),
                    "source": {"kind": "video_frame", "ref": f"synthetic://frame/{i}"}}
                   for i in range(2)],
        "camera": {"intrinsics": {}, "poses": []}, "static": {}, "keyframes": [],
        "objects": [{"id": "fixture_object", "track_id": "fixture_track", "class_guess": "item",
                     "confidence": "unknown", "is_dynamic": "unknown", "obb": [],
                     "motion_guess": {"type": "unknown", "conf": "unknown"}, "contacts": [],
                     "attribute_confidence": {key: "unknown" for key in ("class", "identity", "geometry", "motion")},
                     "hypotheses": [],
                     "observations": [{"frame_index": 1, "track_id": "fixture_track", "t": 1.0,
                                       "bbox": "unknown", "mask_ref": "unknown", "visible_fraction": "unknown",
                                       "visibility_state": "unknown", "depth": "unknown",
                                       "source": {"geometry": "unknown", "segmentation": "unknown", "tracking": "synthetic_fixture"},
                                       "confidence": {"geometry": "unknown", "segmentation": "unknown", "tracking": "unknown"}}]}],
    }


class FixtureProvider:
    name = "synthetic_fixture_provider"

    def detect(self, image, frame_index, phrases, cfg):
        assert image.size == (16, 12) and frame_index == 0 and phrases == ["item"]
        return [DetectionRecord("fixture_detection_1", frame_index, "item", 0.9,
                                (1, 2, 8, 10), source=self.name)]


def _chain(tmp_path, monkeypatch):
    video = tmp_path / "synthetic_video_placeholder.mp4"
    video.write_bytes(b"synthetic fixture bytes; no real video decoder or model")
    base = _base(hashlib.sha256(video.read_bytes()).hexdigest())
    assert validate_evidence(base, adapt_v1=False)["ok"]
    signal = {"version": "1.0", "diagnostics": [{"stage": "visibility", "severity": "warning",
               "code": "object_never_visible", "path": "/objects/0", "message": "synthetic test signal",
               "hint": "review synthetic fixture"}], "metrics": {"frames": [{"index": 1, "t": 1.0}], "objects": {}}}
    program = {"objects": [{"id": "fixture_object", "motion": {"type": "static"}}]}
    bundle = plan_active_perception_requests(signal, base, program, source_video=str(video))
    assert bundle["status"] == "ready" and len(bundle["requests"]) == 1
    request = bundle["requests"][0]
    assert request["review_status"] == "pending"
    request["review_status"] = "approved"
    request["review_decision"] = {"status": "approved", "reviewer": "synthetic_test_reviewer"}
    evidence_file = tmp_path / "synthetic_evidence.json"
    evidence_file.write_text(json.dumps(base), encoding="utf-8")
    request_file = tmp_path / "active_perception_requests.json"
    request_file.write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setattr(targeted, "probe", lambda *args: {"width": 16, "height": 12,
                                                            "fps": 1.0, "duration": 2.0})

    def fixture_decode(video_file, out_dir, sample_fps, max_seconds, max_side, start, ffmpeg_bin, ffprobe_bin):
        out_dir.mkdir(parents=True)
        path = out_dir / "synthetic_decoded_frame.png"
        Image.new("RGB", (16, 12), "blue").save(path)
        return [{"index": 0, "t": 1.0 - start, "source_pts_s": 1.0,
                 "source_frame_index": 1, "file": str(path), "width": 16, "height": 12}]

    monkeypatch.setattr(targeted, "extract_frames", fixture_decode)
    report = targeted.run_targeted_observations(request_file, video, tmp_path / "targeted",
                                                FixtureProvider(), evidence_file=evidence_file)
    return base, bundle, report, request["request_id"]


def test_synthetic_verification_to_reversible_evidence_update(tmp_path, monkeypatch):
    base, bundle, report, request_id = _chain(tmp_path, monkeypatch)
    assert report["status"] == "completed" and report["results"][0]["status"] == "observed"
    frame = report["results"][0]["selected_frames"][0]
    assert frame["time_basis"] == "decoded_source_pts" and frame["video_time_s"] == 1.0
    assert frame["source_frame_index"] == 1
    decision = {"status": "confirmed", "reviewer": "synthetic_test_reviewer"}
    updated, event = apply_reviewed_targeted_candidate(base, bundle, report, request_id,
                                                         "fixture_detection_1", decision)
    assert event["provenance"]["request_ref"] == request_id
    assert event["video_sha256"] == bundle["source_binding"]["video_sha256"] == report["provenance"]["video_sha256"]
    assert event["provenance"]["frame_pts_s"] == frame["video_time_s"] == base["frames"][1]["t"]
    assert event["counts"]["observations_updated"] == 1
    assert updated["objects"][0]["observations"][0]["bbox"]["values"] == [1.0, 2.0, 8.0, 10.0]
    assert event["quality"]["measured_coverage_delta"]["bbox"] == 1.0
    assert event["validation"]["after"]["ok"] and rollback_evidence_update(updated, event) == base


@pytest.mark.parametrize("change,expected", [
    ("approval", "reviewer approval"), ("identity", "identity confirmation"),
    ("request_id", "request ID"), ("video_hash", "video hashes"),
    ("evidence_hash", "Evidence content"), ("estimated_time", "decoded source PTS"),
    ("wrong_pts", "decoded PTS differs"), ("wrong_frame", "frame index or decoded PTS"),
])
def test_link_rejects_unapproved_or_broken_provenance(tmp_path, monkeypatch, change, expected):
    base, bundle, report, request_id = _chain(tmp_path, monkeypatch)
    decision = {"status": "confirmed", "reviewer": "synthetic_test_reviewer"}
    base, bundle, report = deepcopy((base, bundle, report))
    if change == "approval":
        bundle["requests"][0]["review_decision"] = {}
    elif change == "identity":
        decision = {"status": "pending", "reviewer": "synthetic_test_reviewer"}
    elif change == "request_id":
        bundle["requests"][0]["request_id"] = "wrong-request"
    elif change == "video_hash":
        report["provenance"]["video_sha256"] = "f" * 64
    elif change == "evidence_hash":
        report["provenance"]["evidence_sha256"] = "f" * 64
    elif change == "estimated_time":
        report["results"][0]["selected_frames"][0]["time_basis"] = "estimated_sample_time"
    elif change == "wrong_pts":
        report["results"][0]["selected_frames"][0]["video_time_s"] = 1.2
        report["results"][0]["observations"][0]["video_time_s"] = 1.2
    elif change == "wrong_frame":
        report["results"][0]["selected_frames"][0]["source_frame_index"] = 0
    with pytest.raises(ValueError, match=expected):
        apply_reviewed_targeted_candidate(base, bundle, report, request_id,
                                          "fixture_detection_1", decision)


def test_source_hash_layouts_must_not_conflict():
    evidence = _base("a" * 64)
    assert declared_video_sha256(evidence) == "a" * 64
    evidence["meta"]["source_video_sha256"] = "a" * 64
    assert declared_video_sha256(evidence) == "a" * 64
    evidence["meta"]["source_video_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="conflict"):
        declared_video_sha256(evidence)
