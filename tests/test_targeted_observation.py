"""Targeted observation tests use a synthetic DetectionProvider, not model outputs."""
import json
from pathlib import Path
import hashlib
import sys
import types

from PIL import Image
import pytest

from gwm.feedback.active_perception import request_identifier
from gwm.perception.base import DetectionRecord
from gwm.perception import targeted
from gwm.perception.provenance import evidence_sha256


def _request(object_id="item_a", *, start=1.0, end=3.0, status="approved"):
    request = {"target": {"program_object_id": object_id, "evidence_object_id": object_id,
                          "track_id": f"track_{object_id}", "class_guess": "item"},
               "verification_signals": [{"code": "object_never_visible", "path": "/objects/0"}],
               "suggested_video_range": {"start_s": start, "end_s": end},
               "suggested_keyframes": [{"t": 1.5}, {"t": 2.5}, {"t": 2.5}],
               "review_status": status}
    request["request_id"] = request_identifier(request)
    return request


def _source(tmp_path, requests):
    path = tmp_path / "active_perception_requests.json"
    video = tmp_path / "original.mp4"
    video.write_bytes(b"synthetic video placeholder; extraction is mocked in these unit tests")
    objects = {}
    for request in requests:
        target = request["target"]
        object_id = target["evidence_object_id"]
        objects[object_id] = {"id": object_id, "track_id": target["track_id"], "class_guess": "item",
                              "confidence": 0.8, "is_dynamic": True, "obb": [],
                              "motion_guess": {"type": "prismatic", "conf": 0.8}, "contacts": [], "observations": [],
                              "attribute_confidence": {"class": 0.8, "identity": 0.8, "geometry": 0.8, "motion": 0.8},
                              "hypotheses": []}
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    evidence = {"schema_version": "2.0", "meta": {"clip": "synthetic", "duration": 4.0, "scale": "relative",
                                                       "source_video_sha256": video_hash},
                "frames": [{"frame_index": i, "t": float(i), "source": {"kind": "video_frame", "ref": f"f_{i}.jpg"}}
                           for i in range(5)],
                "camera": {"intrinsics": {}, "poses": []}, "static": {}, "objects": list(objects.values()), "keyframes": []}
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    path.write_text(json.dumps({"status": "ready", "source_binding": {"video_sha256": video_hash,
                                  "evidence_sha256": evidence_sha256(evidence)}, "requests": requests}), encoding="utf-8")
    return path, video, evidence_path


def _fake_extractor(monkeypatch):
    calls = []
    monkeypatch.setattr(targeted, "probe", lambda video, ffprobe_bin="ffprobe":
                        {"width": 16, "height": 12, "fps": 2.0, "duration": 4.0})

    def extract(video, out_dir, sample_fps, max_seconds, max_side, start, ffmpeg_bin, ffprobe_bin):
        calls.append((start, max_seconds))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True)
        frames = []
        for index, offset in enumerate((0.0, 0.5, 1.0, 1.5, 2.0)):
            path = out_dir / f"f_{index:05d}.jpg"
            Image.new("RGB", (16, 12), (index * 20, 10, 10)).save(path)
            frames.append({"index": index, "t": offset, "file": str(path), "width": 16, "height": 12})
        return frames

    monkeypatch.setattr(targeted, "extract_frames", extract)
    return calls


class SyntheticProvider:
    name = "synthetic_test_provider"

    def __init__(self):
        self.calls = []

    def detect(self, image, frame_index, phrases, cfg):
        self.calls.append((frame_index, tuple(phrases), image.size))
        match = DetectionRecord(f"match_{frame_index}", frame_index, "item", 0.9, (1, 1, 5, 5), source=self.name)
        other = DetectionRecord(f"other_{frame_index}", frame_index, "unrelated", 0.7, (6, 2, 9, 6), source=self.name)
        return [match, match, other]


@pytest.mark.parametrize("overrides,expected", [
    ([], (0.25, 0.2, 8)),
    (["--grounding-box-threshold", "0.31", "--grounding-text-threshold", "0.27", "--max-objects", "3"],
     (0.31, 0.27, 3)),
])
def test_cli_passes_configured_grounding_thresholds_to_provider(tmp_path, monkeypatch, overrides, expected):
    _fake_extractor(monkeypatch)
    monkeypatch.setattr(targeted, "load_config", lambda: {"perception": {
        "grounding_box_threshold": 0.25, "grounding_text_threshold": 0.2, "max_objects": 8},
        "targeted_observation": {"sample_fps": 2.0}})
    request = _request()
    source, video, evidence = _source(tmp_path, [request])
    model_dir = tmp_path / "model_fixture"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"test placeholder; never loaded")
    calls = []

    class ConfigCheckingProvider:
        name = "synthetic_cli_provider"

        def __init__(self, path):
            self.model_dir = path

        def detect(self, image, frame_index, phrases, cfg):
            calls.append((frame_index, dict(cfg["perception"])))
            return []

    fake_module = types.ModuleType("gwm.perception.grounded_sam2_backend")
    fake_module.GroundingDinoDetectionProvider = ConfigCheckingProvider
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    targeted.main(["--requests", str(source), "--video", str(video), "--evidence", str(evidence),
                   "--out", str(tmp_path / "cli_out"), "--provider", "grounding-dino",
                   "--model-dir", str(model_dir), *overrides])
    assert calls and all((settings["grounding_box_threshold"], settings["grounding_text_threshold"],
                          settings["max_objects"]) == expected for _, settings in calls)
    report = json.loads((tmp_path / "cli_out" / "targeted_observations.json").read_text(encoding="utf-8"))
    assert report["results"][0]["status"] == "no_match"
    assert tuple(report["provenance"]["detection_settings"][key] for key in
                 ("grounding_box_threshold", "grounding_text_threshold", "max_objects")) == expected


@pytest.mark.parametrize("args", [
    ["--grounding-box-threshold", "nan"], ["--grounding-text-threshold", "1.1"],
    ["--max-objects", "0"],
])
def test_cli_rejects_invalid_detection_thresholds_before_model_loading(args, tmp_path):
    with pytest.raises(SystemExit) as exc:
        targeted.main(["--requests", "unused.json", "--video", "unused.mp4",
                       "--out", str(tmp_path / "unused"), "--provider", "grounding-dino", *args])
    assert exc.value.code == 2


def test_reviewed_range_selects_real_frame_refs_filters_objects_and_deduplicates(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    request = _request()
    source, video, evidence = _source(tmp_path, [request])
    provider = SyntheticProvider()
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", provider,
                                                 {"max_frames_per_request": 2}, evidence_file=evidence)
    result = report["results"][0]
    assert report["status"] == "completed" and result["status"] == "observed"
    assert calls == [(1.0, 2.0)]
    assert [frame["video_time_s"] for frame in result["selected_frames"]] == [1.5, 2.5]
    assert len(result["observations"]) == 2
    assert {item["detection"]["class_label"] for item in result["observations"]} == {"item"}
    assert all(item["identity_status"] == "candidate_unverified" for item in result["observations"])
    assert provider.calls == [(1, ("item",), (16, 12)), (3, ("item",), (16, 12))]
    assert report["provenance"]["video_sha256"]
    assert json.loads((tmp_path / "out" / "targeted_observations.json").read_text())["results"][0]["request_id"] == request["request_id"]


def test_video_boundary_clamps_one_request_and_reports_other_failure(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    inside, outside = _request(end=8.0), _request("item_b", start=5.0, end=6.0)
    source, video, evidence = _source(tmp_path, [inside, outside])
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", evidence_file=evidence)
    first, second = report["results"]
    assert first["actual_range"] == {"start_s": 1.0, "end_s": 4.0}
    assert first["status"] == "provider_unavailable" and first["observations"] == []
    assert len(first["selected_frames"]) > 0
    assert second["status"] == "failed" and "outside" in second["reason"]
    assert calls == [(1.0, 3.0)]
    assert report["status"] == "partial"


def test_selection_requires_approval_or_explicit_id_and_deduplicates_request(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    request = _request(status="pending")
    source, video, evidence = _source(tmp_path, [request, request])
    none = targeted.run_targeted_observations(source, video, tmp_path / "not_selected", evidence_file=evidence)
    assert none["status"] == "nothing_selected" and calls == []
    selected = targeted.run_targeted_observations(source, video, tmp_path / "selected",
                                                   selected_request_ids=[request["request_id"]], evidence_file=evidence)
    assert len(selected["results"]) == 1 and selected["duplicate_request_ids"] == [request["request_id"]]
    assert len(calls) == 1


def test_legacy_request_without_id_remains_selectable(tmp_path, monkeypatch):
    _fake_extractor(monkeypatch)
    request = _request()
    expected = request.pop("request_id")
    source, video, evidence = _source(tmp_path, [request])
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", selected_request_ids=[expected], evidence_file=evidence)
    assert report["results"][0]["request_id"] == expected


def test_one_request_provider_failure_does_not_hide_other_result(tmp_path, monkeypatch):
    _fake_extractor(monkeypatch)
    first, second = _request("item_a"), _request("item_b")
    first["search_phrases"], second["search_phrases"] = ["break"], ["item"]
    source, video, evidence = _source(tmp_path, [first, second])

    class FailingProvider(SyntheticProvider):
        def detect(self, image, frame_index, phrases, cfg):
            if phrases == ["break"]:
                raise RuntimeError("local provider inference failed")
            return super().detect(image, frame_index, phrases, cfg)

    report = targeted.run_targeted_observations(source, video, tmp_path / "out", FailingProvider(), evidence_file=evidence)
    assert report["status"] == "partial"
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["reason_code"] == "provider_frame_failure"
    assert "local provider inference failed" in report["results"][0]["frame_failures"][0]["reason"]
    assert report["results"][1]["status"] == "observed"


def test_invalid_phrase_and_execution_issue_never_fabricate_observations(tmp_path, monkeypatch):
    _fake_extractor(monkeypatch)
    request = _request()
    request["target"]["class_guess"] = "unknown"
    source, video, evidence = _source(tmp_path, [request])
    invalid = targeted.run_targeted_observations(source, video, tmp_path / "out", SyntheticProvider(), evidence_file=evidence)
    assert invalid["results"][0]["status"] == "failed" and invalid["results"][0]["observations"] == []
    source.write_text(json.dumps({"status": "execution_blocked", "execution_issues": [{"code": "missing_pass_file"}],
                                  "requests": [request]}), encoding="utf-8")
    blocked = targeted.run_targeted_observations(source, video, tmp_path / "blocked", SyntheticProvider())
    assert blocked["status"] == "source_execution_blocked"
    assert blocked["results"][0]["selected_frames"] == []


def test_mismatched_video_is_rejected_before_extraction_or_provider(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    source, video, evidence = _source(tmp_path, [_request()])
    video.write_bytes(b"a different video with the same filename")
    provider = SyntheticProvider()
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", provider, evidence_file=evidence)
    assert report["status"] == "source_mismatch"
    assert {item["code"] for item in report["source_issues"]} >= {
        "request_video_hash_mismatch", "evidence_video_hash_mismatch"}
    assert report["results"][0]["observations"] == []
    assert not calls and not provider.calls


def test_mismatched_evidence_content_and_track_are_rejected(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    source, video, evidence_path = _source(tmp_path, [_request()])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["objects"][0]["track_id"] = "different_track"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    provider = SyntheticProvider()
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", provider, evidence_file=evidence_path)
    assert report["status"] == "source_mismatch"
    assert {item["code"] for item in report["source_issues"]} >= {
        "request_evidence_hash_mismatch", "request_track_mismatch"}
    assert not calls and not provider.calls


def test_request_object_id_not_in_evidence_is_rejected(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    source, video, evidence = _source(tmp_path, [_request()])
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["requests"][0]["target"]["evidence_object_id"] = "nonexistent"
    source.write_text(json.dumps(payload), encoding="utf-8")
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", SyntheticProvider(), evidence_file=evidence)
    assert report["status"] == "source_mismatch"
    assert "request_object_mismatch" in {item["code"] for item in report["source_issues"]}
    assert not calls and report["results"][0]["observations"] == []


def test_bad_target_isolated_from_other_selected_request(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    source, video, evidence = _source(tmp_path, [_request("item_a"), _request("item_b")])
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["requests"][0]["target"]["evidence_object_id"] = "nonexistent"
    source.write_text(json.dumps(payload), encoding="utf-8")
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", SyntheticProvider(), evidence_file=evidence)
    assert report["status"] == "partial"
    assert report["results"][0]["status"] == "rejected"
    assert report["results"][1]["status"] == "observed"
    assert len(calls) == 1


def test_provider_requires_evidence_binding_but_preview_does_not(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    source, video, evidence = _source(tmp_path, [_request()])
    provider = SyntheticProvider()
    blocked = targeted.run_targeted_observations(source, video, tmp_path / "blocked", provider)
    assert blocked["status"] == "source_mismatch"
    assert blocked["source_issues"][0]["code"] == "missing_evidence"
    assert not calls and not provider.calls
    preview = targeted.run_targeted_observations(source, video, tmp_path / "preview")
    assert preview["status"] == "provider_unavailable" and len(preview["results"][0]["selected_frames"]) > 0
    assert preview["results"][0]["observations"] == []
