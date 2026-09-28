"""Targeted observation tests use a synthetic DetectionProvider, not model outputs."""
import json
from pathlib import Path

from PIL import Image

from gwm.feedback.active_perception import request_identifier
from gwm.perception.base import DetectionRecord
from gwm.perception import targeted


def _request(object_id="item_a", *, start=1.0, end=3.0, status="approved"):
    request = {"target": {"program_object_id": object_id, "evidence_object_id": object_id,
                          "track_id": "track_a", "class_guess": "item"},
               "verification_signals": [{"code": "object_never_visible", "path": "/objects/0"}],
               "suggested_video_range": {"start_s": start, "end_s": end},
               "suggested_keyframes": [{"t": 1.5}, {"t": 2.5}, {"t": 2.5}],
               "review_status": status}
    request["request_id"] = request_identifier(request)
    return request


def _source(tmp_path, requests):
    path = tmp_path / "active_perception_requests.json"
    path.write_text(json.dumps({"status": "ready", "requests": requests}), encoding="utf-8")
    video = tmp_path / "original.mp4"
    video.write_bytes(b"synthetic video placeholder; extraction is mocked in these unit tests")
    return path, video


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


def test_reviewed_range_selects_real_frame_refs_filters_objects_and_deduplicates(tmp_path, monkeypatch):
    calls = _fake_extractor(monkeypatch)
    request = _request()
    source, video = _source(tmp_path, [request])
    provider = SyntheticProvider()
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", provider,
                                                 {"max_frames_per_request": 2})
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
    source, video = _source(tmp_path, [inside, outside])
    report = targeted.run_targeted_observations(source, video, tmp_path / "out")
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
    source, video = _source(tmp_path, [request, request])
    none = targeted.run_targeted_observations(source, video, tmp_path / "not_selected")
    assert none["status"] == "nothing_selected" and calls == []
    selected = targeted.run_targeted_observations(source, video, tmp_path / "selected",
                                                   selected_request_ids=[request["request_id"]])
    assert len(selected["results"]) == 1 and selected["duplicate_request_ids"] == [request["request_id"]]
    assert len(calls) == 1


def test_legacy_request_without_id_remains_selectable(tmp_path, monkeypatch):
    _fake_extractor(monkeypatch)
    request = _request()
    expected = request.pop("request_id")
    source, video = _source(tmp_path, [request])
    report = targeted.run_targeted_observations(source, video, tmp_path / "out", selected_request_ids=[expected])
    assert report["results"][0]["request_id"] == expected


def test_one_request_provider_failure_does_not_hide_other_result(tmp_path, monkeypatch):
    _fake_extractor(monkeypatch)
    first, second = _request("item_a"), _request("item_b")
    first["search_phrases"], second["search_phrases"] = ["break"], ["item"]
    source, video = _source(tmp_path, [first, second])

    class FailingProvider(SyntheticProvider):
        def detect(self, image, frame_index, phrases, cfg):
            if phrases == ["break"]:
                raise RuntimeError("local provider inference failed")
            return super().detect(image, frame_index, phrases, cfg)

    report = targeted.run_targeted_observations(source, video, tmp_path / "out", FailingProvider())
    assert report["status"] == "partial"
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["reason_code"] == "provider_frame_failure"
    assert "local provider inference failed" in report["results"][0]["frame_failures"][0]["reason"]
    assert report["results"][1]["status"] == "observed"


def test_invalid_phrase_and_execution_issue_never_fabricate_observations(tmp_path, monkeypatch):
    _fake_extractor(monkeypatch)
    request = _request()
    request["target"]["class_guess"] = "unknown"
    source, video = _source(tmp_path, [request])
    invalid = targeted.run_targeted_observations(source, video, tmp_path / "out", SyntheticProvider())
    assert invalid["results"][0]["status"] == "failed" and invalid["results"][0]["observations"] == []
    source.write_text(json.dumps({"status": "execution_blocked", "execution_issues": [{"code": "missing_pass_file"}],
                                  "requests": [request]}), encoding="utf-8")
    blocked = targeted.run_targeted_observations(source, video, tmp_path / "blocked", SyntheticProvider())
    assert blocked["status"] == "source_execution_blocked"
    assert blocked["results"][0]["selected_frames"] == []
