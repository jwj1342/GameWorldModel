"""The frame sampler must retain decoded source timing, not output numbering time."""
import json
import re
from types import SimpleNamespace

from PIL import Image
import pytest

from gwm.perception import frames
from gwm.perception.provenance import file_sha256


def test_one_fps_selects_source_frames_14_plus_30k_with_true_pts(tmp_path, monkeypatch):
    video = tmp_path / "synthetic_clip.mp4"
    video.write_bytes(b"synthetic container placeholder")
    monkeypatch.setattr(frames, "probe", lambda *args: {"width": 16, "height": 12, "fps": 30.0, "duration": 6.0})
    seen = {}

    def fake_run(command, **kwargs):
        if "-show_frames" in command:
            return SimpleNamespace(stdout=json.dumps({"frames": [
                {"best_effort_timestamp_time": f"{index / 30:.6f}"} for index in range(180)]}))
        selected = [int(value) for value in re.findall(r"eq\(n\\,(\d+)\)", command[command.index("-vf") + 1])]
        seen["indices"] = selected
        output = tmp_path / "sampled"
        for ordinal in range(1, len(selected) + 1):
            Image.new("RGB", (16, 12), "red").save(output / f"f_{ordinal:05d}.jpg")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)
    selected = frames.extract_frames(video, tmp_path / "sampled", sample_fps=1.0,
                                     max_seconds=6.0, max_side=16)
    assert seen["indices"] == [14, 44, 74, 104, 134, 164]
    assert [item["source_frame_index"] for item in selected] == seen["indices"]
    assert [round(item["source_pts_s"], 3) for item in selected] == [0.467, 1.467, 2.467, 3.467, 4.467, 5.467]
    assert selected[0]["t"] == 0.466667 and selected[0]["t"] != 0.0
    assert all(item["video_sha256"] == file_sha256(video) for item in selected)
    assert all(item["image_sha256"] == file_sha256(item["file"]) for item in selected)
    manifest = json.loads((tmp_path / "sampled" / "frames.json").read_text())
    assert manifest["video_sha256"] == file_sha256(video)


def test_missing_decoded_pts_cannot_be_replaced_with_sample_index(tmp_path, monkeypatch):
    video = tmp_path / "synthetic_clip.mp4"
    video.write_bytes(b"synthetic")
    monkeypatch.setattr(frames, "probe", lambda *args: {"width": 16, "height": 12, "fps": 30.0, "duration": 1.0})
    monkeypatch.setattr(frames.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"frames": [{"best_effort_timestamp_time": None}]})))
    with pytest.raises(ValueError, match="no PTS"):
        frames.extract_frames(video, tmp_path / "sampled", sample_fps=1.0, max_seconds=1.0)
