"""CPU regressions for declared physics sampling, without models or browsers."""
import copy
import json
import math

import numpy as np
from PIL import Image
import pytest

from gwm.perception.gt_timing import validate_render_times
from gwm.perception.visibility import checked_visibility

DT = 1 / 60


def physics_index(times):
    frames = []
    for t in times:
        actual = math.floor(t / DT + .5) * DT
        frames.append({"requested_t": t, "actual_t": actual,
                       "t": actual, "state": {"t": actual}})
    return {"time_sampling": {"mode": "nearest_physics_frame", "dt": DT},
            "frames": frames}


@pytest.mark.parametrize("fps", [24, 25, 30000 / 1001, 30, 60])
@pytest.mark.parametrize("offset", [0, .007])
def test_source_rates_and_offsets_accept_only_declared_nearest_frames(fps, offset):
    times = [offset + i / fps for i in range(6)]
    index = physics_index(times)
    before = copy.deepcopy(index)
    validate_render_times(index, times, {})
    assert index == before


@pytest.mark.parametrize("sampling", [None, {}, {"mode": "exact"},
    {"mode": "nearest_physics_frame"}, {"mode": "other"},
    {"mode": "nearest_physics_frame", "dt": 0},
    {"mode": "nearest_physics_frame", "dt": -.1},
    {"mode": "nearest_physics_frame", "dt": True},
    {"mode": "nearest_physics_frame", "dt": float("nan")},
    {"mode": "nearest_physics_frame", "dt": float("inf")}, "nearest_physics_frame"])
def test_unverified_sampling_cannot_relax_exact_time(sampling):
    index = physics_index([.04])
    if sampling is None:
        index.pop("time_sampling")
    else:
        index["time_sampling"] = sampling
    with pytest.raises(ValueError):
        validate_render_times(index, [.04], {})


@pytest.mark.parametrize("field", ["requested_t", "actual_t", "t", "state"])
def test_quantized_render_still_rejects_wrong_request_or_state(field):
    index = physics_index([.04])
    index["frames"][0][field] = {"t": 2 * DT + .001} if field == "state" else .04
    if field == "requested_t":
        index["frames"][0][field] = .041
    with pytest.raises(ValueError):
        validate_render_times(index, [.04], {})


@pytest.mark.parametrize("requested,actual", [(.04, .04), (.025, DT), (.04, 3 * DT)])
def test_half_step_window_alone_does_not_prove_correct_physics_frame(requested, actual):
    index = physics_index([requested])
    index["frames"][0].update(actual_t=actual, t=actual, state={"t": actual})
    with pytest.raises(ValueError):
        validate_render_times(index, [requested], {})


def test_half_step_rounding_matches_positive_javascript_ties():
    times = [float(np.nextafter(DT / 2, 0)), DT / 2]
    index = {"time_sampling": {"mode": "nearest_physics_frame", "dt": DT},
             "frames": [{"requested_t": t, "actual_t": a, "t": a, "state": {"t": a}}
                        for t, a in zip(times, [0., DT])]}
    validate_render_times(index, times, {})


def test_visibility_retains_quantized_time_provenance_after_serialization():
    frame = physics_index([.04])["frames"][0]
    record = {**frame, "time_sampling": {"mode": "nearest_physics_frame", "dt": DT},
              "object_id": "target", "frame_index": 3, "image_size": [40, 42],
              "program_sha256": "a" * 64, "same_state_verified": True,
              "method": "isolated_instance_id", "fraction": .25}
    record = json.loads(json.dumps(record))
    before = copy.deepcopy(record)
    assert checked_visibility(record, "target", 3, .04, [40, 42], "a" * 64) == (.25, "")
    assert record == before
    record["requested_t"] += .001
    assert checked_visibility(record, "target", 3, .04, [40, 42], "a" * 64) == ("unknown", "time_mismatch")


def test_gt_backends_and_visibility_share_sampling_policy(tmp_path, monkeypatch):
    from gwm.perception import gt_backend as gt
    from gwm.compiler.ids import id_to_color

    program = {"objects": [{"id": "target", "class": "box"}], "static": []}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(program))
    image = tmp_path / "frame.png"
    Image.new("RGB", (40, 40)).save(image)
    visible = np.zeros((42, 40, 3), dtype=np.uint8)
    visible[:8, :8] = id_to_color(1)
    full = np.zeros_like(visible)
    full[:16, :16] = id_to_color(1)
    Image.fromarray(visible).save(tmp_path / "id.png")
    Image.fromarray(full).save(tmp_path / "full.png")
    index = physics_index([.04])
    index["frames"][0].update(camera={"pos": [2 * DT, 0, 1], "quat": [0, 0, 0, 1], "fov": 60},
        files={"depth": "unused", "id": "id.png"}, unoccluded_id={"1": "full.png"},
        visibility_reference_state_unchanged=True)
    monkeypatch.setattr(gt, "_render_gt", lambda *a: (program, index, tmp_path))
    monkeypatch.setattr(gt, "decode_depth_png", lambda *a: np.ones((42, 40)))
    cfg = {"perception": {}}
    geometry = gt.GTGeometryBackend(source, 40).estimate([str(image)], [3], cfg, frame_times=[.04])
    tracks = gt.GTSegmentationBackend(source, 40).segment([str(image)], [3], [], cfg, frame_times=[.04])
    record = tracks.objects[0].visibility_measurements[3]
    assert record["time_sampling"] == index["time_sampling"]
    assert record["requested_t"] == .04 and record["actual_t"] == 2 * DT
    assert checked_visibility(record, "target", 3, .04, [40, 42], geometry.source_program_sha256) == (.25, "")
    from gwm.perception.run import save_visibility_references
    retained = tmp_path / "retained"
    save_visibility_references(tracks, retained)
    assert (retained / "visibility_refs.npz").is_file()
    # JSON is the existing masks_index format; revalidation must not lose policy.
    restored = json.loads(json.dumps(record))
    tracks.objects[0].visibility_measurements[3] = restored
    save_visibility_references(tracks, retained)
    assert checked_visibility(restored, "target", 3, .04, [40, 42], geometry.source_program_sha256) == (.25, "")
