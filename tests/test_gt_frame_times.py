"""Synthetic CPU timing tests; no browser, GT video, models or API calls."""
import copy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from gwm.config import load_config
from gwm.perception.gt_timing import requested_frame_times, validate_render_times


TIMES = [.1, .366667, .6]


def render_index(times):
    return {"frames": [{"t": t, "actual_t": t, "requested_t": t, "state": {"t": t}} for t in times]}


def test_times_are_observations_not_indices_or_fps():
    times = np.array([.1, .866667, 2.366667], dtype=np.float64)
    before = times.copy()
    assert requested_frame_times(["a", "b", "c"], [0, 3, 9], times) == tuple(times)
    np.testing.assert_array_equal(times, before)


@pytest.mark.parametrize("times", [None, [], [.1], [.1, .1], [.2, .1], [0, float("nan")],
                                  [0, float("inf")], [0, True], [-.1, .1], [0, "0.1"],
                                  [.1001, .1002]])
def test_invalid_or_unidentified_times_are_rejected(times):
    with pytest.raises(ValueError):
        requested_frame_times(["a", "b"], [0, 1], times)


@pytest.mark.parametrize("files,indices", [([], []), (["a"], [0, 1]), (["a", "b"], [0, 0]),
                                           (["a", "b"], [-1, 2]), (["a", "b"], [True, 1])])
def test_invalid_frame_association_is_rejected(files, indices):
    with pytest.raises(ValueError):
        requested_frame_times(files, indices, TIMES[:2])


@pytest.mark.parametrize("field", ["requested_t", "actual_t", "t", "state"])
def test_rendered_wrong_time_is_not_accepted(field):
    index = render_index(TIMES)
    index["frames"][1][field] = {"t": .25} if field == "state" else .25
    with pytest.raises(ValueError, match="match source frame time"):
        validate_render_times(index, TIMES, {})


@pytest.mark.parametrize("field", ["requested_t", "actual_t", "t", "state"])
def test_missing_render_time_is_not_invented(field):
    index = render_index(TIMES)
    index["frames"][0].pop(field)
    with pytest.raises(ValueError):
        validate_render_times(index, TIMES, {})


def test_count_and_physics_time_quantization_are_checked():
    with pytest.raises(ValueError, match="count"):
        validate_render_times(render_index(TIMES[:-1]), TIMES, {})
    index = render_index(TIMES)
    for field in ("actual_t", "t"):
        index["frames"][1][field] = 22 / 60
    index["frames"][1]["state"]["t"] = 22 / 60
    before = copy.deepcopy(index)
    validate_render_times(index, TIMES, {})  # 0.333us difference, not relabeled.
    assert index == before
    index["frames"][1]["actual_t"] = 23 / 60
    with pytest.raises(ValueError):
        validate_render_times(index, TIMES, {})


@pytest.mark.parametrize("tolerance", [True, -.1, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        validate_render_times(render_index(TIMES), TIMES, {"perception": {"visibility_time_tolerance_s": tolerance}})


def test_transport_preserves_time_precision(tmp_path, monkeypatch):
    import gwm.feedback.render as renderer
    calls = []
    times = [.1, .366667, 2 / 3]
    def harness(script, args):
        calls.append(args)
        (tmp_path / "index.json").write_text("{}")
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(renderer, "run_harness", harness)
    renderer.render("synthetic", times, tmp_path, 40, 42)
    transported = [float(v) for v in calls[0][calls[0].index("--times") + 1].split(",")]
    assert transported == times
    assert transported[1] != round(times[1], 3)


def test_both_gt_backends_render_actual_work_times(tmp_path, monkeypatch):
    from gwm.perception import gt_backend as gt
    from gwm.compiler.ids import id_to_color
    source = tmp_path / "program.json"
    program = {"objects": [{"id": "target", "class": "box"}], "static": []}
    source.write_text(json.dumps(program))
    image = tmp_path / "frame.png"
    Image.new("RGB", (40, 40)).save(image)
    colors = np.zeros((42, 40, 3), dtype=np.uint8)
    colors[:20, :20] = id_to_color(1)
    Image.fromarray(colors).save(tmp_path / "id.png")
    calls = []
    def render_gt(path, times, width, height, *args):
        calls.append(times)
        index = render_index(times)
        for f in index["frames"]:
            f.update(camera={"pos": [f["t"], 0, 1], "quat": [0, 0, 0, 1], "fov": 60},
                     files={"depth": "unused.png", "id": "id.png"})
        return program, index, tmp_path
    monkeypatch.setattr(gt, "_render_gt", render_gt)
    monkeypatch.setattr(gt, "decode_depth_png", lambda *args: np.ones((42, 40)))
    files, indices = [str(image)] * 3, [0, 3, 9]
    cfg = {"frames": {"sample_fps": 99}, "perception": {}}
    geometry = gt.GTGeometryBackend(source, 40).estimate(files, indices, cfg, frame_times=TIMES)
    tracks = gt.GTSegmentationBackend(source, 40).segment(files, indices, [], cfg, frame_times=TIMES)
    assert calls == [tuple(TIMES), tuple(TIMES)]
    assert geometry.frame_indices == indices
    assert [m[0, 3] for m in geometry.cam_to_world] == TIMES
    assert set(tracks.objects[0].masks) == set(indices)


@pytest.mark.parametrize("kind", ["geometry", "segmentation"])
def test_direct_gt_call_without_times_fails_before_render(kind, monkeypatch):
    from gwm.perception import gt_backend as gt
    monkeypatch.setattr(gt, "_render_gt", lambda *a: pytest.fail("must reject before rendering"))
    with pytest.raises(ValueError, match="explicit frame_times"):
        if kind == "geometry":
            gt.GTGeometryBackend("unused").estimate(["unused"], [0], {})
        else:
            gt.GTSegmentationBackend("unused").segment(["unused"], [0], [], {})


@pytest.mark.parametrize("use_gt", [False, True])
@pytest.mark.parametrize("camera_ablation", [False, True])
def test_driver_uses_selected_work_times_without_changing_neural_api(tmp_path, monkeypatch, use_gt, camera_ablation):
    from gwm.perception import run, gt_backend
    from gwm.perception.base import Geometry, Tracks
    video = tmp_path / "synthetic.bin"
    video.write_bytes(b"source identity only, extractor is stubbed")
    frames = []
    for i in range(10):
        path = tmp_path / f"frame_{i}.png"
        Image.new("RGB", (40, 40)).save(path)
        frames.append({"index": i, "t": i / 4 + .1 + (.016667 if i % 2 else 0), "file": str(path)})
    original = copy.deepcopy(frames)
    expected_indices = [0, 4, 9]
    expected_times = [frames[i]["t"] for i in expected_indices]
    cfg = load_config()
    cfg["frames"]["geometry_frames"] = 3
    cfg["perception"].update(geometry_backend="gt" if use_gt else "static", geometry_fallbacks=[],
                             segmentation_backend="gt" if use_gt else "none", segmentation_fallbacks=[])
    if camera_ablation:
        cfg["perception"]["gt_camera_program"] = "synthetic-program.json"
    geometry = Geometry(expected_indices, [np.eye(3)] * 3, [np.eye(4)] * 3,
                        [np.ones((2, 2))] * 3, backend="gt" if use_gt else "static")
    calls = {}
    def estimate(files, indices, config, **kwargs):
        calls["geometry"] = (indices, kwargs)
        return geometry
    def segment(files, indices, phrases, config, **kwargs):
        calls["segmentation"] = (indices, kwargs)
        return Tracks([], (40, 40), "gt" if use_gt else "none")
    def camera_estimate(files, indices, config, *, frame_times):
        calls["camera"] = (indices, frame_times)
        return geometry
    monkeypatch.setattr(run, "extract_frames", lambda *a, **kw: frames)
    monkeypatch.setattr(run, "prepare_geometry_frames", lambda work, *a: [f["file"] for f in work])
    monkeypatch.setattr(run, "make_geometry_backend", lambda *a: SimpleNamespace(estimate=estimate))
    monkeypatch.setattr(run, "make_segmentation_backend", lambda *a: SimpleNamespace(segment=segment))
    monkeypatch.setattr(gt_backend, "GTGeometryBackend", lambda *a: SimpleNamespace(estimate=camera_estimate))
    monkeypatch.setattr(run, "build_evidence", lambda *a, **kw: {"meta": {}})
    run.run_perception(str(video), "synthetic", tmp_path / "out", cfg, client=None)
    expected_kwargs = {"frame_times": expected_times} if use_gt else {}
    assert calls["geometry"] == (expected_indices, expected_kwargs)
    assert calls["segmentation"] == (expected_indices, expected_kwargs)
    if camera_ablation:
        assert calls["camera"] == (expected_indices, expected_times)
    assert frames == original
