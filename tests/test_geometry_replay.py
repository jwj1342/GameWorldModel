"""CPU synthetic fixtures only; no historical artifacts, inference or rendering."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gwm.config import load_config
from gwm.perception.base import Geometry, Tracks, TrackedObject
from gwm.perception.evidence import align_world, build_evidence, obb_from_points, unproject, to_world, _fix
from gwm.perception.geometry_replay import load_replay, save_replay


def fixture():
    # Camera-facing plane exercises the existing steep-view fallback.
    depth = np.ones((80, 80), np.float32)
    mask = np.zeros((160, 160), bool)
    mask[48:112, 48:112] = True
    depth[24:56, 24:56] = np.float32(.7312345)
    K = np.array([[80., 0, 40], [0, 80, 40], [0, 0, 1]])
    geom = Geometry([0, 1, 2], [K.copy() for _ in range(3)],
                    [np.eye(4) for _ in range(3)], [depth.copy() for _ in range(3)],
                    [np.ones_like(depth) for _ in range(3)],
                    [np.zeros_like(depth, dtype=bool) for _ in range(3)], backend="synthetic")
    tracks = Tracks([TrackedObject("sample", "box", .9, {i: mask.copy() for i in range(3)})],
                    (160, 160), "synthetic")
    frames = [{"index": i, "t": i * .1} for i in range(3)]
    cfg = load_config(["configs/default.yaml"])
    cfg["perception"]["motion"]["use_registration"] = False
    return geom, tracks, frames, cfg


def test_alignment_recording_preserves_decision_and_inlier_identity():
    geom, tracks, _, cfg = fixture()
    baseline = align_world(geom, tracks, cfg)
    record = {}
    recorded = align_world(geom, tracks, cfg, replay=record)
    np.testing.assert_array_equal(baseline[0], recorded[0])
    assert baseline[1:] == recorded[1:]
    assert record["selected"] is not None
    # Independently reconstruct the sampled input pixels, then check support.
    points = []
    for fi, y, x in record["static_point_sources"]:
        k = geom.frame_indices.index(fi)
        d, K = geom.depth[k][y, x], geom.intrinsics[k]
        camera = np.array([[(x-K[0, 2])/K[0, 0]*d, (y-K[1, 2])/K[1, 1]*d, d]])
        points.append(to_world(camera, _fix(geom.cam_to_world[k]))[0])
    points = np.array(points)
    for candidate in record["candidates"]:
        support = points[candidate["inlier_point_indices"]]
        assert np.max(np.abs(support @ candidate["normal"] + candidate["offset"])) < record["ransac_threshold"]
    selected = record["candidates"][record["selected"]]
    T = np.array(record["T"])
    aligned = points[selected["inlier_point_indices"]] @ T[:3, :3].T + T[:3, 3]
    np.testing.assert_allclose(aligned[:, 1], 0, atol=1e-12)


@pytest.mark.parametrize("optional", [True, False])
def test_lossless_roundtrip_and_replay(tmp_path, optional):
    geom, tracks, frames, cfg = fixture()
    if not optional: geom.depth_conf = geom.dynamic_mask = None
    record = {}
    align_world(geom, tracks, cfg, replay=record)
    cfg["api_key"] = "synthetic-secret"
    out = tmp_path / "replay"
    save_replay(out, geom, tracks, frames, cfg, record, [], "production_in_memory")
    g, tr, manifest = load_replay(out)
    assert not np.array_equal(geom.depth[0], geom.depth[0].astype(np.float16).astype(np.float32))
    for name in ("intrinsics", "cam_to_world", "depth", "depth_conf", "dynamic_mask"):
        a, b = getattr(geom, name), getattr(g, name)
        if a is None: assert b is None
        else:
            for aa, bb in zip(a, b):
                assert aa.dtype == bb.dtype
                np.testing.assert_array_equal(aa, bb)
    for fi, mask in tracks.objects[0].masks.items():
        np.testing.assert_array_equal(mask, tr.objects[0].masks[fi])
    replay = {}
    align_world(g, tr, manifest["config"], replay=replay)
    assert replay == record
    assert manifest["config"]["api_key"] == "<redacted>"
    assert "synthetic-secret" not in (out / "manifest.json").read_text()
    assert manifest["physical_ground_correctness"] == "unknown"
    assert len(manifest["source_sha256"]) == 6


@pytest.mark.parametrize("object_depth,expected_branch", [(.7312345, "guard_rejected"), (.99, "applied")])
def test_build_records_raw_obb_without_changing_evidence(tmp_path, monkeypatch, object_depth, expected_branch):
    from gwm.perception import overlay
    monkeypatch.setattr(overlay, "render_overlay", lambda *a, **kw: None)
    geom, tracks, frames, cfg = fixture()
    for depth in geom.depth: depth[24:56, 24:56] = np.float32(object_depth)
    a = build_evidence("synthetic", frames, geom, tracks, cfg, tmp_path / "off", [], [])
    assert not (tmp_path / "off" / "geometry_replay").exists()
    cfg["perception"]["geometry_replay"] = True
    b = build_evidence("synthetic", frames, geom, tracks, cfg, tmp_path / "on", [], [])
    a["meta"].pop("created_at"); b["meta"].pop("created_at")
    assert a == b  # Includes camera, geometry, motion and ground proximity.
    g, tr, manifest = load_replay(tmp_path / "on" / "geometry_replay")
    T, scale, _ = align_world(g, tr, cfg)
    from gwm.perception.masks import resize_mask
    for sample in manifest["objects"]:
        i = sample["frame"]
        mask = resize_mask(tr.objects[0].masks[i], g.size())
        P = to_world(unproject(g.depth[i], g.intrinsics[i], mask, stride=2), _fix(g.cam_to_world[i]))
        raw = obb_from_points((P @ T[:3, :3].T + T[:3, 3]) * scale)
        assert raw == sample["raw_obb"]
        assert sample["contact_branch"] == expected_branch
        assert "pre_smoothing_obb" in sample
        if expected_branch == "applied":
            assert sample["raw_obb"]["center"] != sample["pre_smoothing_obb"]["center"]
    with pytest.raises(FileExistsError):
        build_evidence("synthetic", frames, geom, tracks, cfg, tmp_path / "on", [], [])


def test_missing_failed_and_legacy_are_not_success(tmp_path):
    with pytest.raises(ValueError, match="historical record missing"):
        load_replay(tmp_path)
    geom, tracks, frames, cfg = fixture()
    # No valid depth means no selected plane, not an invented y=0 success.
    geom.depth = [np.zeros_like(d) for d in geom.depth]
    record = {}
    _, _, info = align_world(geom, tracks, cfg, replay=record)
    assert info["ground"] == "not_found" and record["selected"] is None
    save_replay(tmp_path / "legacy", geom, tracks, frames, cfg, record, [], "legacy_lossy_cache")
    _, _, m = load_replay(tmp_path / "legacy")
    assert m["historical_exactness"] == "unavailable"
    assert m["alignment"]["selected"] is None
    original = (tmp_path / "legacy" / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_replay(tmp_path / "legacy", geom, tracks, frames, cfg, record, [], "production_in_memory")
    assert (tmp_path / "legacy" / "manifest.json").read_bytes() == original
    with (tmp_path / "legacy" / "inputs.npz").open("ab") as f: f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_replay(tmp_path / "legacy")


def test_subsampled_support_still_identifies_original_pixels():
    # >60,000 points exercises random subsampling. Only alignment is computed.
    geom = Geometry([0], [np.array([[1000., 0, 500], [0, 1000, 500], [0, 0, 1]])],
                    [np.eye(4)], [np.ones((1000, 1000), np.float32)])
    tracks = Tracks([], (1000, 1000))
    cfg = {"perception": {"ground_ransac_thresh_m": .02}}
    baseline = align_world(geom, tracks, cfg)
    record = {}
    result = align_world(geom, tracks, cfg, replay=record)
    np.testing.assert_array_equal(result[0], baseline[0])
    assert result[1:] == baseline[1:]
    sources = record["static_point_sources"]
    assert len(sources) == len({tuple(p) for p in sources}) == 60000
    assert all(fi == 0 and y % 4 == x % 4 == 0 and 0 <= y < 1000 and 0 <= x < 1000
               for fi, y, x in sources)
    assert len(record["candidates"][record["selected"]]["inlier_point_indices"]) == 60000


def test_disabled_recording_keeps_legacy_call_shape(tmp_path, monkeypatch):
    from gwm.perception import evidence, overlay
    geom, tracks, frames, cfg = fixture()
    calls = []

    def legacy_align(*args):
        calls.append(len(args))
        return np.eye(4), 1., {"ground": "not_found"}

    monkeypatch.setattr(evidence, "align_world", legacy_align)
    monkeypatch.setattr(overlay, "render_overlay", lambda *a, **kw: None)
    evidence.build_evidence("synthetic", frames, geom, tracks, cfg, tmp_path, [], [])
    assert calls == [3]
    assert not (tmp_path / "geometry_replay").exists()


def test_source_identity_and_decoded_frame_fields_roundtrip(tmp_path):
    geom, tracks, frames, cfg = fixture()
    video_hash = "a" * 64
    for f in frames:
        f.update(video_sha256=video_hash, source_pts_s=f["t"] + .467,
                 source_frame_index=14 + f["index"] * 3, image_sha256="b" * 64)
    before = copy.deepcopy(frames)
    save_replay(tmp_path / "replay", geom, tracks, frames, cfg, {}, [], "production_in_memory",
                source_video_sha256=video_hash.upper())
    _, _, m = load_replay(tmp_path / "replay", expected_video_sha256=video_hash)
    assert m["source_video_sha256"] == video_hash
    assert m["frames"] == frames
    assert frames == before
    with pytest.raises(ValueError, match="source video mismatch"):
        load_replay(tmp_path / "replay", expected_video_sha256="c" * 64)


@pytest.mark.parametrize("mode", ["different_frames", "different_declared", "invalid_hash"])
def test_source_conflicts_rejected_before_writing(tmp_path, mode):
    geom, tracks, frames, cfg = fixture()
    for f in frames: f["video_sha256"] = "a" * 64
    declared = None
    if mode == "different_frames": frames[-1]["video_sha256"] = "b" * 64
    elif mode == "different_declared": declared = "b" * 64
    else: frames[0]["video_sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="source video"):
        save_replay(tmp_path / "replay", geom, tracks, frames, cfg, {}, [], "production_in_memory",
                    source_video_sha256=declared)
    assert not (tmp_path / "replay").exists()


def test_legacy_unknown_source_readable_but_not_matchable(tmp_path):
    geom, tracks, frames, cfg = fixture()
    out = tmp_path / "geometry_replay_v1"
    save_replay(out, geom, tracks, frames, cfg, {}, [], "legacy_lossy_cache")
    # Simulate an already-written archive without the additive provenance field.
    path = out / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    m.pop("source_video_sha256")
    path.write_text(json.dumps(m), encoding="utf-8")
    _, _, loaded = load_replay(out)
    assert loaded["source_video_sha256"] == "unknown"
    assert loaded["historical_exactness"] == "unavailable"
    assert all("source_pts_s" not in f for f in loaded["frames"])
    with pytest.raises(ValueError, match="source video mismatch or unavailable"):
        load_replay(out, expected_video_sha256="a" * 64)
    with pytest.raises(ValueError, match="must be known"):
        load_replay(out, expected_video_sha256="unknown")


def test_loader_rejects_inconsistent_manifest_source(tmp_path):
    geom, tracks, frames, cfg = fixture()
    frames[0]["video_sha256"] = "a" * 64
    out = tmp_path / "replay"
    save_replay(out, geom, tracks, frames, cfg, {}, [], "production_in_memory")
    path = out / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    m["source_video_sha256"] = "b" * 64
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="source video mismatch"):
        load_replay(out)


def test_legacy_record_directory_not_overwritten(tmp_path, monkeypatch):
    from gwm.perception import overlay
    geom, tracks, frames, cfg = fixture()
    cfg["perception"]["geometry_replay"] = True
    legacy = tmp_path / "geometry_replay_v1"
    legacy.mkdir()
    marker = legacy / "marker.txt"
    marker.write_text("keep")
    monkeypatch.setattr(overlay, "render_overlay", lambda *a, **kw: None)
    with pytest.raises(FileExistsError):
        build_evidence("synthetic", frames, geom, tracks, cfg, tmp_path, [], [])
    assert marker.read_text() == "keep"
    assert not (tmp_path / "geometry_replay").exists()


@pytest.mark.parametrize("legacy_rebuild", [False, True])
def test_production_and_legacy_entrypoints_bind_video_source(tmp_path, monkeypatch, legacy_rebuild):
    from PIL import Image
    from gwm.perception import run, overlay
    from gwm.perception.provenance import file_sha256
    geom, tracks, frames, cfg = fixture()
    # Synthetic providers/extraction: this verifies entry-point wiring, not decoding.
    video = tmp_path / "synthetic-source.bin"
    video.write_bytes(b"synthetic source identity only, not an encoded video")
    video_hash = file_sha256(video)
    for f in frames:
        path = tmp_path / f"frame_{f['index']}.png"
        Image.new("RGB", (160, 160), "red").save(path)
        f.update(file=str(path), video_sha256=video_hash, source_pts_s=f["t"] + .467,
                 source_frame_index=14 + f["index"] * 3)
    def extract(*args, **kwargs):
        directory = args[1]
        directory.mkdir(parents=True)
        (directory / "frames.json").write_text(json.dumps({"frames": frames}))
        return frames
    monkeypatch.setattr(run, "extract_frames", extract)
    monkeypatch.setattr(run, "prepare_geometry_frames", lambda *a: [f["file"] for f in frames])
    monkeypatch.setattr(run, "make_geometry_backend", lambda *a: SimpleNamespace(estimate=lambda *a: geom))
    monkeypatch.setattr(run, "make_segmentation_backend", lambda *a: SimpleNamespace(segment=lambda *a: tracks))
    monkeypatch.setattr(overlay, "render_overlay", lambda *a, **kw: None)
    cfg["perception"]["geometry_replay"] = not legacy_rebuild
    out = tmp_path / "run"
    ev = run.run_perception(str(video), "synthetic", out, cfg, phrases=["box"])
    if legacy_rebuild:
        cfg["perception"]["geometry_replay"] = True
        ev = run.rebuild_evidence(out, cfg)
    _, _, manifest = load_replay(out / "geometry_replay", expected_video_sha256=video_hash)
    assert manifest["source_video_sha256"] == ev["meta"]["source_video_sha256"] == video_hash
    assert manifest["frames"][0]["source_pts_s"] == .467
    assert manifest["input_source"] == ("legacy_lossy_cache" if legacy_rebuild else "production_in_memory")
    if legacy_rebuild: assert manifest["historical_exactness"] == "unavailable"
