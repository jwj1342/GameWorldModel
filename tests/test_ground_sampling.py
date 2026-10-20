"""Synthetic mask roles only; no video, weights, providers or model calls."""
import copy

import numpy as np
import pytest

from gwm.perception import evidence
from gwm.perception.base import Tracks, TrackedObject
from gwm.perception.ground_sampling import GroundSamplingPolicy, mask_sha256
from gwm.perception.masks import resize_mask
from test_geometry_replay import fixture


VIDEO = "a" * 64


def setup():
    geom, tracks, frames, cfg = fixture()
    # The class is deliberately not a ground synonym: roles, not names, decide.
    obj = tracks.objects[0]
    obj.phrase = "unclassified surface"
    obj.masks = {fi: np.ones((160, 160), bool) for fi in geom.frame_indices}
    return geom, tracks, frames, cfg


def declaration(obj, frames, role="ground_support"):
    return {"object_id": obj.id, "role": role, "frame_indices": list(frames),
            "source": {"kind": "human_review", "reference": "synthetic-review-row",
                       "video_sha256": VIDEO},
            "mask_sha256": {str(fi): mask_sha256(obj.masks[fi]) for fi in frames}}


def enabled(cfg, records):
    cfg["perception"]["ground_sampling"] = {"mode": "reviewed_roles", "roles": records}


def test_default_matches_original_mask_exclusion_and_alignment():
    geom, tracks, _, cfg = fixture()
    baseline = evidence.align_world(geom, tracks, cfg, replay={})
    policy = GroundSamplingPolicy(tracks, geom.frame_indices, cfg)
    union, excluded, support = policy.masks(tracks, 0, geom.size(), resize_mask)
    expected = resize_mask(tracks.objects[0].masks[0], geom.size())
    np.testing.assert_array_equal(union, expected)
    np.testing.assert_array_equal(excluded, expected)
    assert not support.any()
    cfg["perception"].pop("ground_sampling")
    other = evidence.align_world(geom, tracks, cfg, replay={})
    np.testing.assert_array_equal(baseline[0], other[0])
    assert baseline[1:] == other[1:]
    assert "ground_sampling" not in other[2]


def test_surface_role_restores_sampling_and_preserves_object_union(tmp_path, monkeypatch):
    geom, tracks, _, cfg = setup()
    calls = []
    original = evidence.to_world
    def record_points(*args):
        result = original(*args)
        calls.append(result.copy())
        return result
    monkeypatch.setattr(evidence, "to_world", record_points)
    a = evidence.align_world(geom, tracks, cfg)
    original_object_points = calls[1::2]
    assert a[2]["n_static_points"] == 0
    assert a[2]["ground"] == "not_found"
    enabled(cfg, [declaration(tracks.objects[0], geom.frame_indices)])
    mask_before = [m.copy() for m in tracks.objects[0].masks.values()]
    cfg_before = copy.deepcopy(cfg)
    calls.clear()
    replay = {}
    _, _, info = evidence.align_world(geom, tracks, cfg, replay=replay, source_video_sha256=VIDEO)
    assert info["n_static_points"] == 1200
    assert info["n_object_points"] == a[2]["n_object_points"]
    assert info["ground"].startswith("candidate_")
    assert len(original_object_points) == len(calls[1::2])
    assert all(np.array_equal(x, y) for x, y in zip(original_object_points, calls[1::2]))
    # The production adaptive threshold, never the frozen diagnostic constant.
    points = np.concatenate(calls[::2])
    scene_scale = float(np.percentile(np.abs(points-points.mean(0)).max(1), 50))
    assert replay["ransac_threshold"] == pytest.approx(
        max(scene_scale * cfg["perception"]["ground_ransac_thresh_m"], 1e-4), rel=1e-14, abs=1e-15)
    assert info["ground_sampling"]["physical_ground_correctness"] == "unknown"
    assert cfg == cfg_before
    assert all(np.array_equal(x, y) for x, y in zip(mask_before, tracks.objects[0].masks.values()))


def test_explicit_frames_only_no_interval_interpolation():
    geom, tracks, _, cfg = setup()
    enabled(cfg, [declaration(tracks.objects[0], [1])])
    replay = {}
    evidence.align_world(geom, tracks, cfg, replay=replay, source_video_sha256=VIDEO)
    assert {p[0] for p in replay["static_point_sources"]} == {1}
    assert [f["eligible_support_pixels"] for f in replay["ground_sampling"]["frames"]] == [0, 6400, 0]


@pytest.mark.parametrize("role", ["exclude", "unknown"])
def test_unknown_or_exclude_never_promoted(role):
    geom, tracks, _, cfg = setup()
    enabled(cfg, [declaration(tracks.objects[0], geom.frame_indices, role)])
    _, _, info = evidence.align_world(geom, tracks, cfg, source_video_sha256=VIDEO)
    assert info["ground"] == "not_found"
    assert any(d["code"] == "ground_sampling_no_support" for d in info["ground_sampling"]["diagnostics"])


def test_exclusion_wins_overlap_independent_of_object_order():
    geom, tracks, _, cfg = setup()
    foreground = np.zeros((160, 160), bool)
    foreground[40:100, 40:100] = True
    tracks.objects.append(TrackedObject("other", "unclassified foreground", .1,
                                       {0: foreground}))
    enabled(cfg, [declaration(tracks.objects[0], [0])])
    policy = GroundSamplingPolicy(tracks, geom.frame_indices, cfg, source_video_sha256=VIDEO)
    a = policy.masks(tracks, 0, geom.size(), resize_mask)
    b = policy.masks(Tracks(list(reversed(tracks.objects)), tracks.frame_size), 0, geom.size(), resize_mask)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    union, excluded, support = a
    assert union.all() and support.all()
    np.testing.assert_array_equal(excluded, resize_mask(foreground, geom.size()))
    np.testing.assert_array_equal(support & ~excluded, ~resize_mask(foreground, geom.size()))


@pytest.mark.parametrize("filter_kind", ["dynamic", "invalid_depth", "empty_support"])
def test_filtered_or_empty_support_remains_unknown(filter_kind):
    geom, tracks, _, cfg = setup()
    if filter_kind == "dynamic": geom.dynamic_mask = [np.ones_like(d, bool) for d in geom.depth]
    elif filter_kind == "invalid_depth": geom.depth = [np.zeros_like(d) for d in geom.depth]
    else: tracks.objects[0].masks = {fi: np.zeros((160, 160), bool) for fi in geom.frame_indices}
    enabled(cfg, [declaration(tracks.objects[0], geom.frame_indices)])
    replay = {}
    T, scale, info = evidence.align_world(geom, tracks, cfg, replay=replay, source_video_sha256=VIDEO)
    assert info["ground"] == "not_found" and replay["selected"] is None
    np.testing.assert_array_equal(T, np.eye(4))
    assert scale == 1.
    assert all(f["eligible_support_pixels"] == 0 for f in info["ground_sampling"]["frames"])


def test_confidence_filter_still_applies_to_support():
    geom, tracks, _, cfg = setup()
    geom.depth_conf = [np.broadcast_to(np.arange(80, dtype=float), (80, 80)).copy() for _ in geom.depth]
    enabled(cfg, [declaration(tracks.objects[0], geom.frame_indices)])
    replay = {}
    evidence.align_world(geom, tracks, cfg, replay=replay, source_video_sha256=VIDEO)
    threshold = np.percentile(geom.depth_conf[0], 30)
    assert all(geom.depth_conf[fi][y, x] >= threshold for fi, y, x in replay["static_point_sources"])


def test_missing_mask_has_warning_not_interpolated_support():
    geom, tracks, _, cfg = setup()
    record = declaration(tracks.objects[0], [0])
    tracks.objects[0].masks.pop(0)
    enabled(cfg, [record])
    _, _, info = evidence.align_world(geom, tracks, cfg, source_video_sha256=VIDEO)
    assert info["ground"] == "not_found"
    assert any(d["code"] == "ground_support_missing_mask" for d in info["ground_sampling"]["diagnostics"])


@pytest.mark.parametrize("kind", ["human_review", "assistant_review", "provider"])
def test_source_actor_kind_is_retained_without_claiming_truth(kind):
    geom, tracks, _, cfg = setup()
    record = declaration(tracks.objects[0], [0])
    record["source"]["kind"] = kind
    enabled(cfg, [record])
    policy = GroundSamplingPolicy(tracks, geom.frame_indices, cfg, source_video_sha256=VIDEO)
    assert policy.audit["declarations"][0]["source"]["kind"] == kind
    assert policy.audit["physical_ground_correctness"] == "unknown"


@pytest.mark.parametrize("change", ["video", "missing_video", "reference", "source_kind", "mask_hash",
                                     "object", "frame", "duplicate_frame", "bool_frame", "role", "conflict"])
def test_invalid_source_or_role_rejected(change):
    geom, tracks, _, cfg = setup()
    record = declaration(tracks.objects[0], [0])
    records = [record]
    video_hash = VIDEO
    if change == "video": record["source"]["video_sha256"] = "b" * 64
    elif change == "missing_video": video_hash = None
    elif change == "reference": record["source"].pop("reference")
    elif change == "source_kind": record["source"]["kind"] = "guessed"
    elif change == "mask_hash": tracks.objects[0].masks[0][0, 0] = False
    elif change == "object": record["object_id"] = "missing"
    elif change == "frame": record["frame_indices"] = [999]
    elif change == "duplicate_frame": record["frame_indices"] = [0, 0]
    elif change == "bool_frame": record["frame_indices"] = [True]
    elif change == "role": record["role"] = "probably_ground"
    else: records.append(copy.deepcopy(record))
    enabled(cfg, records)
    with pytest.raises(ValueError):
        evidence.align_world(geom, tracks, cfg, source_video_sha256=video_hash)


def test_build_and_replay_preserve_source_bound_roles(tmp_path, monkeypatch):
    from gwm.perception import overlay
    from gwm.perception.geometry_replay import load_replay
    from gwm.perception.contract import validate_evidence
    geom, tracks, frames, cfg = setup()
    for f in frames: f["video_sha256"] = VIDEO
    enabled(cfg, [declaration(tracks.objects[0], geom.frame_indices)])
    cfg["perception"]["geometry_replay"] = True
    monkeypatch.setattr(overlay, "render_overlay", lambda *a, **kw: None)
    ev = evidence.build_evidence("synthetic", frames, geom, tracks, cfg, tmp_path, [], [])
    assert validate_evidence(ev)["ok"]
    g, tr, manifest = load_replay(tmp_path / "geometry_replay", expected_video_sha256=VIDEO)
    replay = {}
    evidence.align_world(g, tr, manifest["config"], replay=replay, source_video_sha256=VIDEO)
    assert replay == manifest["alignment"]
    assert manifest["config"]["perception"]["ground_sampling"] == cfg["perception"]["ground_sampling"]
