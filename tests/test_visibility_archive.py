"""Small synthetic fixtures: archive provenance, cache integrity, merge wiring."""
import copy
import hashlib
import importlib
import json

import numpy as np
import pytest
from PIL import Image

from gwm.config import REPO, load_config
from gwm.perception.base import Geometry, Tracks, TrackedObject
from gwm.perception.provenance import file_sha256
from gwm.perception.run import save_visibility_references, rebuild_evidence, save_geometry
from gwm.perception.visibility import checked_visibility


def tracks_fixture(tmp_path, count=3):
    source = tmp_path / "original.png"
    Image.new("RGB", (40, 40), (1, 2, 3)).save(source)
    obj = TrackedObject("target", "wall", 1)
    for i in range(count):
        obj.visibility_measurements[i] = {
            "object_id": "target", "registry_index": 1, "frame_index": i,
            "image_size": [40, 40], "program_sha256": "a" * 64,
            "requested_t": float(i), "actual_t": float(i), "fraction": .1,
            "method": "isolated_instance_id", "same_state_verified": True,
            "reference_path": str(source), "reference_sha256": file_sha256(source),
        }
    return Tracks([obj], (40, 40), backend="gt"), source


def test_one_archive_retains_original_png_bytes_and_member_identity(tmp_path):
    tracks, source = tracks_fixture(tmp_path)
    out = tmp_path / "cache"
    save_visibility_references(tracks, out)
    assert sorted(p.name for p in out.iterdir()) == ["visibility_refs.npz"]
    archive = out / "visibility_refs.npz"
    records = tracks.objects[0].visibility_measurements
    with np.load(archive, allow_pickle=False) as data:
        assert len(data.files) == 3
        for i, record in records.items():
            assert record["reference"] == f"visibility_refs.npz#entry_1_frame_{i}"
            assert data[f"entry_1_frame_{i}"].tobytes() == source.read_bytes()
            assert record["reference_sha256"] == file_sha256(source)
            assert record["reference_archive_sha256"] == file_sha256(archive)
    before = copy.deepcopy(records)
    digest = file_sha256(archive)
    save_visibility_references(tracks, out)
    assert records == before
    assert file_sha256(archive) == digest


@pytest.mark.parametrize("fault", ["missing", "corrupt", "member_missing", "member_changed",
                                   "wrong_member", "wrong_hash", "escape", "wrong_object"])
def test_cache_integrity_fault_returns_unknown_without_rewriting(tmp_path, fault):
    tracks, _ = tracks_fixture(tmp_path, 1)
    out = tmp_path / "cache"
    save_visibility_references(tracks, out)
    path = out / "visibility_refs.npz"
    record = tracks.objects[0].visibility_measurements[0]
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"invalid zip")
    elif fault in ("member_missing", "member_changed"):
        key = "other" if fault == "member_missing" else "entry_1_frame_0"
        np.savez_compressed(path, **{key: np.array([1, 2, 3], dtype=np.uint8)})
        record["reference_archive_sha256"] = file_sha256(path)
    elif fault == "wrong_member":
        record["reference"] = "visibility_refs.npz#entry_2_frame_0"
    elif fault == "wrong_hash":
        record["reference_archive_sha256"] = "b" * 64
    elif fault == "escape":
        record["reference"] = "../original.png"
    else:
        record["object_id"] = "other"
    before = path.read_bytes() if path.exists() else None
    save_visibility_references(tracks, out)
    assert record["fraction"] == "unknown"
    assert record["reason"] == "cached_reference_changed_or_missing"
    assert (path.read_bytes() if path.exists() else None) == before


def test_duplicate_identity_rejected(tmp_path):
    tracks, _ = tracks_fixture(tmp_path, 1)
    duplicate = copy.deepcopy(tracks.objects[0])
    duplicate.id = "another"
    duplicate.visibility_measurements[0]["object_id"] = duplicate.id
    tracks.objects.append(duplicate)
    with pytest.raises(ValueError, match="duplicate visibility reference identity"):
        save_visibility_references(tracks, tmp_path / "cache")
    assert not (tmp_path / "cache/visibility_refs.npz").exists()


@pytest.mark.parametrize("scalar", [np.float32, np.float64, np.int32, np.int64])
def test_numpy_real_scalars_accepted(scalar):
    record = {"object_id": "x", "frame_index": 0, "image_size": [4, 4],
              "program_sha256": "a", "same_state_verified": True,
              "requested_t": scalar(0), "actual_t": scalar(0),
              "fraction": scalar(1), "method": "isolated_instance_id"}
    assert checked_visibility(record, "x", 0, scalar(0), (4, 4), "a", scalar(0)) == (1., "")


@pytest.mark.parametrize("value", [True, np.bool_(True), np.nan, np.inf, "0.5", 1j])
def test_invalid_real_scalars_rejected(value):
    record = {"object_id": "x", "frame_index": 0, "image_size": [4, 4],
              "program_sha256": "a", "same_state_verified": True,
              "requested_t": 0, "actual_t": 0, "fraction": value,
              "method": "isolated_instance_id"}
    assert checked_visibility(record, "x", 0, 0, (4, 4), "a")[1] == "invalid_fraction"
    with pytest.raises(ValueError):
        checked_visibility(record, "x", 0, 0, (4, 4), "a", value)
    record.update(fraction=.5, actual_t=value)
    assert checked_visibility(record, "x", 0, 0, (4, 4), "a")[1] == "time_mismatch"


def test_default_yaml_has_no_duplicate_keys_and_preserves_both_branches():
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            assert key not in result, f"duplicate YAML key: {key}"
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    cfg = yaml.load((REPO / "configs/default.yaml").read_text(encoding="utf-8"), Loader=UniqueLoader)
    pc = cfg["perception"]
    assert pc["visibility_time_tolerance_s"] == .0001
    assert pc["motion"]["use_registration"] is True
    assert pc["motion"]["registration_smooth_window"] == 15
    assert pc["motion"]["infer_articulation"] is False
    assert pc["motion"]["use_static_detector"] is False


@pytest.mark.parametrize("cache_state", ["valid", "tampered"])
@pytest.mark.parametrize("rotation_gate", [None, .05])
def test_archived_cache_rebuild_preserves_registration_and_occlusion(tmp_path, monkeypatch, cache_state, rotation_gate):
    evmod = importlib.import_module("gwm.perception.evidence")
    overlay = importlib.import_module("gwm.perception.overlay")
    monkeypatch.setattr(evmod, "align_world", lambda *args: (np.eye(4), 1, {"ground": "not_found"}))
    monkeypatch.setattr(overlay, "render_overlay", lambda *args: None)
    tracks, source = tracks_fixture(tmp_path, 6)
    save_visibility_references(tracks, tmp_path)
    archive = tmp_path / "visibility_refs.npz"
    source.unlink()  # A new process must not need the temporary GT PNGs.
    frames = [{"index": i, "t": float(i), "file": "missing.png"} for i in range(6)]
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/frames.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
    (tmp_path / "keyframes.json").write_text("[]", encoding="utf-8")
    K = np.array([[40, 0, 20], [0, 40, 20], [0, 0, 1]], float)
    geom = Geometry(list(range(6)), [K]*6, [np.eye(4)]*6, [np.full((40, 40), 2.)]*6,
                    backend="gt", scale="metric", source_program_sha256="a"*64)
    save_geometry(geom, tmp_path / "geometry.npz")
    np.savez_compressed(tmp_path / "masks.npz", **{f"target__{i}": np.ones((40, 40), bool) for i in range(6)})
    (tmp_path / "masks_index.json").write_text(json.dumps({"objects": {
        "target": {"phrase": "wall", "score": 1,
                   "visibility_measurements": tracks.objects[0].visibility_measurements}},
        "frame_size": [40, 40], "backend": "gt"}), encoding="utf-8")
    if cache_state == "tampered":
        archive.write_bytes(b"changed")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    registered = np.array([[float(i), 0, 0] for i in range(6)])
    registration_calls = []
    def track(clouds, *, rotation_min_gain=None):
        registration_calls.append(rotation_min_gain)
        return {"ok": True, "positions": registered.tolist(),
                "orientations": [np.eye(3).tolist()]*6, "residuals": [.01]*6}
    monkeypatch.setattr(evmod, "track_rigid", track)
    calls = []
    original = evmod.estimate_motion

    def inspect_motion(ts, positions, quaternions, cfg, **kwargs):
        calls.append((positions.copy(), quaternions.copy(), kwargs))
        return original(ts, positions, quaternions, cfg, **kwargs)

    monkeypatch.setattr(evmod, "estimate_motion", inspect_motion)
    cfg = load_config()
    cfg["perception"]["ground_contact_centers"] = False
    cfg["perception"]["motion"]["registration_smooth_window"] = 1
    cfg["perception"]["motion"]["rotation_min_gain"] = rotation_gate
    rebuilt = rebuild_evidence(tmp_path, cfg)
    assert registration_calls == [rotation_gate]
    assert len(calls) == 1
    positions, quaternions, kwargs = calls[0]
    np.testing.assert_allclose(positions - positions[0], registered)
    assert quaternions.shape == (6, 4)
    assert rebuilt["objects"][0]["motion_guess"]["position_source"] == "cross_frame_registration"
    assert "dropped_short_tracks" in rebuilt["meta"]
    if cache_state == "valid":
        np.testing.assert_allclose(kwargs["visible_fractions"], [.1]*6)
        assert "occlusion" in rebuilt["objects"][0]["motion_guess"]["notes"]
    else:
        assert kwargs["visible_fractions"] is None
        assert all(o["visible_fraction"] == "unknown" for o in rebuilt["objects"][0]["observations"])
    assert file_sha256(archive) == digest
