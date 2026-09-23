"""Small synthetic render passes; no browser, model or GPU is needed."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from PIL import Image

from gwm.compiler.ids import id_to_color, registry_order
from gwm.feedback.validate_render import validate_render_result


def _fixture(tmp_path, *, visible=None, moving=True, instances=False):
    program = {
        "static": [{"id": "base"}],
        "objects": [
            {"id": "moving", "motion": {"type": "prismatic", "axis": [1, 0, 0], "rate": 1},
             **({"instances": [{"pos": [0, 0, 0]}, {"pos": [1, 0, 0]}]} if instances else {})},
            {"id": "other", "motion": {"type": "static"}},
        ],
    }
    registry = registry_order(program)
    visible = visible or [set(entry["id"] for entry in registry)] * 3
    index = {"width": 16, "height": 12, "passes": ["rgb", "depth", "id"], "frames": []}
    for frame_no, t in enumerate((0.0, 0.5, 1.0)):
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :] * 9
        depth = np.zeros_like(rgb)
        depth[:, :, 0] = np.arange(12, dtype=np.uint8)[:, None] * 10
        ids = np.zeros_like(rgb)
        state = []
        for entry_no, entry in enumerate(registry):
            object_id = entry["id"]
            if object_id in visible[frame_no]:
                left = 1 + entry_no * 3
                ids[2:5, left:left + 2] = id_to_color(entry["idx"])
            instance = entry["instance"]
            state.append({"id": object_id, "kind": entry["kind"],
                          "name": object_id + (f"#{instance}" if instance else ""),
                          "pos": [t if moving and object_id == "moving" else 0.0, float(instance), 0.0],
                          "quat": [0, 0, 0, 1]})
        files = {}
        for pass_name, pixels in (("rgb", rgb), ("depth", depth), ("id", ids)):
            name = f"{pass_name}_{frame_no}.png"
            Image.fromarray(pixels).save(tmp_path / name)
            files[pass_name] = name
        index["frames"].append({"t": t, "files": files, "state": {"t": t, "objects": state}})
    return program, index


def _check(program, index, path, config=None):
    return validate_render_result(program, index, path, config, expected_times=[0, 0.5, 1])


def _codes(report):
    return {item["code"] for item in report["diagnostics"]}


def test_valid_render_and_instance_names(tmp_path):
    for instances in (False, True):
        program, index = _fixture(tmp_path, instances=instances)
        before_program, before_index = deepcopy(program), deepcopy(index)
        report = _check(program, index, tmp_path)
        assert report["decision"] == "proceed", report["diagnostics"]
        assert program == before_program and index == before_index
        assert report["metrics"]["objects"]["moving"]["id_visible_frames"] == [0, 1, 2]
        assert all({"stage", "severity", "code", "path", "hint"} <= item.keys()
                   for item in report["diagnostics"])


def test_missing_and_undecodable_pass(tmp_path):
    program, index = _fixture(tmp_path)
    (tmp_path / index["frames"][0]["files"]["rgb"]).unlink()
    (tmp_path / index["frames"][1]["files"]["depth"]).write_bytes(b"bad png")
    report = _check(program, index, tmp_path)
    assert {"missing_pass_file", "undecodable_pass"} <= _codes(report)
    assert report["decision"] == "block" and report["mode"] == "report"


def test_missing_id_does_not_claim_never_visible(tmp_path):
    program, index = _fixture(tmp_path)
    for frame in index["frames"]:
        (tmp_path / frame["files"]["id"]).unlink()
    report = _check(program, index, tmp_path)
    assert "object_visibility_unverifiable" in _codes(report)
    assert "object_never_visible" not in _codes(report)


def test_blank_and_mismatched_pass(tmp_path):
    program, index = _fixture(tmp_path)
    Image.new("RGB", (16, 12)).save(tmp_path / index["frames"][0]["files"]["rgb"])
    Image.new("RGB", (8, 12), "white").save(tmp_path / index["frames"][1]["files"]["depth"])
    assert {"blank_pass", "pass_size_mismatch"} <= _codes(_check(program, index, tmp_path))


def test_never_visible_and_temporary_absence(tmp_path):
    program, index = _fixture(tmp_path, visible=[{"base", "moving"}] * 3)
    report = _check(program, index, tmp_path)
    assert "object_never_visible" in _codes(report)
    assert report["metrics"]["objects"]["other"]["id_absent_frames"] == [0, 1, 2]
    assert report["decision"] == "warn"
    program, index = _fixture(tmp_path, visible=[{"base", "moving", "other"}, {"base", "other"}, {"base", "moving", "other"}])
    report = _check(program, index, tmp_path)
    assert "object_temporarily_not_visible" in _codes(report)
    assert "object_never_visible" not in _codes(report)
    assert report["metrics"]["objects"]["moving"]["id_absent_frames"] == [1]


def test_motion_and_thresholds(tmp_path):
    program, index = _fixture(tmp_path, moving=False)
    assert "dynamic_object_not_moving" in _codes(_check(program, index, tmp_path))
    program, index = _fixture(tmp_path)
    assert "dynamic_object_not_moving" in _codes(_check(program, index, tmp_path, {"min_translation_m": 2.0}))
    assert "dynamic_object_not_moving" not in _codes(_check(program, index, tmp_path, {"min_translation_m": 0.01}))
    for frame in index["frames"]:
        frame["state"]["objects"][1]["pos"] = [0, 0, 0]
    index["frames"][2]["state"]["objects"][1]["quat"] = [0, 0, 0.2, 0.979795897]
    assert "dynamic_object_not_moving" not in _codes(_check(program, index, tmp_path))
    program["objects"][0]["motion"]["trigger"] = True
    for frame in index["frames"]:
        frame["state"]["objects"][1]["quat"] = [0, 0, 0, 1]
    assert "trigger_motion_not_exercised" in _codes(_check(program, index, tmp_path))


def test_browser_events_and_missing_index_log(tmp_path):
    program, index = _fixture(tmp_path)
    index["browser_events"] = ["[error] console broke", "[pageerror] script broke", "[requestfailed] /asset.png"]
    report = _check(program, index, tmp_path)
    assert {"browser_error", "browser_pageerror", "browser_requestfailed"} <= _codes(report)
    assert report["metrics"]["browser"] == {"error": 1, "pageerror": 1, "requestfailed": 1}
    (tmp_path / "browser.log").write_text("[pageerror] boot failed\n", encoding="utf-8")
    missing = validate_render_result(program, None, tmp_path)
    assert {"missing_render_index", "browser_pageerror"} <= _codes(missing)


def test_wrong_ids_and_state_mismatch(tmp_path):
    program, index = _fixture(tmp_path)
    name = index["frames"][0]["files"]["id"]
    pixels = np.asarray(Image.open(tmp_path / name)).copy()
    pixels[2:5, 4:6] = [254, 253, 252]
    Image.fromarray(pixels).save(tmp_path / name)
    index["frames"][1]["state"]["objects"][1]["id"] = "unexpected"
    report = _check(program, index, tmp_path)
    assert {"unknown_render_id", "state_id_mismatch"} <= _codes(report)


def test_frame_count_timestamps_and_safe_paths(tmp_path):
    program, index = _fixture(tmp_path)
    index["frames"][1]["t"] = 0.0
    index["frames"][2]["files"]["rgb"] = "../outside.png"
    report = _check(program, index, tmp_path)
    assert {"non_monotonic_render_times", "timestamp_mismatch", "missing_pass_file"} <= _codes(report)
    index["frames"].pop()
    assert "frame_count_mismatch" in _codes(_check(program, index, tmp_path))
