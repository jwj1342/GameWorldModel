"""Synthetic state records only: no browser, perception or model calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gwm.feedback.gt_metrics import evaluate, match_objects, tracks


def program(ids, motion="static"):
    return {"objects": [{"id": key, "motion": {"type": motion}} for key in ids]}


def states(ids, times=(0, 1, 2, 3, 4)):
    return [{"t": float(t), "objects": [
        {"id": key, "name": key, "kind": "object", "pos": [i * 2.0, 0, 0]}
        for i, key in enumerate(ids)]} for t in times]


def test_baseline_missing_extra_and_motion_labels():
    reference = states(["a", "b"])
    perfect = evaluate(program(["a", "b"]), program(["a", "b"]), reference, reference)
    assert perfect["full"]["trajectory"]["max_m"] == 0
    missing = evaluate(program(["a", "b"]), program(["a"]), reference, states(["a"]))
    assert missing["full"]["objects"]["recall"] == 0.5
    extra = evaluate(program(["a", "b"]), program(["a", "b", "c"]), reference, states(["a", "b", "c"]))
    assert extra["full"]["objects"]["precision"] == pytest.approx(2 / 3)
    wrong = evaluate(program(["a"], "spin"), program(["a"]), states(["a"]), states(["a"]))
    assert wrong["full"]["motion_type"]["accuracy"] == 0


@pytest.mark.parametrize("end", [8.0, 8 - 1 / 30, 7.9])
def test_small_endpoint_shortfall_preserves_measured_error(end):
    # 两个物体，只挪其中一个。全都挪那是规范自由度，align_gauge 会对齐掉、量不到东西；
    # 只挪一个，剩下的才是相对结构误差，这一项查的是覆盖不足不该把它抹掉。
    reference = states(["a", "b"], np.linspace(0, 8, 241))
    prediction = states(["a", "b"], np.linspace(0, end, 240))
    for frame in prediction:
        frame["objects"][0]["pos"][0] += 0.3
    result = evaluate(program(["a", "b"]), program(["a", "b"]), reference, prediction)
    for name, start in (("full", 0), ("holdout", 6.4)):
        metric = result[name]["trajectory"]
        # 对齐把这 0.3 的相对偏移平摊到两个物体上，各剩一半
        assert metric["median_m"] == pytest.approx(0.15)
        for key in ("a", "b"):
            assert metric["coverage"][key] == pytest.approx((end - start) / (8 - start))
            assert metric["evaluated_window_s"][key] == pytest.approx([start, end])
        assert metric["status"] == ("complete" if end == 8 else "partial")
        assert result[name]["window_s"] == [start, 8]


def test_coverage_threshold_and_no_extrapolation():
    reference, prediction = states(["a"], (0, 4, 8)), states(["a"], (0, 4, 7))
    for threshold, available in ((0.9, False), (0.8, True)):
        metric = evaluate(program(["a"]), program(["a"]), reference, prediction,
                          min_time_coverage=threshold)["full"]["trajectory"]
        assert metric["coverage"]["a"] == 0.875
        assert (metric["median_m"] is not None) == available
    result = evaluate(program(["a"]), program(["a"]), reference, states(["a"], (0, 1)),
                      min_time_coverage=0)
    assert result["holdout"]["trajectory"]["median_m"] is None
    assert result["holdout"]["trajectory"]["coverage"]["a"] == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True, "0.9"])
def test_invalid_coverage_configuration(threshold):
    with pytest.raises(ValueError):
        evaluate(program(["a"]), program(["a"]), states(["a"]), states(["a"]),
                 min_time_coverage=threshold)


def test_configuration_default_and_override(monkeypatch):
    import gwm.feedback.gt_metrics as metrics
    monkeypatch.setattr(metrics, "load_config", lambda: {"gt_evaluation": {"min_time_coverage": 0.8}})
    args = (program(["a"]), program(["a"]), states(["a"], (0, 4, 8)), states(["a"], (0, 4, 7)))
    assert metrics.evaluate(*args)["full"]["trajectory"]["median_m"] == 0
    assert metrics.evaluate(*args, min_time_coverage=0.9)["full"]["trajectory"]["median_m"] is None


def test_coverage_is_per_object_and_exact_threshold_is_accepted():
    reference = states(["a", "b"], (0, 4, 8))
    prediction = states(["a", "b"], (0, 4, 7.2, 8))
    prediction[-1]["objects"].pop(0)
    result = evaluate(program(["a", "b"]), program(["a", "b"]), reference, prediction)
    metric = result["full"]["trajectory"]
    assert metric["coverage"] == {"a": 0.9, "b": 1.0}
    assert metric["n"] == 2 and metric["status"] == "partial"
    assert result["holdout"]["trajectory"]["unavailable"] == {"a": "insufficient_time_coverage"}


@pytest.mark.parametrize("prediction_times", [(0, 1), (5, 6)])
def test_prediction_cannot_shrink_reference_or_extrapolate(prediction_times):
    reference = states(["a"])
    reference[-1]["objects"][0]["pos"] = [10, 0, 0]
    result = evaluate(program(["a"]), program(["a"]), reference, states(["a"], prediction_times))
    assert result["full"]["window_s"] == [0, 4]
    assert result["holdout"]["window_s"] == [3.2, 4]
    assert result["holdout"]["trajectory"]["max_m"] is None
    assert result["holdout"]["trajectory"]["status"] == "unavailable"


def test_identity_is_frozen_before_tail_and_future_changes_do_not_change_it():
    reference = states(["a", "b"])
    prediction = deepcopy(reference)
    prediction[-1]["objects"][0]["pos"] = [2, 0, 0]
    prediction[-1]["objects"][1]["pos"] = [0, 0, 0]
    result = evaluate(program(["a", "b"]), program(["a", "b"]), reference, prediction)
    assert result["matching"]["pairs"] == {"a": "a", "b": "b"}
    assert result["holdout"]["trajectory"]["max_m"] > 1
    prediction[-1]["objects"][0]["pos"] = [100, 0, 0]
    changed = evaluate(program(["a", "b"]), program(["a", "b"]), reference, prediction)
    assert changed["matching"] == result["matching"]


def test_instances_are_counted_and_motion_uses_parent_id():
    reference = states(["a", "b"])
    for frame in reference:
        for i, obj in enumerate(frame["objects"], 1):
            obj.update(id="block", name=f"block#{i}")
    prediction = deepcopy(reference)
    for frame in prediction:
        frame["objects"].pop()
    assert set(tracks(reference)) == {"block#1", "block#2"}
    result = evaluate(program(["block"], "spin"), program(["block"], "spin"), reference, prediction)
    assert result["full"]["objects"]["gt"] == 2
    assert result["full"]["objects"]["recall"] == 0.5
    assert result["full"]["motion_type"]["confusion"] == {"spin": {"spin": 1}}


def test_assignment_maximizes_feasible_matches():
    times = np.array([0., 1.])
    def track(pos):
        return {"t": times, "pos": np.tile(pos, (2, 1))}
    pairs = match_objects({"a": track([0, 0, 0]), "b": track([0.9, 0, 0])},
                          {"p": track([0, 0, 0]), "q": track([0, 0.9, 0])}, times, 1)
    assert {g: p for g, p, _ in pairs} == {"a": "q", "b": "p"}


def test_empty_prediction_is_zero_recall_not_execution_failure():
    result = evaluate(program(["a"]), program([]), states(["a"]), states([]))
    assert result["full"]["objects"] == {"gt": 1, "pred": 0, "matched": 0, "recall": 0, "precision": None}
    assert result["full"]["trajectory"]["median_m"] is None
    assert result["full"]["motion_type"]["accuracy"] is None
    assert evaluate(program(["a"]), program([]), states(["a"]), [])["status"] == "invalid_states"


@pytest.mark.parametrize("damage", ["duplicate_time", "duplicate_instance", "nan_position", "missing_objects",
                                    "non_object_frame", "non_object_entry"])
def test_broken_records_are_not_perfect_results(damage):
    prediction = states(["a"])
    if damage == "duplicate_time":
        prediction[1]["t"] = prediction[0]["t"]
    elif damage == "duplicate_instance":
        prediction[0]["objects"] *= 2
    elif damage == "nan_position":
        prediction[0]["objects"][0]["pos"][0] = float("nan")
    elif damage == "missing_objects":
        del prediction[0]["objects"]
    elif damage == "non_object_frame":
        prediction[0] = None
    else:
        prediction[0]["objects"][0] = None
    result = evaluate(program(["a"]), program(["a"]), states(["a"]), prediction)
    assert result["status"] == "invalid_states" and "full" not in result


@pytest.mark.parametrize("side", ["gt", "pred"])
def test_records_from_another_program_cannot_get_default_static_credit(side):
    reference, prediction = states(["a"]), states(["a"])
    mismatched = reference if side == "gt" else prediction
    for frame in mismatched:
        frame["objects"][0].update(id="unrelated", name="unrelated")
    result = evaluate(program(["a"], "spin"), program(["a"], "spin"), reference, prediction)
    assert result["status"] == "state_program_mismatch"
    assert side in result["error"] and "full" not in result


def test_reference_origin_and_late_objects_are_explicit():
    reference = states(["a"], (5, 6, 7, 8, 9))
    reference[-1]["objects"].append({"id": "late", "kind": "object", "pos": [8, 0, 0]})
    result = evaluate(program(["a", "late"]), program(["a", "late"]), reference, reference)
    assert result["full"]["window_s"] == [5, 9]
    assert result["holdout"]["window_s"] == [8.2, 9]
    assert result["full"]["trajectory"]["unavailable"]["late"] == "not_observed_in_prefix"
    assert "不宣称" in result["note"]


def test_period_error_changes_metric_without_assuming_tail_is_always_worse():
    times = np.linspace(0, 6, 121)
    def periodic(period):
        records = states(["a"], times)
        for record in records:
            record["objects"][0]["pos"][0] = float(np.sin(2 * np.pi * record["t"] / period))
        return records
    result = evaluate(program(["a"], "periodic_translate"), program(["a"], "periodic_translate"),
                      periodic(2), periodic(2.5))
    assert result["full"]["trajectory"]["max_m"] > 0.1
    assert result["holdout"]["trajectory"]["max_m"] > 0.1


def sensitivity_module():
    path = Path(__file__).resolve().parents[1] / "scripts/check_gt_metrics.py"
    spec = importlib.util.spec_from_file_location("sensitivity_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitivity_script_fails_when_all_perturbations_look_perfect(tmp_path, monkeypatch):
    module = sensitivity_module()
    # 至少三个物体：只有一个的时候，挪它就等于挪整个场景，那是规范自由度，
    # 对齐会吃掉，「单个物体平移」这一项探不到东西；删一个之后还要剩得下可配对的。
    fixture = {"objects": [{"id": "a", "pose": {"pos": [0, 0, 0]},
                            "motion": {"type": "periodic_translate", "period": 2}},
                           {"id": "b", "pose": {"pos": [3, 0, 0]}, "motion": {"type": "static"}},
                           {"id": "c", "pose": {"pos": [0, 0, 3]}, "motion": {"type": "static"}}]}
    source, output = tmp_path / "program.json", tmp_path / "checks.json"
    source.write_text(json.dumps(fixture), encoding="utf-8")
    perfect = evaluate(program(["a"]), program(["a"]), states(["a"]), states(["a"]))
    monkeypatch.setattr(module, "states_for", lambda *args: states(["a"]))
    monkeypatch.setattr(module, "evaluate", lambda *args: deepcopy(perfect))
    assert module.main(["--program", str(source), "--work", str(tmp_path), "--out", str(output)]) == 1
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows[0]["check"]["status"] == "pass"
    # 「整体平移」查的是指标**不该**动，所以指标纹丝不动的时候它理应通过。
    # 也就是说单靠这一项发现不了一个卡死的指标——那要靠基准行和另外五项。
    stuck = {row["kind"]: row["check"]["status"] for row in rows[1:]}
    assert stuck.pop("整体平移（规范自由度）") == "pass"
    assert all(status == "fail" for status in stuck.values()), stuck


def test_sensitivity_script_records_execution_failure(tmp_path, monkeypatch):
    module = sensitivity_module()
    source, output = tmp_path / "program.json", tmp_path / "checks.json"
    source.write_text(json.dumps(program(["a"])), encoding="utf-8")
    def fail(*args):
        raise subprocess.CalledProcessError(1, "synthetic recorder")
    monkeypatch.setattr(module, "states_for", fail)
    assert module.main(["--program", str(source), "--work", str(tmp_path), "--out", str(output)]) == 1
    row = json.loads(output.read_text(encoding="utf-8"))[0]
    assert row["status"] == "execution_failed" and row["check"]["status"] == "fail"


def test_sensitivity_checks_degradation_and_undefined_metrics():
    module = sensitivity_module()
    base = evaluate(program(["a"]), program(["a"]), states(["a"]), states(["a"]))
    bad = deepcopy(base)
    bad["full"]["trajectory"]["max_m"] = 0.4
    bad["holdout"]["trajectory"]["max_m"] = 0.1
    assert module.sensitivity_check(base, bad, "周期改错")["status"] == "pass"
    bad["full"]["trajectory"]["max_m"] = None
    assert module.sensitivity_check(base, bad, "周期改错")["status"] == "fail"
    assert module.perturb(program(["a"]), "周期改错") == program(["a"])


def test_sensitivity_script_success_and_inapplicable_period(tmp_path, monkeypatch):
    module = sensitivity_module()
    # 至少三个物体：只有一个的时候，挪它就等于挪整个场景，那是规范自由度，
    # 对齐会吃掉，「单个物体平移」这一项探不到东西；删一个之后还要剩得下可配对的。
    fixture = {"objects": [{"id": "a", "pose": {"pos": [0, 0, 0]},
                            "motion": {"type": "periodic_translate", "period": 2}},
                           {"id": "b", "pose": {"pos": [3, 0, 0]}, "motion": {"type": "static"}},
                           {"id": "c", "pose": {"pos": [0, 0, 3]}, "motion": {"type": "static"}}]}
    source, output = tmp_path / "program.json", tmp_path / "checks.json"
    def synthetic_records(candidate, *args):
        records = []
        for t in np.linspace(0, 4, 81):
            entries = []
            for obj in candidate["objects"]:
                pos = list(obj["pose"]["pos"])
                if obj["motion"]["type"] == "periodic_translate":
                    pos[0] += float(np.sin(2 * np.pi * t / obj["motion"]["period"]))
                entries.append({"id": obj["id"], "name": obj["id"], "kind": "object", "pos": pos})
            records.append({"t": float(t), "objects": entries})
        return records
    monkeypatch.setattr(module, "states_for", synthetic_records)
    source.write_text(json.dumps(fixture), encoding="utf-8")
    args = ["--program", str(source), "--work", str(tmp_path), "--out", str(output)]
    assert module.main(args) == 0
    assert all(row["check"]["status"] == "pass" for row in json.loads(output.read_text(encoding="utf-8")))
    fixture["objects"][0]["motion"] = {"type": "spin"}
    source.write_text(json.dumps(fixture), encoding="utf-8")
    assert module.main(args) == 1
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows[-1]["check"]["status"] == "not_applicable"
