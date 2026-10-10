"""Synthetic Programs; no model, perception provider or browser."""
import copy
import json
from pathlib import Path
import pytest
from gwm.compiler.validate import validate

PROGRAM = json.loads((Path(__file__).resolve().parents[1] / "examples/collision_chain/program.json").read_text(encoding="utf-8"))

def test_dynamic_example_remains_valid():
    assert validate(PROGRAM)["ok"]

@pytest.mark.parametrize("field,value", [
    ("linear_velocity", [float("nan"), 0, 0]),
    ("linear_velocity", [0, float("inf"), 0]),
    ("mass", float("inf")), ("mass", float("nan")),
    ("restitution", float("inf")), ("restitution", float("nan")),
])
def test_nonfinite_dynamic_parameters_rejected(field, value):
    program = copy.deepcopy(PROGRAM)
    program["objects"][0]["motion"][field] = value
    result = validate(program)
    assert not result["ok"]
    assert any(f["path"].endswith("/" + field) for f in result["errors"])

@pytest.mark.parametrize("field,value", [("mass", 0), ("mass", -1), ("restitution", -0.1), ("restitution", 1.1)])
def test_dynamic_parameter_ranges_rejected(field, value):
    program = copy.deepcopy(PROGRAM)
    program["objects"][0]["motion"][field] = value
    assert not validate(program)["ok"]

@pytest.mark.parametrize("trigger", [None, False])
def test_untriggered_dynamic_motion_allowed(trigger):
    program = copy.deepcopy(PROGRAM)
    if trigger is not None:
        program["objects"][0]["motion"]["trigger"] = trigger
    assert validate(program)["ok"]

def test_dynamic_trigger_rejected_explicitly():
    program = copy.deepcopy(PROGRAM)
    program["objects"][0]["motion"]["trigger"] = True
    report = validate(program)
    assert not report["ok"]
    assert any(f["code"] == "unsupported_dynamic_trigger" and f["path"].endswith("/trigger") for f in report["errors"])

def test_event_cannot_silently_trigger_dynamic_target():
    program = copy.deepcopy(PROGRAM)
    program["objects"][1]["events"] = [{"type": "trigger_on_enter", "target": program["objects"][0]["id"]}]
    report = validate(program)
    assert not report["ok"]
    assert any(f["code"] == "unsupported_dynamic_trigger" and f["path"].endswith("/events/0/target") for f in report["errors"])

def test_scripted_trigger_remains_supported():
    program = copy.deepcopy(PROGRAM)
    program["objects"][0]["motion"] = {"type": "prismatic", "axis": [1, 0, 0], "rate": 1, "trigger": True}
    assert validate(program)["ok"]
