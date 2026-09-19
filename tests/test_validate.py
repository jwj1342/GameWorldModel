import json, copy
from pathlib import Path
from gwm.compiler.validate import validate

EX = json.loads((Path(__file__).resolve().parents[1] / "examples/handwritten/program.json").read_text())

def test_example_valid():
    r = validate(EX); assert r["ok"], r["errors"]

def test_duplicate_id():
    p = copy.deepcopy(EX); p["objects"][0]["id"] = p["static"][0]["id"]
    r = validate(p); assert not r["ok"] and any(e["code"] == "duplicate_id" for e in r["errors"])

def test_unknown_ref_and_schedule():
    p = copy.deepcopy(EX); p["objects"][0]["support"] = "nope"; p["objects"][1]["motion"]["schedule"] = [{"t": 5, "to_deg": 10}, {"t": 1, "to_deg": 0}]
    r = validate(p); codes = {e["code"] for e in r["errors"]}; assert {"unknown_ref", "not_monotonic"} <= codes

def test_schema_rejects_unknown_motion():
    p = copy.deepcopy(EX); p["objects"][0]["motion"] = {"type": "teleport"}
    r = validate(p); assert not r["ok"] and r["errors"][0]["code"] == "schema"
