import json, copy
from pathlib import Path
from gwm.compiler.validate import validate, validate_candidate

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

def test_candidate_report_is_stable_and_model_independent():
    r = validate_candidate(EX, evidence={"ignored_by_v1": True})
    assert r["version"] == "1.0" and r["ok"]
    assert r["findings"] == r["warnings"]
    assert all({"stage", "severity", "path", "code", "message", "suggestion"} <= set(f) for f in r["findings"])
    json.dumps(r)

def test_geometry_rejects_zero_and_non_unit_quaternions():
    p = copy.deepcopy(EX); p["objects"][0]["pose"]["quat"] = [0, 0, 0, 0]
    r = validate_candidate(p)
    assert not r["ok"] and any(e["code"] == "zero_quaternion" and e["stage"] == "geometry" for e in r["errors"])

    p["objects"][0]["pose"]["quat"] = [0, 0, 0, 2]
    r = validate_candidate(p)
    assert not r["ok"] and any(e["code"] == "non_unit_quaternion" for e in r["errors"])

def test_geometry_rejects_non_finite_camera_transform():
    p = copy.deepcopy(EX); p["camera"]["keyframes"][0]["pos"][0] = float("inf")
    r = validate_candidate(p)
    assert not r["ok"] and any(e["code"] == "non_finite_transform" for e in r["errors"])

def test_geometry_rejects_ground_penetration():
    p = copy.deepcopy(EX); p["objects"][2]["pose"]["pos"][1] = -1.0
    r = validate_candidate(p)
    assert not r["ok"] and any(e["code"] == "ground_penetration" for e in r["errors"])

def test_geometry_checks_support_contact():
    p = copy.deepcopy(EX)
    p["objects"][2]["support"] = "ground"
    p["objects"][2]["pose"]["pos"] = [20.0, 0.4, 20.0]
    r = validate_candidate(p)
    assert not r["ok"] and any(e["code"] == "support_no_overlap" for e in r["errors"])
