"""场景程序的契约：模型写出来的东西必须过校验，编译出来的东西必须和内核对得上。
这两条一旦破了，整条管线会静默产出坏场景，所以值得测。"""
import copy, json
from pathlib import Path

from gwm.compiler.bundle import kernel_config
from gwm.compiler.validate import validate, validate_candidate

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((REPO / "examples/handwritten/program.json").read_text())


def test_shipped_example_is_valid():
    """示例程序是文档和调试的入口，它必须一直能过校验。"""
    r = validate(EXAMPLE)
    assert r["ok"], r["errors"]


def test_validation_catches_what_a_model_gets_wrong():
    """模型最常见的几种错：重复 id、引用不存在的东西、时间倒流、用了词表外的运动类型。"""
    p = copy.deepcopy(EXAMPLE)
    p["objects"][0]["id"] = p["static"][0]["id"]          # 和静态结构重名
    p["objects"][1]["support"] = "nope"                    # 引用不存在
    p["objects"][1]["motion"]["schedule"] = [{"t": 5, "to_deg": 10}, {"t": 1, "to_deg": 0}]
    codes = {e["code"] for e in validate(p)["errors"]}
    assert {"duplicate_id", "unknown_ref", "not_monotonic"} <= codes

    p2 = copy.deepcopy(EXAMPLE)
    p2["objects"][0]["motion"] = {"type": "teleport"}
    assert not validate(p2)["ok"]


def test_kernel_config_matches_the_scene_order():
    """物体 ID 颜色和注册表顺序是 Python 和 JS 之间的契约，编译器写的这份是唯一来源。
    顺序必须是「先静态、后物体、实例展开」，和 kernel/scene.js 的建场顺序一致。"""
    kc = kernel_config(EXAMPLE)
    ids = [r["id"] for r in kc["registry"]]
    assert ids[: len(EXAMPLE["static"])] == [s["id"] for s in EXAMPLE["static"]]
    coin = [r for r in kc["registry"] if r["id"] == "coin"]
    assert len(coin) == len(next(o for o in EXAMPLE["objects"] if o["id"] == "coin")["instances"])
    assert len({tuple(r["color"]) for r in kc["registry"]}) == len(kc["registry"])   # 颜色不能撞，否则掩码会混
    assert kc["collectible_classes"] and kc["hazard_classes"]


def test_candidate_report_is_stable_and_model_independent():
    report = validate_candidate(EXAMPLE, evidence={"ignored_by_v1": True})
    assert report["version"] == "1.0" and report["ok"]
    assert report["findings"] == report["warnings"]
    assert all({"stage", "severity", "path", "code", "message", "suggestion"} <= set(f)
               for f in report["findings"])
    json.dumps(report)


def test_geometry_rejects_zero_and_non_unit_quaternions():
    program = copy.deepcopy(EXAMPLE)
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 0]
    report = validate_candidate(program)
    assert not report["ok"]
    assert any(e["code"] == "zero_quaternion" and e["stage"] == "geometry" for e in report["errors"])
    program["objects"][0]["pose"]["quat"] = [0, 0, 0, 2]
    report = validate_candidate(program)
    assert not report["ok"] and any(e["code"] == "non_unit_quaternion" for e in report["errors"])


def test_geometry_rejects_non_finite_camera_transform():
    program = copy.deepcopy(EXAMPLE)
    program["camera"]["keyframes"][0]["pos"][0] = float("inf")
    report = validate_candidate(program)
    assert not report["ok"] and any(e["code"] == "non_finite_transform" for e in report["errors"])


def test_geometry_rejects_ground_penetration():
    program = copy.deepcopy(EXAMPLE)
    program["objects"][2]["pose"]["pos"][1] = -1.0
    report = validate_candidate(program)
    assert not report["ok"] and any(e["code"] == "ground_penetration" for e in report["errors"])


def test_geometry_checks_support_contact():
    program = copy.deepcopy(EXAMPLE)
    program["objects"][2]["support"] = "ground"
    program["objects"][2]["pose"]["pos"] = [20.0, 0.4, 20.0]
    report = validate_candidate(program)
    assert not report["ok"] and any(e["code"] == "support_no_overlap" for e in report["errors"])
