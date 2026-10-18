"""物理稳定结果的采纳规则。仿真本身在 harness 侧，这里只测「哪些该信」。"""
import pytest

from gwm.synthesis.stabilise import apply_settled


def program():
    return {"objects": [
        {"id": "cup", "class": "cup", "pose": {"pos": [0.0, 1.0, 0.0], "quat": [0, 0, 0, 1]},
         "geom": {"kind": "primitive", "shape": "box", "extent": [0.1, 0.1, 0.1]}},
        {"id": "lamp", "class": "lamp", "pose": {"pos": [1.0, 2.0, 0.0], "quat": [0, 0, 0, 1]},
         "geom": {"kind": "primitive", "shape": "box", "extent": [0.2, 0.4, 0.2]}},
    ]}


def test_落下来一点的物体采纳新位姿():
    p = program()
    report = {"settled": [{"id": "cup", "pos": [0.0, 0.75, 0.0], "quat": [0, 0.1, 0, 0.995], "moved": 0.25}]}
    rows = apply_settled(p, report)
    assert rows == [{"id": "cup", "moved_m": 0.25, "applied": True}]
    assert p["objects"][0]["pose"]["pos"] == [0.0, 0.75, 0.0]
    assert p["objects"][0]["pose"]["quat"] == [0, 0.1, 0, 0.995]


def test_掉得太远的不采纳():
    """掉很远说明它根本没有支撑，或者尺寸估错了。让它掉下去只会把看得见的错误
    变成看不见的错误。"""
    p = program()
    rows = apply_settled(p, {"settled": [{"id": "lamp", "pos": [1.0, 0.1, 0.0], "moved": 1.9}]},
                         max_shift_m=0.8)
    assert rows[0]["applied"] is False and rows[0]["reason"] == "shift_exceeds_limit"
    assert p["objects"][1]["pose"]["pos"] == [1.0, 2.0, 0.0], "不采纳就不该动"


def test_几乎没动的不记一笔():
    p = program()
    assert apply_settled(p, {"settled": [{"id": "cup", "pos": [0, 1.0, 0], "moved": 0.0001}]}) == []


def test_程序里没有的物体被忽略():
    p = program()
    assert apply_settled(p, {"settled": [{"id": "ghost", "pos": [0, 0, 0], "moved": 0.5}]}) == []


def test_空报告不炸():
    assert apply_settled(program(), {}) == [] and apply_settled(program(), None) == []
