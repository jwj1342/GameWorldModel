"""占画面太大的检测按结构件处理。

实测一条合成片段上检出了一个占画面 67% 的「slab」和一个占 61% 的「ground」，
而真值里最大的物体只占 2.1%。那种东西如果进了 objects 列表，既压低物体精确率，
又会变成一块和整个场景一样大、还能用鼠标拖动的板子。
"""
import pytest

from gwm.synthesis.direct import evidence_to_program


def obj(oid, cls, area, motion="trajectory"):
    # 字段照着 tests/test_pipeline_logic.py 的夹具来：缺省的字段由旧版证据适配器补齐，
    # 显式写一个空的 attribute_confidence 反而过不了 schema
    box = {"center": [0.0, 0.5, 0.0], "quat": [0, 0, 0, 1], "size": [1.0, 1.0, 1.0], "conf": 0.9}
    return {"id": oid, "class_guess": cls, "confidence": 0.9, "is_dynamic": motion != "static",
            "obb": [{**box, "t": float(t)} for t in (0.0, 1.0, 2.0)],
            "motion_guess": {"type": motion, "conf": 0.9, "axis": [1, 0, 0], "speed": 0.5},
            "contacts": [], "mask_area_frac": area}


def evidence(objects):
    times = (0.0, 1.0, 2.0)
    return {"meta": {"clip": "c", "duration": 2.0, "scale": "relative"},
            "camera": {"intrinsics": {"fov_deg": 60, "width": 640, "height": 360},
                       "poses": [{"t": float(t), "pos": [0, 1.2, 5], "quat": [0, 0, 0, 1]}
                                 for t in times]},
            "static": {"planes": [{"kind": "ground", "conf": 0.8, "center_hint": [0, 0, 0],
                                   "extent_hint": [10, 0.2, 10]}]},
            "objects": objects, "keyframes": []}


def ids(nodes):
    return {n["id"] for n in nodes}


def test_占屏过大的检测不进物体列表():
    program = evidence_to_program(evidence([obj("slab_1", "slab", 0.67)]), "c", {})
    assert "slab_1" not in ids(program["objects"])
    assert "slab_1" in ids(program["static"])


def test_正常大小的物体照常进物体列表():
    program = evidence_to_program(evidence([obj("crate_1", "crate", 0.03)]), "c", {})
    assert "crate_1" in ids(program["objects"])


@pytest.mark.parametrize("area,expect_object", [(0.34, True), (0.36, False)])
def test_阈值可配且边界清楚(area, expect_object):
    cfg = {"synthesis": {"max_object_area_frac": 0.35}}
    program = evidence_to_program(evidence([obj("thing_1", "crate", area)]), "c", cfg)
    assert ("thing_1" in ids(program["objects"])) is expect_object


def test_按面积判成结构件时注明理由():
    """不能悄悄降级：报告里要看得出这个节点为什么跑到 static 去了。"""
    program = evidence_to_program(evidence([obj("slab_1", "slab", 0.67)]), "c", {})
    node = next(n for n in program["static"] if n["id"] == "slab_1")
    assert "67%" in node["notes"] and "area" in node["notes"]
