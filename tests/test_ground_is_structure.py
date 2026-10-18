"""class 是地面的节点永远进 static[]，不进 objects[]。

为什么要单立一条，而不是靠「占画面太大」那条兜住：地面被当成可拖动物体的后果不只是
多了一个不该能拖的东西。它往往是场景里最大的那块，横在开场视角的最前面，后面的物体
一个都点不到——实测有片段十五个物体里十一个被同一块 ground 挡住。更糟的一种是它还
带着运动：相机在动，配准把相机的运动算到了地面头上，于是程序里写着一块 1.9 x 0.4 x
1.5 米的地板沿轨迹飞。这两种情况下它的掩码都没到 35% 的上限，面积那条拦不住。
"""
from gwm.synthesis.direct import evidence_to_program
from gwm.taxonomy import is_ground_class


def obj(oid, cls, area, motion="trajectory"):
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


def test_地面类别认得出来():
    assert is_ground_class("ground")
    assert is_ground_class(" Floor ")
    assert is_ground_class("terrain")
    assert not is_ground_class("groundhog")      # 不是子串匹配，免得误伤
    assert not is_ground_class("table")
    assert not is_ground_class("")


def test_地面即使被判成在动也进固定结构():
    # 掩码只占 10%，面积那条拦不住；运动假设是「有轨迹」，原来的条件也拦不住
    program = evidence_to_program(evidence([obj("ground_1", "ground", 0.10, "trajectory")]), "c", {})
    assert "ground_1" not in ids(program["objects"]), "地面不该是可拖动物体"
    assert "ground_1" in ids(program["static"])
    node = next(n for n in program["static"] if n["id"] == "ground_1")
    assert "ground-class" in str(node.get("notes", "")), "要写清楚为什么被归成结构件"


def test_名字里带地面的普通物体不受影响():
    # 只在类别**完全等于**地面词时才强制，不然「playground slide」这种会被误伤
    program = evidence_to_program(evidence([obj("slide_1", "playground slide", 0.05)]), "c", {})
    assert "slide_1" in ids(program["objects"])
