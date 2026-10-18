"""取景约定：整体相似缩放。

链路把相对尺度的重建标定到米，靠的是「相机离地 1.2 米」这个假设（手持手机站着拍）。
而生成器造的是平台跳跃关卡——走廊 4 到 7 米宽、18 到 30 米长，相机高度是场地尺度的
函数，实测中位数 7.385 米。两者差六到十二倍，后果见 #68。

缩放必须是**相似变换**：场景内部的相对结构一点不能变，变的只是它和相机高度之间的比例。
所以带长度量纲的字段全都要跟着走，角度、时间、质量这些不能动。
"""
import math

import pytest

from gwm.dataset.framing import scale_program


def program():
    return {
        "meta": {"clip": "c"},
        "camera": {"intrinsics": {"fov_deg": 60.0, "far": 100.0},
                   "keyframes": [{"t": 0.0, "pos": [0.0, 7.4, 10.0], "quat": [0, 0, 0, 1]}]},
        "static": [{"id": "ground", "class": "ground",
                    "geom": {"kind": "primitive", "shape": "box", "extent": [20.0, 0.4, 20.0]},
                    "pose": {"pos": [0.0, -0.2, 0.0]}}],
        "objects": [{"id": "crate", "class": "crate",
                     "geom": {"kind": "primitive", "shape": "box", "extent": [1.0, 2.0, 3.0]},
                     "pose": {"pos": [4.0, 1.0, -2.0], "quat": [0, 0, 0, 1]},
                     "motion": {"type": "dynamic", "mass": 5.0, "linear_velocity": [1.0, 0.0, 0.0],
                                "restitution": 0.4},
                     "interaction": {"grab": "slide", "axis": [1, 0, 0], "limit": [-0.4, 0.4]}}],
        "binding": {"template": "platformer_3p",
                    "slots": {"player_spawn": [1.0, 1.2, 1.0],
                              "goal_volume": {"pos": [5.0, 1.0, 5.0], "extent": [1.5, 2.0, 1.5]}}},
    }


def test_长度量纲的字段都跟着缩():
    out = scale_program(program(), 0.15)
    obj = out["objects"][0]
    assert obj["geom"]["extent"] == pytest.approx([0.15, 0.3, 0.45])
    assert obj["pose"]["pos"] == pytest.approx([0.6, 0.15, -0.3])
    assert obj["motion"]["linear_velocity"] == pytest.approx([0.15, 0.0, 0.0])
    assert out["static"][0]["geom"]["extent"] == pytest.approx([3.0, 0.06, 3.0])
    assert out["camera"]["keyframes"][0]["pos"] == pytest.approx([0.0, 1.11, 1.5])
    assert out["binding"]["slots"]["player_spawn"] == pytest.approx([0.15, 0.18, 0.15])
    assert out["binding"]["slots"]["goal_volume"]["extent"] == pytest.approx([0.225, 0.3, 0.225])


def test_不带长度量纲的字段一个都不许动():
    out = scale_program(program(), 0.15)
    obj = out["objects"][0]
    assert obj["motion"]["mass"] == 5.0, "质量不是长度"
    assert obj["motion"]["restitution"] == 0.4
    assert obj["pose"]["quat"] == [0, 0, 0, 1]
    assert obj["interaction"]["axis"] == [1, 0, 0], "轴是方向，缩放会把它变成非单位向量"
    assert out["camera"]["intrinsics"]["fov_deg"] == 60.0, "视场角是角度"


def test_滑轨行程缩而铰链角度不缩():
    out = scale_program(program(), 0.15)
    assert out["objects"][0]["interaction"]["limit"] == pytest.approx([-0.06, 0.06])
    hinge = program()
    hinge["objects"][0]["interaction"] = {"grab": "hinge", "axis": [0, 1, 0], "limit": [0.0, 1.57]}
    out2 = scale_program(hinge, 0.15)
    assert out2["objects"][0]["interaction"]["limit"] == [0.0, 1.57], "开合角是角度，不能缩"


def test_相机高度缩到手持的范围():
    # 这是这件事的全部意义：7.4 米的俯看变成一米一的手持
    out = scale_program(program(), 0.15)
    assert math.isclose(out["camera"]["keyframes"][0]["pos"][1], 1.11, rel_tol=1e-6)


def test_不缩放时原样返回且不共享引用():
    src = program()
    out = scale_program(src, 1.0)
    assert out == src
    out["objects"][0]["pose"]["pos"][0] = 99.0
    assert src["objects"][0]["pose"]["pos"][0] == 4.0, "返回的必须是副本"
