"""结构件的位置取首帧包围盒中心，不取逐帧中位。

逐帧包围盒中心在整段视频里**单调漂移**，不是在真值附近抖。拿真值结构件量
（位置恒定，所以靶子不动），逐帧误差中位沿着片段走是 0.366 / 0.419 / 0.511 /
0.578 米，一路走高。取中位等于把估计往漂移里拖。

四种估计量在 15 个真值结构件上的对照：

    首帧          离真值中位 0.277 m，容差 0.45 内 0.867
    逐帧中位      离真值中位 0.458 m，容差 0.45 内 0.467
    best_frame    离真值中位 0.427 m，容差 0.45 内 0.667
    最高 conf 帧  离真值中位 0.401 m，容差 0.45 内 0.600

尺寸仍然取逐帧中位——那是另一个量，而且量过中位更好（#64）。
"""
import numpy as np
import pytest

from gwm.synthesis.direct import evidence_to_program


def evidence(centers, sizes=None, cls="rail"):
    """沿用 test_ground_is_structure 里那份已知合法的证据骨架，只把逐帧包围盒换掉。"""
    n = len(centers)
    sizes = sizes or [[2.0, 0.3, 0.4]] * n
    times = [i * 0.25 for i in range(n)]
    obj = {"id": "rail_1", "class_guess": cls, "confidence": 0.9, "is_dynamic": False,
           "obb": [{"center": list(c), "quat": [0, 0, 0, 1], "size": list(sz),
                    "conf": 0.9, "t": float(t)}
                   for c, sz, t in zip(centers, sizes, times)],
           "motion_guess": {"type": "static", "conf": 0.9, "moves": False},
           "contacts": [], "mask_area_frac": 0.10}
    return {"meta": {"clip": "c", "duration": float(times[-1]), "scale": "relative"},
            "camera": {"intrinsics": {"fov_deg": 60, "width": 640, "height": 360},
                       "poses": [{"t": float(t), "pos": [0, 1.2, 5], "quat": [0, 0, 0, 1]}
                                 for t in times]},
            "static": {"planes": [{"kind": "ground", "conf": 0.8, "center_hint": [0, 0, 0],
                                   "extent_hint": [10, 0.2, 10]}]},
            "objects": [obj], "keyframes": []}


def structure_pos(program):
    nodes = program.get("static") or []
    rails = [n for n in nodes if n.get("id", "").startswith("rail")]
    assert rails, f"结构件没进 static：{[n.get('id') for n in nodes]}"
    return np.asarray(rails[0]["pose"]["pos"], float)


def test_取首帧不取中位():
    """首帧在 x=0，之后一路漂到 x=4。中位会落在 x=2 附近，首帧留在 0。"""
    centers = [[float(i) * 0.5, 0.2, 0.0] for i in range(9)]
    pos = structure_pos(evidence_to_program(evidence(centers), "c", {}))
    assert pos[0] == pytest.approx(0.0, abs=1e-6), "漂移的那一串不该把位置拖走"


def test_开关关掉就退回中位():
    centers = [[float(i) * 0.5, 0.2, 0.0] for i in range(9)]
    cfg = {"synthesis": {"structure_pos_first_frame": False}}
    pos = structure_pos(evidence_to_program(evidence(centers), "c", cfg))
    assert pos[0] == pytest.approx(2.0, abs=1e-6), "关掉之后该是逐帧中位"


def test_尺寸仍然取逐帧中位():
    """位置和尺寸是两个量：位置会漂，尺寸量过逐帧中位更好（#64），不要一起改。"""
    centers = [[0.0, 0.2, 0.0]] * 9
    sizes = [[9.9, 0.3, 0.4]] + [[2.0, 0.3, 0.4]] * 8      # 首帧尺寸是个离群值
    nodes = evidence_to_program(evidence(centers, sizes), "c", {}).get("static") or []
    rail = [n for n in nodes if n.get("id", "").startswith("rail")][0]
    assert rail["geom"]["extent"][0] == pytest.approx(2.0, abs=1e-6), \
        "尺寸该用中位把首帧那个离群值挡掉"


def test_没有漂移时两种估计一致():
    centers = [[1.5, 0.2, -0.5]] * 9
    a = structure_pos(evidence_to_program(evidence(centers), "c", {}))
    b = structure_pos(evidence_to_program(
        evidence(centers), "c", {"synthesis": {"structure_pos_first_frame": False}}))
    assert np.allclose(a, b)
