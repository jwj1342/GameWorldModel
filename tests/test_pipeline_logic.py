"""管线里几段容易出错又不好肉眼验证的数值逻辑。"""
import numpy as np
import pytest

from gwm.compiler.validate import validate
from gwm.config import load_config
from gwm.perception.evidence import classify_motion, obb_from_points
from gwm.synthesis.direct import evidence_to_program, recentre_periodic

CFG = load_config()


def _evidence_with_lift():
    """一个沿 y 轴周期升降的平台，周期 4 秒、振幅 1.2 米，中心在 y=1。"""
    ts = np.linspace(0, 8, 33)
    centres = [[0.0, 1.0 + 1.2 * np.sin(2 * np.pi * t / 4.0), 0.0] for t in ts]
    return {
        "meta": {"clip": "x", "duration": 8.0, "scale": "relative"},
        "camera": {"intrinsics": {"fov_deg": 60, "width": 640, "height": 360},
                   "poses": [{"t": float(t), "pos": [0, 1.2, 5], "quat": [0, 0, 0, 1]} for t in ts]},
        "static": {"planes": [{"kind": "ground", "conf": 0.8, "center_hint": [0, 0, 0], "extent_hint": [10, 0.2, 10]}]},
        "objects": [{"id": "lift", "class_guess": "platform", "confidence": 0.9, "is_dynamic": True,
                     "obb": [{"t": float(t), "center": c, "quat": [0, 0, 0, 1], "size": [2, 0.3, 2]} for t, c in zip(ts, centres)],
                     "motion_guess": {"type": "periodic_translate", "axis": [0, 1, 0], "period": 4.0, "amp": 1.2, "conf": 0.8},
                     "contacts": []}],
        "keyframes": [],
    }


@pytest.mark.parametrize("kind, centres, expected", [
    ("周期往复", lambda t: np.stack([np.zeros_like(t), 1 + 1.2 * np.sin(2 * np.pi * t / 4), np.zeros_like(t)], 1), "periodic_translate"),
    ("原地不动", lambda t: np.tile([1.0, 2.0, 3.0], (len(t), 1)), "static"),
    ("单向滑动", lambda t: np.stack([t * 0.5, np.ones_like(t), np.zeros_like(t)], 1), ("prismatic", "trajectory")),
])
def test_motion_classification(kind, centres, expected):
    """运动类型判断是感知里最容易出错的一步，三种典型运动必须认对。"""
    ts = np.linspace(0, 8, 33)
    got = classify_motion(ts, centres(ts), np.zeros_like(ts), CFG)["type"]
    assert got in (expected if isinstance(expected, tuple) else (expected,)), f"{kind} 被判成 {got}"


def test_obb_recovers_the_real_size():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1, 0, -0.25], [1, 0.5, 0.25], size=(500, 3))
    ob = obb_from_points(pts)
    assert abs(ob["size"][0] - 2) < 0.3 and abs(ob["size"][2] - 0.5) < 0.2


def test_camera_convention_roundtrip():
    """相机约定一旦搞反，整个场景会前后或上下翻转，而且渲染出来不一定看得出来。
    相机前方 2 米的点应该落在 y 朝上世界的 -z，再投影回去应该回到画面中心。"""
    from gwm.perception.evidence import _fix, to_world
    from gwm.perception.overlay import project
    m = _fix(np.eye(4))
    assert np.allclose(to_world(np.array([[0.0, 0.0, 2.0]]), m), [[0, 0, -2]])
    K = np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1]], float)
    uv, ok = project(np.array([[0.0, 0.0, -2.0]]), K, m, 1.0)
    assert ok[0] and np.allclose(uv[0], [320, 180])


def test_direct_translation_produces_a_valid_program():
    """没有模型时的兜底路径，必须永远产出能编译的程序。"""
    program = evidence_to_program(_evidence_with_lift(), "x", CFG)
    assert validate(program)["ok"]
    assert program["objects"][0]["motion"]["type"] == "periodic_translate"


def test_periodic_pose_is_the_oscillation_centre():
    """内核把 periodic 的 pose.pos 当作振荡中心。模型常给成首帧位置，那样整条轨迹会偏掉一个振幅，
    这是合成片段上模型分数低于直译的原因，所以生成之后要强制重心化。"""
    ev = _evidence_with_lift()
    node = {"id": "lift", "pose": {"pos": [0.0, 1.0 + 1.2, 0.0]},   # 模型给成了最高点
            "motion": {"type": "periodic_translate", "axis": [0, 1, 0], "amp": 1.2, "period": 4.0, "phase": 0.0}}
    recentre_periodic(node, ev["objects"][0]["obb"])
    assert abs(node["pose"]["pos"][1] - 1.0) < 0.05


def test_binding_fills_the_slots_and_stays_valid():
    """绑定要给出出生点和目标，并且改完之后程序仍然合法。"""
    from gwm.binding.platformer import bind
    ev = _evidence_with_lift()
    ev["objects"].append({"id": "floor", "class_guess": "floor", "confidence": 0.9, "is_dynamic": False,
                          "obb": [{"t": t, "center": [0, -0.1, 0], "quat": [0, 0, 0, 1], "size": [6, 0.2, 6]} for t in (0.0, 1.0)],
                          "motion_guess": {"type": "static", "conf": 0.8}, "contacts": []})
    program = evidence_to_program(ev, "x", CFG)
    assert any(s["id"] == "floor" for s in program["static"])   # 静态类别要进静态结构而不是物体
    bound = bind(program, ev, CFG)
    assert validate(bound)["ok"]
    assert bound["binding"]["slots"]["player_spawn"] and bound["binding"]["slots"]["goal_volume"]
