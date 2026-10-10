"""管线里几段容易出错又不好肉眼验证的数值逻辑。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gwm.compiler.validate import validate
from gwm.config import REPO, load_config
from gwm.perception.evidence import obb_from_points
from gwm.perception.motion import estimate_motion
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
    ("静态噪声", lambda t: np.tile([1.0, 2.0, 3.0], (len(t), 1))
     + 0.001 * np.random.default_rng(0).standard_normal((len(t), 3)), "static"),
    ("单向滑动", lambda t: np.stack([t * 0.5, np.ones_like(t), np.zeros_like(t)], 1), ("prismatic", "trajectory")),
])
def test_motion_classification(kind, centres, expected):
    """运动类型判断是感知里最容易出错的一步，三种典型运动必须认对。"""
    ts = np.linspace(0, 8, 33)
    quaternions = R.from_euler("y", np.zeros((len(ts), 1))).as_quat()
    got = estimate_motion(ts, centres(ts), quaternions, CFG, coordinate_space="world")["motion_guess"]["type"]
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


def test_seed_reaches_the_model_call(monkeypatch):
    """运行级 seed 要真的发到模型调用上，单次调用可以覆盖它。

    之前 VLMClient 协议里就有 seed 参数、客户端也会透传，但整条管线没有任何地方传它，
    所以"同一配置多跑几个 seed"这件事实际做不到。这个用例把接线钉住。
    """
    from types import SimpleNamespace
    import openai
    from gwm.synthesis.vlm import OpenAICompatClient

    sent = {}

    def client_with(cfg_seed):
        def capture(**kwargs):
            sent.clear(); sent.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"),
                                                           finish_reason="stop")], usage=None)
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: fake)
        cfg = {"vlm": {"provider_kind": "openai", "endpoint": "http://127.0.0.1:1/v1",
                       "api_key": "unit-test-placeholder", "model": "stub", "seed": cfg_seed,
                       "temperature": 0.4, "max_tokens": 100, "max_retries": 1}}
        return OpenAICompatClient(cfg, None)

    def seed_sent(cfg_seed, call_seed=None):
        c = client_with(cfg_seed)
        c.chat("sys", "user", **({"seed": call_seed} if call_seed is not None else {}))
        return sent.get("seed", "absent")

    assert seed_sent(None) == "absent"        # 不配就不发，保持原有行为
    assert seed_sent(1234) == 1234            # 配了就发
    assert seed_sent(1234, 99) == 99          # 单次调用覆盖运行级设置
    assert seed_sent(0) == 0                 # 零是合法seed，不应当作未配置


def test_editability_judge_catches_side_effects_and_fabrication():
    """可编辑性实验的判定逻辑本身要可信，否则跑出来的数字没意义。

    判定要能分清四种情况：改对了、改了但有副作用、该改的没改、
    以及该拒绝的时候编造了改动。
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from run_editability import judge

    want = {"id": "x", "kind": "运动参数", "instruction": "让它快一倍",
            "expect_paths": ["/objects/0/motion/period"],
            "expect_value": {"/objects/0/motion/period": 2.0}}
    prog = {"objects": [{"motion": {"period": 2.0}}]}

    good = {"ok": True, "ops": [{"op": "replace"}], "changed": ["/objects/0/motion/period"],
            "program": prog, "error": "", "note": ""}
    assert judge(want, good)["ok"]

    # 顺手把别的物体也改了，这是副作用，要判失败
    noisy = {**good, "changed": ["/objects/0/motion/period", "/objects/3/motion/rate_dps"]}
    assert not judge(want, noisy)["ok"]

    # 改了，但改的不是要求的地方
    missed = {**good, "changed": ["/objects/1/pose/pos"]}
    assert not judge(want, missed)["ok"]

    # 路径对了但数值不对
    wrong_value = {**good, "program": {"objects": [{"motion": {"period": 3.5}}]}}
    assert not judge(want, wrong_value)["ok"]

    refuse = {"id": "y", "kind": "应当拒绝", "instruction": "把那只猫移走", "expect_refuse": True}
    assert judge(refuse, {"ok": False, "ops": [], "changed": [], "program": {}, "error": "", "note": ""})["ok"]
    # 场景里没有猫却给出了改动，这是编造，要判失败
    assert not judge(refuse, {"ok": True, "ops": [{"op": "replace"}], "changed": ["/objects/0/pose/pos"],
                              "program": {}, "error": "", "note": ""})["ok"]
def test_gt_metrics_match_objects_and_distance():
    """真值指标的物体匹配：对得上的要配对，离太远的要判成没匹配上。

    匹配做错会让召回和误检一起失真，而这两项是论文主表里的数字，
    所以这段逻辑单独钉一下。不需要浏览器。
    """
    from gwm.feedback.gt_metrics import match_objects

    times = np.linspace(0, 1, 5)
    gt = {"a": {"t": times, "pos": np.tile([0.0, 0, 0], (5, 1)), "class": "x"},
          "b": {"t": times, "pos": np.tile([5.0, 0, 0], (5, 1)), "class": "x"}}

    # 预测和真值几乎重合，两个都该配上
    pred = {"p": {"t": times, "pos": np.tile([0.1, 0, 0], (5, 1)), "class": "x"},
            "q": {"t": times, "pos": np.tile([5.1, 0, 0], (5, 1)), "class": "x"}}
    pairs = dict((g, p) for g, p, _ in match_objects(gt, pred, times, max_dist_m=1.0))
    assert pairs == {"a": "p", "b": "q"}

    # 离得太远的不许凑数，宁可判成没匹配上
    far = {"z": {"t": times, "pos": np.tile([50.0, 0, 0], (5, 1)), "class": "x"}}
    assert all(p is None for _, p, _ in match_objects(gt, far, times, max_dist_m=3.0))

    # 预测里一个都没有时，全部算漏检
    assert all(p is None for _, p, _ in match_objects(gt, {}, times, max_dist_m=3.0))
