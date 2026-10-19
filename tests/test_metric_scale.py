"""度量尺度与接触装配。两件事都在改位姿和尺寸，所以都要能说清楚改了多少、为什么。"""
import pytest

from gwm.binding.platformer import scale_program
from gwm.synthesis.contact import ground_top, settle_on_ground
from gwm.synthesis.scale import estimate_metric_scale, size_prior_for


def obj(node_id, cls, extent, pos=(0.0, 1.0, 0.0), **extra):
    return {"id": node_id, "class": cls, "pose": {"pos": list(pos)},
            "geom": {"kind": "primitive", "shape": "box", "extent": list(extent)}, **extra}


def scene(objects, ground_extent=(20.0, 0.4, 20.0), ground_y=-0.2):
    return {"meta": {"clip": "c"},
            "static": [{"id": "ground", "class": "ground", "pose": {"pos": [0.0, ground_y, 0.0]},
                        "geom": {"kind": "primitive", "shape": "box", "extent": list(ground_extent)}}],
            "objects": objects}


def test_multi_word_prior_wins_over_its_suffix():
    """「toy train」是玩具，不是真火车。短键抢走长键的话尺度会差两个数量级。"""
    assert size_prior_for("toy train") == pytest.approx(0.20)
    assert size_prior_for("a wooden table") == pytest.approx(1.40)
    assert size_prior_for("unlabelled thing") is None


def test_scale_uses_the_median_so_one_bad_box_cannot_drag_it():
    # 两个物体重建得偏小一半，第三个离谱地偏小二十倍；中位数应该只跟着前两个走
    program = scene([obj("a", "cup", [0.05, 0.05, 0.05]),
                     obj("b", "table", [0.70, 0.35, 0.35]),
                     obj("c", "chair", [0.045, 0.02, 0.02])])
    factor, report = estimate_metric_scale(program)
    assert factor == pytest.approx(2.0, rel=0.02), report
    assert len(report["samples"]) == 3 and report["source"] == "table"


def test_vlm_sizes_beat_the_lookup_table():
    """模型看过画面，它的估计优先于按类名查表。"""
    program = scene([obj("a", "cup", [0.1, 0.1, 0.1]), obj("b", "cup", [0.1, 0.1, 0.1])])
    factor, report = estimate_metric_scale(program, vlm_sizes={"cup": [0.3, 0.3, 0.3]})
    assert factor == pytest.approx(3.0)
    assert report["source"] == "vlm"


def test_too_few_known_objects_means_no_change():
    program = scene([obj("a", "cup", [0.1, 0.1, 0.1]), obj("b", "mystery", [1, 1, 1])])
    factor, report = estimate_metric_scale(program)
    assert factor == 1.0 and "size prior" in report["reason"]


def test_absurd_factor_is_refused_rather_than_applied():
    """差到这个地步说明是别的环节错了，硬改会把看得见的错误变成看不见的。"""
    program = scene([obj("a", "cup", [1e-4, 1e-4, 1e-4]), obj("b", "mug", [1e-4, 1e-4, 1e-4])])
    factor, report = estimate_metric_scale(program)
    assert factor == 1.0 and "outside" in report["reason"]


def test_scaling_moves_look_at_and_velocity_with_everything_else():
    """只缩 pos 不缩 look_at，场景一放大相机就瞄到别处；初速度同理会显得变慢。"""
    program = scene([obj("a", "cup", [0.1, 0.1, 0.1],
                         motion={"type": "dynamic", "linear_velocity": [1.0, 0.0, -2.0]})])
    program["camera"] = {"intrinsics": {"fov_deg": 50, "far": 100},
                         "keyframes": [{"t": 0.0, "pos": [1.0, 2.0, 3.0], "look_at": [4.0, 5.0, 6.0]}]}
    scaled = scale_program(program, 2.0)
    keyframe = scaled["camera"]["keyframes"][0]
    assert keyframe["pos"] == [2.0, 4.0, 6.0]
    assert keyframe["look_at"] == [8.0, 10.0, 12.0]
    assert scaled["objects"][0]["motion"]["linear_velocity"] == [2.0, 0.0, -4.0]
    assert scaled["camera"]["intrinsics"]["far"] == 200.0


def test_settling_lifts_sunken_objects_onto_the_ground():
    # 地面顶面在 y=0，物体半高 0.25，中心却在 -0.1，也就是陷进去 0.35
    program = scene([obj("a", "cup", [0.5, 0.5, 0.5], pos=(0.0, -0.1, 0.0))])
    assert ground_top(program) == pytest.approx(0.0)
    moves = settle_on_ground(program)
    assert moves == [{"id": "a", "lift_m": pytest.approx(0.35, abs=1e-3), "applied": True}]
    assert program["objects"][0]["pose"]["pos"][1] == pytest.approx(0.25)


def test_settling_leaves_objects_that_already_rest_alone():
    program = scene([obj("a", "cup", [0.5, 0.5, 0.5], pos=(0.0, 0.25, 0.0))])
    assert settle_on_ground(program) == []


def test_settling_refuses_an_implausibly_large_lift():
    """抬升超过限度说明尺寸或地面估错了，记下来但不动——硬抬会掩盖真正的错误。"""
    program = scene([obj("a", "cup", [0.5, 0.5, 0.5], pos=(0.0, -9.0, 0.0))])
    moves = settle_on_ground(program, max_lift_m=1.0)
    assert moves[0]["applied"] is False and moves[0]["reason"] == "lift_exceeds_limit"
    assert program["objects"][0]["pose"]["pos"][1] == pytest.approx(-9.0)


def test_no_ground_means_no_settling():
    program = {"meta": {}, "static": [], "objects": [obj("a", "cup", [0.5, 0.5, 0.5], pos=(0, -5, 0))]}
    assert ground_top(program) is None and settle_on_ground(program) == []


def test_large_objects_anchor_the_global_scale():
    """小物体的包围盒被深度噪声撑大，让它们参与投票等于让噪声最大的样本定全场尺度。"""
    program = scene([obj("t1", "table", [1.4, 0.7, 0.7]),      # 比例 1.0
                     obj("t2", "bookshelf", [1.8, 0.9, 0.5]),  # 比例 1.0
                     obj("s1", "cup", [0.4, 0.4, 0.4]),        # 小物体被撑大，比例 0.25
                     obj("s2", "mug", [0.4, 0.4, 0.4])])
    factor, report = estimate_metric_scale(program)
    assert factor == pytest.approx(1.0, abs=0.01), report
    assert report["anchor"].startswith("objects >=")


def test_oversized_small_object_is_pulled_back_to_its_prior():
    from gwm.synthesis.scale import regularize_object_sizes
    program = scene([obj("train", "toy train", [1.2, 0.3, 0.3], pos=(1.0, 2.0, 3.0))])
    changes = regularize_object_sizes(program)
    assert len(changes) == 1 and changes[0]["id"] == "train"
    extent = program["objects"][0]["geom"]["extent"]
    assert max(extent) == pytest.approx(0.20)
    assert extent[0] / extent[1] == pytest.approx(4.0), "只改大小，长宽比要保住"
    assert program["objects"][0]["pose"]["pos"] == [1.0, 2.0, 3.0], "中心不该动"


def test_sizes_within_tolerance_are_left_untouched():
    """这一步是少数情况下的兜底，不是把所有东西都换成先验。"""
    from gwm.synthesis.scale import regularize_object_sizes
    program = scene([obj("t", "table", [1.9, 0.8, 0.8])])     # 1.9 / 1.4 = 1.36，在容差内
    assert regularize_object_sizes(program) == []
    assert program["objects"][0]["geom"]["extent"] == [1.9, 0.8, 0.8]
