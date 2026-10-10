"""角色推断的参照系：从哪一层量高度、多实例怎么算、姿态算不算。

这三条判错会直接改掉绑定出来的关卡——收集物变摆设、立着的板子变平台——
而且渲染出来的图看不出区别，只有玩起来才发现。所以逐条钉死。
"""
import math
import pytest
from scipy.spatial.transform import Rotation as R

from gwm.binding.affordance import classify, ground_level


def quat(axis: str, deg: float) -> list[float]:
    return [float(v) for v in R.from_euler(axis, deg, degrees=True).as_quat()]


def scene(objects, floor_y: float = 0.0, extra_static=()):
    """地面顶面落在 floor_y，这样物体写 y 就是离地高度。"""
    return {"static": [{"id": "ground", "class": "floor",
                        "geom": {"kind": "primitive", "shape": "box", "extent": [24, 0.2, 24]},
                        "pose": {"pos": [0, floor_y - 0.1, 0]}}, *extra_static],
            "objects": objects}


def coin(y: float = 1.0, **kw):
    return {"id": "coin", "class": "thing", "geom": {"kind": "primitive", "shape": "box", "extent": [0.5, 0.06, 0.5]},
            "pose": {"pos": [0, y, 0]}, "motion": {"type": "spin"}, **kw}


def board(q=None, y: float = 1.0, extent=(2.0, 0.1, 2.0)):
    """一块 2x0.1x2 的板子，躺着能站人，立起来不能，而 extent 两种情况一模一样。"""
    pose = {"pos": [0, y, 0]}
    if q is not None: pose["quat"] = q
    return {"id": "board", "class": "thing", "geom": {"kind": "primitive", "shape": "box", "extent": list(extent)},
            "pose": pose}


def role(program, oid):
    return classify(program)[oid]["role"]


@pytest.mark.parametrize("floor_y", [0.0, 10.0, -7.5, 1000.0])
def test_whole_scene_offset_does_not_change_any_role(floor_y):
    """整个场景在世界坐标里抬高或压低，玩家和物体的相对关系一点没变，角色就不该变。

    重建出来的场景落在哪一层取决于尺度对齐怎么定原点，是观测不到的规范自由度。
    之前拿绝对高度和 reach_max_m 比，场景一抬高，收集物就集体变成摆设。
    """
    program = scene([coin(y=floor_y + 1.0), board(y=floor_y + 0.5)], floor_y=floor_y)
    assert ground_level(program) == pytest.approx(floor_y)
    assert role(program, "coin") == "collectible"
    assert role(program, "board") == "platform"


def test_collectible_on_a_high_ledge_is_still_reachable_through_support():
    """台子有三米高，硬币放在台面上。玩家爬上台子就能捡，所以该从台面量起。"""
    ledge = {"id": "ledge", "class": "ledge", "geom": {"kind": "primitive", "shape": "box", "extent": [4, 0.4, 4]},
             "pose": {"pos": [0, 2.8, 0]}}       # 顶面在 3.0
    program = scene([coin(y=3.3, support="ledge")], extra_static=[ledge])
    assert role(program, "coin") == "collectible"
    # 不声明 support 就只能按地面量，3.3 米超出够得着的范围
    assert role(scene([coin(y=3.3)], extra_static=[ledge]), "coin") == "decoration"


def test_instance_order_does_not_decide_the_role():
    """同一个物体的多个实例高度不同时，角色不该跟着数组顺序变。

    只读 instances[0] 的话，把够得着的那枚硬币挪到数组末尾，整组硬币就全成了摆设。
    """
    low, high = {"pos": [0, 1.0, 0]}, {"pos": [0, 9.0, 0]}
    geom = {"kind": "primitive", "shape": "box", "extent": [0.5, 0.06, 0.5]}
    def coins(instances):
        return scene([{"id": "coins", "class": "thing", "geom": geom,
                       "instances": instances, "motion": {"type": "spin"}}])
    assert role(coins([low, high]), "coins") == "collectible"
    assert role(coins([high, low]), "coins") == "collectible"
    assert role(coins([high, high]), "coins") == "decoration"


def test_an_upright_board_is_not_a_platform():
    """绕 x 轴转九十度，板子立起来了，但局部 extent 一个字没变。

    按 extent[0] * extent[2] 算顶面，立着的板子和躺着的板子一样是 4 平米，
    会被当成平台，绑定还会把它排进可行走面。
    """
    assert role(scene([board()]), "board") == "platform"
    assert role(scene([board(q=quat("x", 90))]), "board") == "decoration"
    assert role(scene([board(q=quat("z", 90))]), "board") == "decoration"
    # 转满一圈回到原位，还是平台
    assert role(scene([board(q=quat("x", 180))]), "board") == "platform"


@pytest.mark.parametrize("tilt_deg, expected", [(0, "platform"), (30, "platform"), (50, "platform"),
                                                (60, "decoration"), (89, "decoration")])
def test_a_slope_steeper_than_the_controller_can_climb_is_not_a_platform(tilt_deg, expected):
    """角色控制器的 setMaxSlopeClimbAngle 是 55 度，比这陡就爬不上去，不能算平台。"""
    assert role(scene([board(q=quat("x", tilt_deg))]), "board") == expected


def test_a_tilted_top_counts_only_its_horizontal_projection():
    """斜着的顶面，能站的是它的水平投影，不是面本身的面积。

    这块板子 1.5 x 0.4 平放是 0.6 平米，站得住；倾斜五十度之后投影只剩 0.39 平米，
    低于站人的下限。坡度本身只有五十度，角色爬得上去，所以拦住它的是面积不是坡度。
    """
    narrow = (1.5, 0.1, 0.4)
    assert role(scene([board(extent=narrow)]), "board") == "platform"
    assert math.cos(math.radians(50)) * 1.5 * 0.4 < 0.45
    assert role(scene([board(q=quat("x", 50), extent=narrow)]), "board") == "decoration"
