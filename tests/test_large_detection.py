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


def obj_with(oid, cls, motion_guess, area=0.03):
    box = {"center": [0.0, 0.5, 0.0], "quat": [0, 0, 0, 1], "size": [1.0, 1.0, 1.0], "conf": 0.9}
    return {"id": oid, "class_guess": cls, "confidence": 0.9, "is_dynamic": True,
            "obb": [{**box, "t": float(t)} for t in (0.0, 1.0, 2.0)],
            "motion_guess": motion_guess, "contacts": [], "mask_area_frac": area}


def test_类名像结构件而且没有证据会动的_按结构件处理():
    """原来这里要求运动分类**肯定地说**它静止，那是个坏掉的默认。

    运动分类 85% 的时候返回 unknown，而它的贡献实测精确为零（#7）。后果：类名含
    rail/block 的 61 个检测里只有 4 个满足那个与条件，其余全默认进了 objects，
    在评测里变成假阳性——哪怕检测完全正确。真值里 rail 有 45 个，全在 static。
    """
    for guess in ({"type": "unknown", "conf": "unknown"},
                  {"type": "unknown", "conf": "unknown", "moves": False},
                  {"type": "static", "conf": 0.8}):
        program = evidence_to_program(evidence([obj_with("rail_1", "rail", guess)]), "c", {})
        assert "rail_1" in ids(program["static"]), f"{guess} 下没按结构件处理"
        assert "rail_1" not in ids(program["objects"])


def test_建筑的一部分忽略二元moves():
    """**这条测的是 `moves=True` 这一档**，上面那条只测了 False 和缺省。

    `moves` 在静止结构件上假阳性很高：实测 25 条片段的 evidence 里
    `rail_* moves=True` 占 60/68（88.2%）、`block_*` 占 19/26（73%），
    而真值里这两类全是 static。多半是 ICP 在相机运动下无中生有解出转动。

    后果不是少判一个物体，是结构件被送进资产检索——真值里结构件是 primitive，
    **根本没有「该挑哪个」这个答案**，于是那批决策命中恒为 0。

    所以「建筑的一部分」这一档只认明确的运动类型，二元标志不算。
    `belt`/`table` 不在这一档，由下面那条守着不被误伤。
    """
    for cls, oid in (("rail", "rail_1"), ("block", "block_1")):
        guess = {"type": "unknown", "conf": "unknown", "moves": True}
        program = evidence_to_program(evidence([obj_with(oid, cls, guess)]), "c", {})
        assert oid in ids(program["static"]), f"{cls} 的二元 moves 被当成了会动的证据"
        assert oid not in ids(program["objects"])


def test_类名像结构件但有证据会动的_仍然是物体():
    """传送带会动，轨道上的东西也可能会动。有证据就别按结构件处理——
    `moves` 是真值感知上查全 0.925 的那个信号，不是七分类那个坏掉的。"""
    guess = {"type": "unknown", "conf": "unknown", "moves": True}
    program = evidence_to_program(evidence([obj_with("belt_1", "belt", guess)]), "c", {})
    assert "belt_1" in ids(program["objects"]), "有证据会动却被当成了固定结构"


def test_陷地丢弃跑在接触装配之后():
    """**顺序是这条规则的全部要害。**

    第一版我把它放在程序刚生成的时候（`direct.py`），离线估算说「砍掉 6 个、
    全是假阳性、一个命中不丢、精确率 0.214 -> 0.227」，实际接进链路跑出来是
    **砍掉 57 个、召回 0.500 -> 0.268**。

    原因是接触装配本来就负责把陷在地里的物体抬上来，而我在它之前把那些物体丢掉了
    ——丢的正是它要修的那一批。而离线估算用的是**最终程序**（装配已经跑过），
    所以只看到还剩 6 个。拿「处理后的数据」估算「处理前的闸」，估的是另一个分布。
    """
    from gwm.synthesis.contact import drop_sunk

    # 陷得很深的：底面在地平面下方三倍自身高度
    program = {"objects": [
        {"id": "ghost_1", "class": "crate", "geom": {"extent": [1.0, 1.0, 1.0]},
         "pose": {"pos": [0.0, -3.0, 0.0]}},
        {"id": "ok_1", "class": "crate", "geom": {"extent": [1.0, 1.0, 1.0]},
         "pose": {"pos": [0.0, 0.5, 0.0]}},
    ]}
    dropped = drop_sunk(program)
    assert [d["id"] for d in dropped] == ["ghost_1"]
    assert [n["id"] for n in program["objects"]] == ["ok_1"]
    assert program["meta"]["dropped_sunk"][0]["id"] == "ghost_1"   # 丢了什么要留痕


def test_刚生成的程序不该直接丢陷地的():
    """直译出来的位姿取自包围盒中心，地面估歪一点就会有物体陷进地里——那是接触装配
    要修的，不是该丢的。所以 evidence_to_program 里不能有这道闸。"""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "gwm" / "synthesis" / "direct.py").read_text(encoding="utf-8")
    assert "dropped_sunk" not in src, "下陷闸又跑回 direct.py 了，它会丢掉接触装配要修的物体"


def test_陷在地面下太深的直接丢掉_旧位置已移除():
    """底面在地平面下方好几倍自身高度的「物体」不是能拿起来的东西。

    实测（#56）：103 个预测物体里砍掉 6 个，**全是假阳性，一个命中都没丢**，
    精确率 0.214 -> 0.227。这是今天为过检量过的十七条信号里唯一不花代价的一条。
    """
    box = {"center": [0.0, -3.0, 0.0], "quat": [0, 0, 0, 1], "size": [1.0, 1.0, 1.0], "conf": 0.9}
    sunk = {"id": "ghost_1", "class_guess": "crate", "confidence": 0.9, "is_dynamic": True,
            "obb": [{**box, "t": float(t)} for t in (0.0, 1.0, 2.0)],
            "motion_guess": {"type": "unknown", "conf": "unknown"},
            "contacts": [], "mask_area_frac": 0.03}
    program = evidence_to_program(evidence([sunk]), "c", {})
    # 这一步**不该**丢它——接触装配会把它抬上来。丢弃发生在装配之后。
    assert "ghost_1" in ids(program["objects"])


def test_正常坐在地面上的不受影响():
    program = evidence_to_program(evidence([obj("crate_1", "crate", 0.03)]), "c", {})
    assert "crate_1" in ids(program["objects"])
    assert "dropped_sunk" not in program.get("meta", {})
