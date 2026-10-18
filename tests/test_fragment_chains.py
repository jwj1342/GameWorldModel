"""排成一条线的同类小碎块要合成一个长条。

为什么需要（#56）：真值里一根轨道是 6.4 比 1 的长条，但掩码只覆盖它的一小段，
于是一根轨道被切成好几块、每块都成了独立「物体」。实测十一条片段上，落在真值
static 上的 25 个检出只对应 **11 个**真值结构件（平均 2.27 次，一个被切成 6 块），
碎片尺寸中位只有真值的 0.208。

这解释了为什么二十条「把假的滤掉」的信号全部失败：**那些东西不是假的**。

`merge_duplicates` 碰不到它们——碎片沿轨道排开，相对距离远超它的 0.5 阈值，
而把阈值调大会把真正挨近的不同物体并掉。所以这里用碎片有、真实物体没有的特征：
**排成一条线**。
"""
from gwm.synthesis.dedup import merge_fragment_chains


def node(oid, pos, extent, cls="rail"):
    return {"id": oid, "class": cls, "geom": {"kind": "asset", "extent": list(extent)},
            "pose": {"pos": list(pos)}}


def ids(program):
    return [n["id"] for n in program["objects"]]


def test_一排同类碎块合成一个长条():
    # 五块 0.2 米的小方块沿 x 排开，间距 0.3 米——一根 1.5 米的轨道被切碎的样子
    objs = [node(f"rail_{i}", [i * 0.3, 0.05, 0.0], [0.2, 0.1, 0.1]) for i in range(5)]
    program = {"objects": objs}
    merged = merge_fragment_chains(program)
    assert len(merged) == 1, f"没合上：{merged}"
    assert len(program["objects"]) == 1, f"剩下 {ids(program)}"
    kept = program["objects"][0]
    # 合出来必须覆盖整条，而且是细长的
    assert kept["geom"]["extent"][0] > 1.3, f"长度没覆盖整条：{kept['geom']['extent']}"
    assert merged[0]["elongation"] >= 3.0
    assert "碎片" in kept.get("notes", "")


def test_挤成一团的不合():
    """三个挨在一起但不成线的——那是堆叠的箱子，是三个真物体，不能并掉。"""
    objs = [node("box_0", [0.0, 0.05, 0.0], [0.2, 0.2, 0.2], "crate"),
            node("box_1", [0.25, 0.05, 0.0], [0.2, 0.2, 0.2], "crate"),
            node("box_2", [0.12, 0.05, 0.22], [0.2, 0.2, 0.2], "crate")]
    program = {"objects": objs}
    merged = merge_fragment_chains(program)
    assert merged == [], f"把一堆箱子并成长条了：{merged}"
    assert len(program["objects"]) == 3


def test_只有两块不合():
    """两个挨近的本来就该交给 merge_duplicates，这里不抢它的活。"""
    objs = [node("rail_0", [0.0, 0.05, 0.0], [0.2, 0.1, 0.1]),
            node("rail_1", [0.3, 0.05, 0.0], [0.2, 0.1, 0.1])]
    program = {"objects": objs}
    assert merge_fragment_chains(program) == []
    assert len(program["objects"]) == 2


def test_隔得太远的不连成链():
    objs = [node(f"rail_{i}", [i * 5.0, 0.05, 0.0], [0.2, 0.1, 0.1]) for i in range(5)]
    program = {"objects": objs}
    assert merge_fragment_chains(program) == []
    assert len(program["objects"]) == 5


def test_不同类别不合():
    objs = [node("a", [0.0, 0.05, 0.0], [0.2, 0.1, 0.1], "rail"),
            node("b", [0.3, 0.05, 0.0], [0.2, 0.1, 0.1], "crate"),
            node("c", [0.6, 0.05, 0.0], [0.2, 0.1, 0.1], "beacon")]
    program = {"objects": objs}
    assert merge_fragment_chains(program) == []
