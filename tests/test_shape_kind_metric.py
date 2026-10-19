"""形状**种类**判对了没有——一个之前完全没在量的精细度维度。

为什么要单立一个指标（#56）。真值场景全由基本体组成：objects 是
28 box / 16 sphere / 9 cone / 6 cylinder，static 是 69 box。而我们产出的每个物体
节点都是 `kind: "asset"`，其中 51.4% 连 `ref` 都没有（资产库里没有 spool/totem/
beacon 这些类），内核对它们按**类名正则**兜底挑形状——类名准确率只有 12.5%，而且
真值用的那些类名一个都不匹配那些正则，**于是全部渲成盒子**。

真值里盒子只占 28/59，所以超过一半的物体形状是错的。

**而 `shape_ok` 看不见这件事**：它比的是三条边归一化之后的比例，一个该是球、却渲成
同样尺寸盒子的物体在它上面是满分。所以必须单独量种类。

`effective_shape` 复刻了内核 `fallbackForClass` 的映射——那是跨语言重复一份逻辑，
是明确的债，但不复刻就量不了。哪天把有效形状写进程序，那段就该删掉。
"""
import pytest

from gwm.feedback.gt_metrics import effective_shape


def node(geom, cls=None):
    n = {"geom": geom}
    if cls is not None:
        n["class"] = cls
    return n


def test_基本体直接读形状():
    assert effective_shape(node({"kind": "primitive", "shape": "sphere", "radius": 0.1})) == "sphere"
    assert effective_shape(node({"kind": "primitive", "shape": "cone"})) == "cone"
    assert effective_shape(node({"kind": "primitive"})) == "box", "没写 shape 的按盒子算"


def test_解析到资产的算网格():
    g = {"kind": "asset", "query": "buoy", "ref": "watercraft/buoy", "extent": [1, 1, 1]}
    assert effective_shape(node(g, "buoy")) == "mesh"


def test_没解析到的走内核那套类名兜底():
    """这正是现状：资产库里没有的类，内核按类名正则挑形状。"""
    assert effective_shape(node({"kind": "asset", "query": "marble"}, "marble")) == "sphere"
    assert effective_shape(node({"kind": "asset", "query": "coin"}, "coin")) == "cylinder"
    assert effective_shape(node({"kind": "asset", "query": "door"}, "door")) == "box"


def test_真值用的那些类名一个都不匹配_所以全是盒子():
    """这是这个指标要暴露的核心事实。真值里 spool / totem / beacon / girder /
    slab / pod 这些类，内核的正则一个都不匹配，于是全部落到默认的盒子。"""
    for cls in ("spool", "totem", "beacon", "girder", "slab", "pod", "canister", "bollard"):
        assert effective_shape(node({"kind": "asset", "query": cls}, cls)) == "box", \
            f"{cls} 落到了别的分支，内核那边的正则可能改过，两边要对齐"


def test_复合体算一类():
    assert effective_shape(node({"kind": "asset", "query": "tree"}, "tree")) == "composite"
    assert effective_shape(node({"kind": "composite", "parts": []})) == "composite"


def test_认不出的几何返回_None():
    assert effective_shape(node({"kind": "heightfield"})) is None
    assert effective_shape({"geom": {}}) is None


def test_和内核那份映射保持一致():
    """跨语言重复了一份逻辑，所以要有东西盯着它别漂。这里对着 kernel/scene.js
    的源码核对正则本身。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "kernel" / "scene.js").read_text(encoding="utf-8")
    for pattern in ("coin|disc|ring|wheel", "ball|sphere|orb|marble",
                    "tree|plant|bush", "door|gate|panel"):
        assert pattern in src, f"内核那边的正则 {pattern} 变了，effective_shape 要跟着改"


def test_指标进了评测和汇总表():
    from gwm.feedback.gt_metrics import evaluate
    from gwm.feedback.report import as_markdown, compare

    gt = {"objects": [{"id": "a", "pose": {"pos": [0, 0.5, 0]},
                       "geom": {"kind": "primitive", "shape": "sphere", "radius": 0.2}},
                      {"id": "b", "pose": {"pos": [4, 0.5, 0]},
                       "geom": {"kind": "primitive", "shape": "box", "extent": [1, 1, 1]}}]}
    pred = {"objects": [{"id": "p", "class": "spool", "pose": {"pos": [0, 0.5, 0]},
                         "geom": {"kind": "asset", "query": "spool", "extent": [0.4, 0.4, 0.4]}},
                        {"id": "q", "class": "spool", "pose": {"pos": [4, 0.5, 0]},
                         "geom": {"kind": "asset", "query": "spool", "extent": [1, 1, 1]}}]}

    def states(prog):
        return [{"t": t, "camera": {"pos": [0, 1.1, 5]},
                 "objects": [{"id": n["id"], "name": n["id"], "kind": "object",
                              "pos": n["pose"]["pos"]} for n in prog["objects"]]}
                for t in (0.0, 0.5, 1.0)]

    out = evaluate(gt, pred, states(gt), states(pred), max_dist_m=0.45)
    sk = out["full"]["shape_kind"]
    assert sk["n"] == 2
    # 真值一球一盒，我们两个都渲成盒子 -> 判对一个
    assert sk["correct"] == 1 and sk["accuracy"] == pytest.approx(0.5)
    assert sk["confusion"].get("sphere->box") == 1

    md = as_markdown(compare({"t": [{"config_identity": "t", "metrics": out}]}, "full"))
    assert "形状种类" in md
