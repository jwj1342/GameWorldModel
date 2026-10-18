"""场景几何那一档：object + static 一起算。

为什么要有它。`objects` 那一档的分母只有真值 objects，而这批数据的真值里 static
有 69 个、object 只有 59 个——**54% 的几何体从来没进过任何指标**。后果是一个正确
落在结构件上的检出被记成假阳性：实测 112 个预测物体里 22.3% 落在真值 static 上，
而且验过是碎片不是误分类（见 #56）。它们在 `objects` 的精确率里全部算错。

这一档是**纯增量**：规范变换沿用 objects 那一档拟合出来的，不重新拟合。重新拟合
会把对齐往结构件上拉（static 点更多），`objects` 的历史数就变了，两档也不可比。
"""
import numpy as np
import pytest

from gwm.feedback.gt_metrics import evaluate


def states(nodes, times=(0.0, 0.5, 1.0), cam=(0.0, 1.1, 0.0)):
    return [{"t": float(t), "camera": {"pos": list(cam)},
             "objects": [{"id": i, "name": i, "kind": k, "pos": list(p)}
                         for i, k, p in nodes]}
            for t in times]


def program(objects, static):
    return {"objects": [{"id": i, "pose": {"pos": list(p)}} for i, p in objects],
            "static": [{"id": i, "pose": {"pos": list(p)}} for i, p in static]}


def test_落在结构件上的检出不再被当成假阳性():
    """一个预测摆在真值 static 的位置上：objects 那一档把它记成假阳性，
    scene 那一档认出来。这正是 22.3% 那一批的处境。"""
    gt_nodes = [("g_obj", "object", (0, 0.5, 0)), ("g_wall", "static", (6, 0.5, 0))]
    pred_nodes = [("p_obj", "object", (0, 0.5, 0)), ("p_wall", "static", (6, 0.5, 0))]
    out = evaluate(program([("g_obj", (0, 0.5, 0))], [("g_wall", (6, 0.5, 0))]),
                   program([("p_obj", (0, 0.5, 0))], [("p_wall", (6, 0.5, 0))]),
                   states(gt_nodes), states(pred_nodes), max_dist_m=0.45)
    ob, sc = out["full"]["objects"], out["full"]["scene"]
    assert ob["gt"] == 1 and ob["matched"] == 1
    assert sc["gt"] == 2 and sc["matched"] == 2, "结构件也该被算进去"
    assert sc["gt_static"] == 1 and sc["pred_static"] == 1


def test_结构件摆错了照样扣分():
    """这一档不是放水：结构件位置错了，场景召回和精确率都要掉。"""
    gt_nodes = [("g_obj", "object", (0, 0.5, 0)), ("g_wall", "static", (6, 0.5, 0))]
    pred_nodes = [("p_obj", "object", (0, 0.5, 0)), ("p_wall", "static", (6, 0.5, 9.0))]
    out = evaluate(program([("g_obj", (0, 0.5, 0))], [("g_wall", (6, 0.5, 0))]),
                   program([("p_obj", (0, 0.5, 0))], [("p_wall", (6, 0.5, 9.0))]),
                   states(gt_nodes), states(pred_nodes), max_dist_m=0.45)
    sc = out["full"]["scene"]
    assert sc["matched"] == 1 and sc["recall"] == pytest.approx(0.5)
    assert sc["precision"] == pytest.approx(0.5)


def test_不改动既有的_objects_那一档():
    """纯增量：加上一堆 static 之后，objects 的数必须一个不变。"""
    objs = [("g1", "object", (0, 0.5, 0)), ("g2", "object", (4, 0.5, 0))]
    preds = [("p1", "object", (0, 0.5, 0)), ("p2", "object", (4, 0.5, 0))]
    gp = program([("g1", (0, 0.5, 0)), ("g2", (4, 0.5, 0))], [])
    pp = program([("p1", (0, 0.5, 0)), ("p2", (4, 0.5, 0))], [])
    bare = evaluate(gp, pp, states(objs), states(preds), max_dist_m=0.45)

    walls = [(f"w{i}", "static", (i * 2.0, 0.5, 7.0)) for i in range(6)]
    gp2 = program([("g1", (0, 0.5, 0)), ("g2", (4, 0.5, 0))],
                  [(f"w{i}", (i * 2.0, 0.5, 7.0)) for i in range(6)])
    withwalls = evaluate(gp2, pp, states(objs + walls), states(preds), max_dist_m=0.45)

    assert withwalls["full"]["objects"] == bare["full"]["objects"], \
        "static 只该进 scene 那一档，objects 的历史数不许动"
    assert withwalls["full"]["scene"]["gt"] == 8
    assert withwalls["full"]["scene"]["recall"] == pytest.approx(2 / 8)


def test_没有结构件时退化成和_objects_一致():
    objs = [("g1", "object", (0, 0.5, 0))]
    preds = [("p1", "object", (0, 0.5, 0))]
    out = evaluate(program([("g1", (0, 0.5, 0))], []), program([("p1", (0, 0.5, 0))], []),
                   states(objs), states(preds), max_dist_m=0.45)
    ob, sc = out["full"]["objects"], out["full"]["scene"]
    assert (sc["gt"], sc["pred"], sc["matched"]) == (ob["gt"], ob["pred"], ob["matched"])


def test_汇总表多出场景两列():
    from gwm.feedback.report import as_markdown, compare
    gt_nodes = [("g_obj", "object", (0, 0.5, 0)), ("g_wall", "static", (6, 0.5, 0))]
    out = evaluate(program([("g_obj", (0, 0.5, 0))], [("g_wall", (6, 0.5, 0))]),
                   program([("p_obj", (0, 0.5, 0))], [("p_wall", (6, 0.5, 0))]),
                   states(gt_nodes),
                   states([("p_obj", "object", (0, 0.5, 0)), ("p_wall", "static", (6, 0.5, 0))]),
                   max_dist_m=0.45)
    table = compare({"t": [{"config_identity": "t", "metrics": out}]}, "full")
    md = as_markdown(table)
    assert "场景召回" in md and "场景精确率" in md
    assert table["t"]["scene_recall"]["median"] == pytest.approx(1.0)
