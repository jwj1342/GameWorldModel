"""粗对齐的启动：整体旋转、漏检、假物体。

规范对齐本身是对的，但它得先建立起一批匹配才能解析求解。启动这一步原来用两边
质心之差，这有两个毛病——整体转了九十度时光平移永远够不着阈值，一对都配不上；
多一个假物体或少一个真物体，质心就被拽走，原本对得上的也散了。后一种在完美感知
下跑真实场景实测到过：十二个本该进 static 的结构落进 objects，对齐被拽偏 1.4 米。
"""
import numpy as np
import pytest

from gwm.feedback.gt_metrics import coarse_gauge, evaluate


def program(ids):
    return {"objects": [{"id": i, "class": "x", "geom": {"kind": "primitive", "shape": "box", "extent": [1, 1, 1]},
                         "motion": {"type": "trajectory"}} for i in ids]}


def states(positions_at, ids, ts):
    return [{"t": float(t), "objects": [{"id": i, "name": i, "kind": "object", "class": "x",
                                         "pos": [float(v) for v in positions_at(i, t)]} for i in ids]}
            for t in ts]


# 三个不对称摆放的物体，整段静止，便于把「哪一步出错」隔离开
LAYOUT = {"a": np.array([0.0, 1.0, 0.0]), "b": np.array([6.0, 1.0, 0.0]), "c": np.array([0.0, 1.0, 4.0])}
IDS = tuple(LAYOUT)
TS = [i / 10 for i in range(81)]          # 0 ~ 8.0 秒


def test_a_ninety_degree_global_rotation_still_starts_up():
    """整体转了九十度，光靠平移凑不到三米阈值以内，一对都匹配不上，
    估偏航那一步根本启动不了，最后召回是 0。"""
    yaw = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    gt = states(lambda i, t: LAYOUT[i], IDS, TS)
    pred = states(lambda i, t: yaw @ LAYOUT[i], IDS, TS)
    r = evaluate(program(IDS), program(IDS), gt, pred)
    assert r["full"]["objects"]["recall"] == 1.0
    assert r["full"]["trajectory"]["median_m"] == pytest.approx(0.0, abs=1e-6)


def test_a_spurious_distant_object_does_not_break_the_existing_matches():
    """多出来一个很远的假物体会把质心拽走，原本对得上的三个就都对不上了。
    这正是完美感知那次实测里发生的事，只是那次多了十二个。"""
    gt = states(lambda i, t: LAYOUT[i], IDS, TS)
    extra = {**LAYOUT, "ghost": np.array([200.0, 1.0, 200.0])}
    pred = states(lambda i, t: extra[i], tuple(extra), TS)
    r = evaluate(program(IDS), program(tuple(extra)), gt, pred)
    assert r["full"]["objects"]["matched"] == 3
    assert r["full"]["objects"]["recall"] == 1.0
    assert r["full"]["objects"]["precision"] == pytest.approx(0.75)


def test_a_missing_object_does_not_break_the_remaining_matches():
    """少检出一个真物体同样会拽走质心。剩下两个该照样对得上，召回如实报 2/3。"""
    gt = states(lambda i, t: LAYOUT[i], IDS, TS)
    pred = states(lambda i, t: LAYOUT[i], ("a", "b"), TS)
    r = evaluate(program(IDS), program(("a", "b")), gt, pred)
    assert r["full"]["objects"]["matched"] == 2
    assert r["full"]["objects"]["recall"] == pytest.approx(2 / 3)
    assert r["full"]["trajectory"]["median_m"] == pytest.approx(0.0, abs=1e-6)


def test_an_unobservable_global_offset_is_still_absorbed():
    """整体平移是观测不到的，照样要对齐掉——换了启动方式不能把原来的能力弄丢。"""
    gt = states(lambda i, t: LAYOUT[i], IDS, TS)
    pred = states(lambda i, t: LAYOUT[i] + np.array([8.3, 0.0, -5.0]), IDS, TS)
    r = evaluate(program(IDS), program(IDS), gt, pred)
    assert r["full"]["trajectory"]["median_m"] == pytest.approx(0.0, abs=1e-6)


def test_coarse_gauge_votes_instead_of_averaging():
    """直接测粗对齐这一层：三个点对得上，一个离群点不该左右结果。"""
    gt = np.array([[0.0, 1, 0.0], [6.0, 1, 0.0], [0.0, 1, 4.0]])
    pred = np.vstack([gt + np.array([10.0, 0, 0]), [[500.0, 1, 500.0]]])
    R, t, info = coarse_gauge(gt, pred, 3.0)
    assert info["inliers"] == 3
    assert np.allclose(R, np.eye(3))
    assert t == pytest.approx([-10.0, 0.0, 0.0], abs=1e-6)


def test_the_report_records_how_the_alignment_started():
    """对齐出问题的时候要能回溯是粗对齐没起来还是精对齐偏了。"""
    gt = states(lambda i, t: LAYOUT[i], IDS, TS)
    pred = states(lambda i, t: LAYOUT[i] + np.array([8.3, 0.0, -5.0]), IDS, TS)
    coarse = evaluate(program(IDS), program(IDS), gt, pred)["full"]["gauge_alignment"]["coarse"]
    assert coarse["inliers"] == 3 and coarse["candidates"] == 4 * 3 * 3
