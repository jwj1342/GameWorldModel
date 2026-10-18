"""场景完整度指标。它存在的理由是：旧的反馈分数只对配对上的物体求平均，
场景里少了三十个物体一分都不扣——所以重建出五个盒子也能一路通过门禁。"""
import pytest

from gwm.feedback.completeness import assess, fidelity, interactivity


def ev_object(oid, size):
    return {"id": oid, "obb": [{"size": list(size), "center": [0, 0, 0], "quat": [0, 0, 0, 1], "t": 0.0}]}


def node(oid, extent, motion=None, **extra):
    out = {"id": oid, "class": oid, "pose": {"pos": [0, 0, 0]},
           "geom": {"kind": "primitive", "shape": "box", "extent": list(extent)}, **extra}
    if motion:
        out["motion"] = motion
    return out


def test_漏掉的物体会被召回率反映出来():
    """这正是旧分数看不见的那件事。"""
    evidence = {"objects": [ev_object(f"o{i}", [1, 1, 1]) for i in range(10)]}
    program = {"objects": [node("o0", [1, 1, 1]), node("o1", [1, 1, 1])], "static": []}
    assert fidelity(program, evidence)["object_recall"] == pytest.approx(0.2)


def test_尺寸误差用对数所以放大一倍和缩小一半一样严重():
    evidence = {"objects": [ev_object("a", [1.0, 1, 1]), ev_object("b", [1.0, 1, 1])]}
    program = {"objects": [node("a", [2.0, 1, 1]), node("b", [0.5, 1, 1])], "static": []}
    report = fidelity(program, evidence)
    assert report["size_log_error_median"] == pytest.approx(abs(__import__("math").log(2)), abs=1e-3)


def test_资产覆盖率只认真网格():
    manifest = {"requests": [{"resolved": "library"}, {"resolved": "library_ref"},
                             {"resolved": "primitive_fallback"}, {"resolved": "primitive_fallback"}]}
    report = fidelity({"objects": [], "static": []}, None, manifest)
    assert report["asset_coverage"] == pytest.approx(0.5)
    assert report["asset_requests"] == 4


def test_可交互四档各自独立():
    program = {"objects": [
        node("a", [1, 1, 1]),                                            # 可拖，无行为
        node("b", [1, 1, 1], {"type": "revolute", "pivot": [0, 0, 0], "axis": [0, 1, 0],
                              "rate_dps": 30}),                          # 关节
        node("c", [1, 1, 1], {"type": "dynamic", "mass": 1}),            # 物理驱动
        node("d", [1, 1, 1], {"type": "trajectory"}, interaction={"grab": False}),  # 不可拖
    ]}
    report = interactivity(program)
    assert report["manipulable_fraction"] == pytest.approx(0.75)
    assert report["articulated_fraction"] == pytest.approx(0.25)
    assert report["simulated_fraction"] == pytest.approx(0.25)
    assert report["scripted_motion_fraction"] == pytest.approx(0.5)   # revolute 和 trajectory


def test_两轴不合成一个总分():
    """合成之后「什么都没重建出来但剩下那个能拖」会拿到不低的分，
    而那正是现在最该避免的失败模式。"""
    report = assess({"objects": [node("a", [1, 1, 1])], "static": []})
    assert set(report) == {"fidelity", "interactivity"}
    assert "score" not in report and "overall" not in report


def test_空场景不炸():
    report = assess({"objects": [], "static": []}, {"objects": []})
    assert report["interactivity"] == {"objects": 0}
    assert report["fidelity"]["object_recall"] is None


def test_画面覆盖率补上了召回率看不见的那一面():
    """一个塞满东西的房间只检出九个物体、九个全都重建了出来，召回率是满分，
    但画面上九成都没被解释。这一项就是为了把那九成报出来。"""
    import numpy as np
    small = np.zeros((10, 10), dtype=bool); small[0:2, 0:5] = True      # 占 10%
    masks = {"a": {0: small}}
    evidence = {"objects": [ev_object("a", [1, 1, 1])]}
    program = {"objects": [node("a", [1, 1, 1])], "static": []}
    report = fidelity(program, evidence, None, masks)
    assert report["object_recall"] == pytest.approx(1.0), "按证据算是满分"
    assert report["evidence_pixel_coverage"] == pytest.approx(0.10), "按画面算只有一成"


def test_多个物体的覆盖率取并集而不是相加():
    import numpy as np
    a = np.zeros((10, 10), dtype=bool); a[0:5, :] = True
    b = np.zeros((10, 10), dtype=bool); b[3:8, :] = True        # 和 a 有重叠
    report = fidelity({"objects": [], "static": []}, None, None, {"a": {0: a}, "b": {0: b}})
    assert report["evidence_pixel_coverage"] == pytest.approx(0.8)


def test_覆盖率按帧平均而不是只看一帧():
    """多关键帧检测出来的物体，掩码从它被检到的那一帧才开始有。只统计第一帧的话，
    物体数从 9 涨到 62 这个指标都不会动——实测踩过这个坑。"""
    import numpy as np
    early = np.zeros((10, 10), dtype=bool); early[0:2, :] = True      # 第 0 帧占 20%
    late = np.zeros((10, 10), dtype=bool); late[0:8, :] = True        # 第 5 帧占 80%
    report = fidelity({"objects": [], "static": []}, None, None,
                      {"a": {0: early}, "b": {5: late}})
    assert report["evidence_pixel_coverage"] == pytest.approx(0.5)    # (0.2 + 0.8) / 2
    assert report["evidence_pixel_coverage_best"] == pytest.approx(0.8)
    assert report["covered_frames"] == 2


# ---- 同类尺寸离谱的诊断（#64，只报不改）----

def test_同类里尺寸离谱的被标出来():
    from gwm.feedback.completeness import size_outliers
    objects = [{"id": f"b{i}", "class": "block",
                "geom": {"extent": [0.3, 0.3, 0.3]}} for i in range(4)]
    objects.append({"id": "huge", "class": "block", "geom": {"extent": [4.0, 0.3, 0.3]}})
    found = size_outliers(objects)
    assert [r["id"] for r in found] == ["huge"]
    assert found[0]["ratio"] > 3.0


def test_同类实例太少就不判():
    """只有两个实例且其中一个是错的时候，中位数落在两者中间，倍数判据会把好的
    那个一起冤枉。"""
    from gwm.feedback.completeness import size_outliers
    objects = [{"id": "a", "class": "block", "geom": {"extent": [0.3, 0.3, 0.3]}},
               {"id": "b", "class": "block", "geom": {"extent": [4.0, 0.3, 0.3]}}]
    assert size_outliers(objects) == []


def test_太小的也算离谱():
    from gwm.feedback.completeness import size_outliers
    objects = [{"id": f"b{i}", "class": "block",
                "geom": {"extent": [1.0, 1.0, 1.0]}} for i in range(4)]
    objects.append({"id": "tiny", "class": "block", "geom": {"extent": [0.05, 0.05, 0.05]}})
    found = size_outliers(objects)
    assert [r["id"] for r in found] == ["tiny"]
    assert found[0]["ratio"] < 0.34


def test_不同类之间不互相比():
    from gwm.feedback.completeness import size_outliers
    objects = [{"id": f"b{i}", "class": "block",
                "geom": {"extent": [0.3, 0.3, 0.3]}} for i in range(3)]
    objects += [{"id": f"w{i}", "class": "wall",
                 "geom": {"extent": [4.0, 3.0, 0.2]}} for i in range(3)]
    assert size_outliers(objects) == [], "墙本来就比方块大，不该因此被标出来"
