"""`--stage-scale` 必须把**轨迹关键帧**也缩了。

漏掉的后果不是「小一点」，是**自相矛盾**：`pose.pos` 被缩了、`motion.keyframes`
没缩，同一个物体的起点和它轨迹的第一帧差着 1/k 倍。这样的真值物体**在真值里就是
坏的**——预测再准也配不上，只是白占召回的分母。

实测（2026-10-19）：stage_scale=0.15 的三批数据里，所有带轨迹关键帧的物体都中招
——ab015 5 个、hold015 2 个、pair015 3 个。ab015 有 59 个真值 objects，
也就是 8.5% 的召回分母是不可能被配上的。

根因是 `scale_program` 的那层循环只扫到 `motion` 这个字典本身，而位置藏在
`keyframes` 这串字典的每一个里面。相机那边是对的（单独扫了 `camera.keyframes`），
`geom.parts` 和 `instances` 也都单独扫了，唯独物体的运动关键帧漏了。
"""
import pytest

from gwm.dataset.framing import scale_program


def program():
    return {
        "objects": [{
            "id": "obj_1",
            "pose": {"pos": [10.0, 2.0, -4.0]},
            "geom": {"kind": "primitive", "shape": "box", "extent": [2.0, 1.0, 3.0]},
            "motion": {"type": "trajectory", "keyframes": [
                {"t": 0.0, "pos": [10.0, 2.0, -4.0]},
                {"t": 1.0, "pos": [20.0, 2.0, -8.0]},
            ]},
        }],
        "static": [],
        "camera": {"keyframes": [{"t": 0.0, "pos": [0.0, 7.4, 12.0]}]},
    }


def test_关键帧跟着缩():
    out = scale_program(program(), 0.15)
    kfs = out["objects"][0]["motion"]["keyframes"]
    assert kfs[0]["pos"] == pytest.approx([1.5, 0.3, -0.6])
    assert kfs[1]["pos"] == pytest.approx([3.0, 0.3, -1.2])


def test_起点和第一个关键帧缩完仍然一致():
    """这是这条 bug 的本质：两者缩之前是一致的，缩完必须还一致。"""
    out = scale_program(program(), 0.15)
    node = out["objects"][0]
    assert node["pose"]["pos"] == pytest.approx(node["motion"]["keyframes"][0]["pos"])


def test_时间不缩():
    """只缩带长度量纲的量。时间、角度、质量都不动。"""
    out = scale_program(program(), 0.15)
    kfs = out["objects"][0]["motion"]["keyframes"]
    assert kfs[0]["t"] == 0.0 and kfs[1]["t"] == 1.0


def test_k为一时原样返回():
    out = scale_program(program(), 1.0)
    assert out["objects"][0]["motion"]["keyframes"][1]["pos"] == [20.0, 2.0, -8.0]


def test_没有关键帧的运动不受影响():
    p = program()
    p["objects"][0]["motion"] = {"type": "periodic_translate", "amplitude": [4.0, 0, 0],
                                 "period_s": 2.0}
    out = scale_program(p, 0.15)
    m = out["objects"][0]["motion"]
    assert m["amplitude"] == pytest.approx([0.6, 0, 0]), "amplitude 本来就在扫描范围里"
    assert m["period_s"] == 2.0, "周期是时间，不缩"
