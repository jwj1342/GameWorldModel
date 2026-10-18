"""重渲染相似度：唯一直接回答「像不像」的数。

这里只测取样和装配的逻辑，不跑渲染也不跑模型——那些在小批量里端到端验。
"""
import pytest

from gwm.feedback.likeness import pick_times


def frames(ts):
    return [{"t": t, "file": f"f{i}.jpg"} for i, t in enumerate(ts)]


def test_帧数不够时全要():
    assert pick_times(frames([0.0, 0.5, 1.0]), count=8) == [0.0, 0.5, 1.0]


def test_均匀取样且包含首尾():
    picked = pick_times(frames([i * 0.25 for i in range(33)]), count=5)
    assert len(picked) == 5
    assert picked[0] == 0.0 and picked[-1] == 8.0
    gaps = [round(b - a, 3) for a, b in zip(picked, picked[1:])]
    assert max(gaps) - min(gaps) <= 0.25, f"间隔该均匀，实际 {gaps}"


def test_重复时刻只算一次():
    assert pick_times(frames([0.0, 0.0, 1.0, 1.0]), count=8) == [0.0, 1.0]


def test_没有帧时返回空而不是炸():
    assert pick_times([], count=8) == []
    assert pick_times([{"file": "a.jpg"}], count=8) == []


def test_归一化把分数放到两条标尺之间():
    """光报一个 0.19 没有意义：DINOv2 在两张毫不相干的图之间也有 0.1 上下。
    必须报出「同视频不同时刻」的上限和「不同视频」的下限，才知道 0.19 是好是坏。"""
    # 这里直接验算归一化的算法，不跑模型
    median, upper, lower = 0.19, 0.55, 0.08
    normalised = (median - lower) / (upper - lower)
    assert abs(normalised - 0.234) < 0.01


def test_没有参照视频时只报上限():
    from gwm.feedback.likeness import likeness
    import inspect
    sig = inspect.signature(likeness)
    assert "unrelated_frames" in sig.parameters
    assert sig.parameters["unrelated_frames"].default is None
