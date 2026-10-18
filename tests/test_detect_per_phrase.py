"""逐短语检测时的跨短语合并。

拼串检测的毛病：一个框落在多个短语的词上时，检测器返回的是拼接（#67 修的词片碎屑
就是这么来的），而且短语之间会争夺同一个框。逐短语分别问能绕开这两点，代价是每帧
多跑 N 次前向。

但逐短语会引入一个新问题：同一个物体可能被好几个短语找到（crate 和 box），不合并
就变成重复检出。这些用例盯的是合并这一步。
"""
from gwm.perception.grounded_sam2_backend import GroundingDinoDetectionProvider as P


def test_重叠的框只留得分最高的():
    iou = P._iou([0, 0, 10, 10], [1, 1, 11, 11])
    assert iou > 0.6, f"这两个框应该算重叠，实际 {iou:.3f}"


def test_不重叠的框互不影响():
    assert P._iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_完全重合是一():
    assert P._iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_包含关系也算重叠():
    # 大框套小框：交集等于小框，按并集算 0.25
    assert abs(P._iou([0, 0, 10, 10], [0, 0, 5, 5]) - 0.25) < 1e-9


def test_零面积的框不会把分母变成零():
    assert P._iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


def test_开关的缺省值是关着的():
    """效果还没量过。过检那 62% 七个判别子全排除之后，换检测是仅剩方向之一，
    但在量出来之前不该默认开——它每帧多跑 N 次前向。"""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    assert cfg["perception"]["detect_per_phrase"] is False
