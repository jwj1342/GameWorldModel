"""样式那一半的验收指标。

重渲染相似度几乎看不见颜色——把场景里每个物体的色相旋转半圈，六条片段的相似度
中位数只从 0.5635 变到 0.5568，两条反而涨了。所以样式恢复得准不准需要另一个数。

这里量的是闭环残差：渲出来的颜色和视频里量到的颜色差多少，两边用同一种方法量
（掩码内逐通道中位数）。这些用例盯的是口径本身，渲染那一段由集成测试覆盖。
"""
import numpy as np

from gwm.feedback.colour import _hex_to_rgb, _median_colour, colour_residual


def test_颜色解析():
    assert _hex_to_rgb("#804020") == (0x80, 0x40, 0x20)
    assert _hex_to_rgb("804020") == (0x80, 0x40, 0x20)
    assert _hex_to_rgb("#80402") is None
    assert _hex_to_rgb("") is None
    assert _hex_to_rgb("#zzzzzz") is None


def test_掩码内取中位数而不是均值():
    # 均值会被少数离群像素拽走，中位数不会。感知侧也是取中位数，口径必须一致
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:, :] = (100, 100, 100)
    rgb[0, 0] = (255, 255, 255)          # 一个高光
    mask = np.ones((4, 4), bool)
    assert _median_colour(rgb, mask, min_pixels=4) == (100, 100, 100)


def test_像素太少就不报():
    rgb = np.zeros((4, 4, 3), np.uint8)
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    assert _median_colour(rgb, mask, min_pixels=16) is None


def test_形状对不上也不报():
    rgb = np.zeros((4, 4, 3), np.uint8)
    assert _median_colour(rgb, np.ones((8, 8), bool), min_pixels=1) is None


def test_证据里没有观测颜色时如实说明(tmp_path):
    out = colour_residual(tmp_path, {"objects": []}, {"objects": [{"id": "a"}]},
                          tmp_path / "r", [0.0])
    assert out["ok"] is False
    assert "观测颜色" in out["reason"]
