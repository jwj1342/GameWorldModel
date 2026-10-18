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


# ---- 按残差调颜色（#55 的第三步）----

def _program():
    return {"objects": [{"id": "a", "class": "box", "material": "#404040"},
                        {"id": "b", "class": "box",
                         "material": {"base_color": "#808080", "roughness": 0.3}}]}


def _residual(**ratios):
    return {"ok": True, "per_object": {k: {"luma_ratio": v, "channel_error": 10.0}
                                       for k, v in ratios.items()}}


def test_渲得暗就把材质调亮():
    from gwm.feedback.colour import fit_colours
    program = _program()
    rows = fit_colours(program, _residual(a=0.5))
    assert len(rows) == 1 and rows[0]["id"] == "a"
    assert rows[0]["gain"] == 2.0
    assert program["objects"][0]["material"] == "#808080"


def test_材质是对象时只换颜色不碰别的字段():
    from gwm.feedback.colour import fit_colours
    program = _program()
    fit_colours(program, _residual(b=0.5))
    material = program["objects"][1]["material"]
    assert material["base_color"] == "#ffffff"
    assert material["roughness"] == 0.3, "别的表面参数不能被覆盖掉"


def test_阻尼让它只走一部分():
    from gwm.feedback.colour import fit_colours
    program = _program()
    rows = fit_colours(program, _residual(a=0.5), damping=0.5)
    assert rows[0]["gain"] == 1.5, "一轮到位容易过冲，增益本身是从带噪声的测量估来的"


def test_离谱的增益被挡住():
    from gwm.feedback.colour import fit_colours
    program = _program()
    rows = fit_colours(program, _residual(a=0.001))
    assert rows[0]["gain"] == 4.0, "渲染里只占几个像素的物体，比值可能毫无意义"


def test_没算成的残差不动任何东西():
    from gwm.feedback.colour import fit_colours
    program = _program()
    assert fit_colours(program, {"ok": False}) == []
    assert program["objects"][0]["material"] == "#404040"


def test_拟合轮数的缺省值进了配置():
    """十条合成片段实测：每通道误差 19.3 -> 4.6 /255，亮度比 0.777 -> 0.971。
    代价是每条多约 12 秒。两轮之后已经很接近 1.0，再加轮数收益递减。"""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    assert cfg["synthesis"]["colour_fit_rounds"] == 2
