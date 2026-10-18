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
    """逐轮扫过：合成片段的亮度比 0.883 / 0.953 / 0.986 / 0.993 / 1.000，第四轮精确
    收敛；两轮只到 0.971，定早了。真实视频第五轮还在爬但增量很小，而代价线性增长。"""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    assert cfg["synthesis"]["colour_fit_rounds"] == 4


# ---- 反照率余量用尽时抬光照 ----

def test_没有被截断就不动光照():
    from gwm.feedback.colour import fit_lighting
    program = {"style": {}}
    assert fit_lighting(program, [{"id": "a", "gain": 1.2, "clipped_by": 1.0}]) is None
    assert program["style"] == {}, "没必要就一个字段都别碰"


def test_被截断时按中位数抬主光和环境光():
    from gwm.feedback.colour import fit_lighting
    program = {"style": {"lighting": {"key": {"intensity": 2.0},
                                      "ambient": {"intensity": 1.0}}}}
    out = fit_lighting(program, [{"id": "a", "clipped_by": 1.2},
                                 {"id": "b", "clipped_by": 1.4},
                                 {"id": "c", "clipped_by": 1.3}])
    assert out["factor"] == 1.3, "取中位数，不被个别极端值拽走"
    assert out["clipped_objects"] == 3
    assert program["style"]["lighting"]["key"]["intensity"] == 2.6
    assert program["style"]["lighting"]["ambient"]["intensity"] == 1.3


def test_一次抬太多会被挡住():
    from gwm.feedback.colour import fit_lighting
    program = {"style": {"lighting": {"key": {"intensity": 1.0}}}}
    out = fit_lighting(program, [{"id": "a", "clipped_by": 9.0}], max_step=1.5)
    assert out["factor"] == 1.5


def test_没写过光照时用内核的缺省值起步():
    from gwm.feedback.colour import fit_lighting
    program = {"style": {}}
    out = fit_lighting(program, [{"id": "a", "clipped_by": 1.2}])
    assert out["before"]["key"] == 1.6, "和 kernel/scene.js 的 DEFAULT_LIGHTING 一致"
    assert program["style"]["lighting"]["key"]["intensity"] == 1.92
