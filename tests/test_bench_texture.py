"""贴图生成里不需要模型的那部分：名字校验、提示词模板、跳过逻辑、接缝度量。

这里一个模型都不加载。出图那一步换成假的 `renderer_factory`，它只数自己被调了几次、
按种子给一张确定的图，所以「该画的画了、该跳的跳了」可以在毫秒级验证。
环绕补边那一条用一个两层的小卷积网络直接验证 torch 这个版本上确实成立，不靠假设。
"""
import json

import numpy as np
import pytest
from PIL import Image

from gwm.bench import texture
from gwm.bench.texture import (TextureError, build_prompt, generate_textures, model_defaults,
                               seam_error, tile, validate_name)


class FakeRenderer:
    """假的出图函数：记下每次调用，按种子给一张确定的噪声图。"""
    device = "fake"
    revision = "rev0"

    def __init__(self):
        self.calls = []

    def __call__(self, *, prompt, negative_prompt, seed, size, steps, guidance):
        self.calls.append({"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed,
                           "size": size, "steps": steps, "guidance": guidance})
        rng = np.random.default_rng(seed)
        return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8), "RGB")


class Factory:
    """数一数「加载模型」发生了几次。"""

    def __init__(self):
        self.loads = 0
        self.renderer = FakeRenderer()

    def __call__(self):
        self.loads += 1
        return self.renderer


MATERIALS = {"oak_floor": {"prompt": "light oak wood floor planks", "seed": 7, "roughness": 0.6},
             "white_plaster": {"prompt": "white painted plaster wall", "seed": 11}}


def run(tmp_path, materials=MATERIALS, factory=None, **kwargs):
    factory = factory or Factory()
    kwargs.setdefault("model", "fake/turbo")
    kwargs.setdefault("size", 64)
    return generate_textures(materials, tmp_path, renderer_factory=factory, **kwargs), factory


# ---------------------------------------------------------------- 名字

@pytest.mark.parametrize("name", ["oak_floor", "a", "wall_2", "0"])
def test_合法的名字原样通过(name):
    assert validate_name(name) == name


@pytest.mark.parametrize("name", ["Oak", "oak floor", "oak-floor", "../etc", "a/b", "", "木", "a.png", 3, None])
def test_不合法的名字报错并说明规则(name):
    with pytest.raises(TextureError, match=r"a-z0-9_"):
        validate_name(name)


def test_坏名字在动手生成之前就被拦下(tmp_path):
    factory = Factory()
    bad = {"good_one": {"prompt": "x", "seed": 1}, "../escape": {"prompt": "x", "seed": 2}}
    with pytest.raises(TextureError):
        run(tmp_path, bad, factory)
    assert factory.loads == 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("entry", [{"prompt": "", "seed": 1}, {"prompt": "x"}, {"prompt": "x", "seed": "7"},
                                   {"prompt": "x", "seed": True}, {"prompt": "x", "seed": -1}, "x"])
def test_条目缺提示词或种子不对时报错(tmp_path, entry):
    with pytest.raises(TextureError, match="mat"):
        run(tmp_path, {"mat": entry})


# ---------------------------------------------------------------- 提示词

def test_提示词套进模板并且是平铺样片的写法():
    full = build_prompt("light oak wood floor planks")
    assert full == texture.PROMPT_TEMPLATE.format(prompt="light oak wood floor planks")
    assert "light oak wood floor planks" in full
    assert "seamless" in full and "top-down" in full


def test_提示词首尾的空白和标点被整理掉():
    assert build_prompt("  grey   woven\nfabric. ") == build_prompt("grey woven fabric")
    with pytest.raises(TextureError):
        build_prompt("  , ")


def test_蒸馏模型不用引导也就不带负面提示词(tmp_path):
    assert model_defaults(texture.DEFAULT_MODEL) == {"steps": 2, "guidance": 0.0}
    assert model_defaults("stable-diffusion-v1-5/stable-diffusion-v1-5")["guidance"] > 1.0

    log, factory = run(tmp_path / "turbo", model="fake/turbo")
    assert factory.renderer.calls[0]["negative_prompt"] is None
    assert log["textures"][0]["negative_prompt"] is None

    log, factory = run(tmp_path / "plain", model="fake/plain")
    assert factory.renderer.calls[0]["negative_prompt"] == texture.NEGATIVE_PROMPT
    assert log["textures"][0]["negative_prompt"] == texture.NEGATIVE_PROMPT


# ---------------------------------------------------------------- 生成与跳过

def test_每条写一张图并在日志里记全出图条件(tmp_path):
    log, factory = run(tmp_path)
    assert log["generated"] == ["oak_floor", "white_plaster"] and log["skipped"] == []
    for name in MATERIALS:
        with Image.open(tmp_path / f"{name}.png") as image:
            assert image.mode == "RGB" and image.size == (64, 64)
    on_disk = json.loads((tmp_path / "textures.json").read_text(encoding="utf-8"))
    assert on_disk["textures"] == log["textures"]
    record = on_disk["textures"][0]
    assert record["name"] == "oak_floor" and record["seed"] == 7
    assert record["full_prompt"] == build_prompt("light oak wood floor planks")
    assert record["model"] == "fake/turbo" and record["revision"] == "rev0"
    assert record["steps"] == 2 and record["guidance"] == 0.0 and record["size"] == 64
    assert record["device"] == "fake" and record["seconds"] >= 0
    assert {"ratio_x", "ratio_y"} <= set(record["seam"])
    # 发给模型的就是日志里记的那一句
    assert factory.renderer.calls[0]["prompt"] == record["full_prompt"]


def test_重跑时全部跳过而且不加载模型(tmp_path):
    first, _ = run(tmp_path)
    before = {name: (tmp_path / f"{name}.png").read_bytes() for name in MATERIALS}
    factory = Factory()
    second, _ = run(tmp_path, factory=factory)
    assert second["generated"] == [] and second["skipped"] == list(MATERIALS)
    assert factory.loads == 0 and factory.renderer.calls == []
    assert second["textures"] == first["textures"]
    assert before == {name: (tmp_path / f"{name}.png").read_bytes() for name in MATERIALS}


def test_图被删掉后只重画那一张而且逐字节相同(tmp_path):
    run(tmp_path)
    target = tmp_path / "oak_floor.png"
    original = target.read_bytes()
    target.unlink()
    log, factory = run(tmp_path)
    assert log["generated"] == ["oak_floor"] and log["skipped"] == ["white_plaster"]
    assert factory.loads == 1 and len(factory.renderer.calls) == 1
    assert target.read_bytes() == original


@pytest.mark.parametrize("change", [
    {"materials": {**MATERIALS, "oak_floor": {"prompt": "dark walnut planks", "seed": 7}}},
    {"materials": {**MATERIALS, "oak_floor": {"prompt": "light oak wood floor planks", "seed": 8}}},
    {"model": "fake/other-turbo"},
    {"steps": 3},
    {"size": 128},
])
def test_提示词种子模型步数尺寸任何一个变了都要重画(tmp_path, change):
    run(tmp_path)
    log, _ = run(tmp_path, **change)
    assert "oak_floor" in log["generated"]


def test_materials里多余的键不影响跳过(tmp_path):
    run(tmp_path)
    extra = {name: {**entry, "roughness": 0.1, "note": "x"} for name, entry in MATERIALS.items()}
    log, factory = run(tmp_path, extra)
    assert log["generated"] == [] and factory.loads == 0


def test_日志坏掉时当作没有全部重画(tmp_path):
    run(tmp_path)
    (tmp_path / "textures.json").write_text("{ not json", encoding="utf-8")
    log, _ = run(tmp_path)
    assert log["generated"] == list(MATERIALS)


def test_删掉的材质不留在日志里(tmp_path):
    run(tmp_path)
    log, _ = run(tmp_path, {"white_plaster": MATERIALS["white_plaster"]})
    assert [r["name"] for r in log["textures"]] == ["white_plaster"] and log["generated"] == []


def test_尺寸必须是64的倍数(tmp_path):
    with pytest.raises(TextureError, match="64"):
        run(tmp_path, size=500)


def test_纯色图被当成出错而不是写出去(tmp_path):
    class Black(FakeRenderer):
        def __call__(self, *, size, **_):
            return Image.new("RGB", (size, size))

    factory = Factory()
    factory.renderer = Black()
    with pytest.raises(RuntimeError, match="纯色"):
        run(tmp_path, factory=factory)
    assert not (tmp_path / "oak_floor.png").exists()


# ---------------------------------------------------------------- 接缝

def periodic_image(size=64):
    """两个方向上都整周期的正弦图案：天生可以平铺。"""
    t = np.arange(size) * 2 * np.pi / size
    wave = np.sin(3 * t)[None, :] + np.cos(2 * t)[:, None]
    return np.repeat(((wave + 2) * 60)[:, :, None], 3, axis=2)


def test_周期图案的接缝和内部差是同一个量级():
    seam = seam_error(periodic_image())
    # 接缝不比内部更跳就行；余弦在接缝处正好是平的，比值可以远小于 1
    assert seam["ratio_x"] < 2 and seam["ratio_y"] < 2
    assert seam["seam_x"] <= seam["interior_x_max"] + 1e-3
    assert seam["seam_y"] <= seam["interior_y_max"] + 1e-3
    # 接缝差远小于「相隔很远的两列」的差：是真的接上了，不是碰巧像
    assert seam["seam_x"] < 0.5 * seam["unrelated_x"] and seam["seam_y"] < 0.5 * seam["unrelated_y"]


def test_渐变图的接缝远大于内部差而且只在那个方向():
    ramp = np.tile(np.linspace(0, 255, 64)[None, :, None], (64, 1, 3))
    seam = seam_error(ramp)
    assert seam["ratio_x"] > 30          # 左边黑、右边白，首尾接不上
    assert seam["seam_x"] > seam["unrelated_x"]   # 比「毫不相干的两列」还差
    assert seam["seam_y"] == 0 and seam["ratio_y"] == 1.0   # 竖直方向每行都一样


def test_接缝度量按定义算():
    image = np.array([[0, 10, 30], [0, 10, 30]], dtype=np.uint8)   # uint8 也不能减出溢出
    seam = seam_error(image)
    assert seam["seam_x"] == 30 and seam["interior_x"] == 15 and seam["interior_x_max"] == 20
    assert seam["ratio_x"] == 2.0
    # 三列的图，错开一列或两列，平均差都是 20
    assert seam["unrelated_x"] == 20


def test_平铺预览是原图横竖重复():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    tiled = tile(image)
    assert tiled.shape == (4, 6, 3)
    assert (tiled[:2, :3] == image).all() and (tiled[2:, 3:] == image).all()


# ---------------------------------------------------------------- 环绕补边

def test_环绕补边之后卷积网络在环面上平移等变():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.Conv2d(8, 8, 3, stride=2, padding=1),     # 下采样那一种
        torch.nn.Conv2d(8, 4, 1),
    )
    x = torch.randn(1, 3, 16, 16)
    shifted = torch.roll(x, shifts=(4, 6), dims=(2, 3))
    # 零补边时不成立，否则这条测试什么也没测
    assert not torch.allclose(net(shifted), torch.roll(net(x), shifts=(2, 3), dims=(2, 3)), atol=1e-5)

    assert texture.make_circular(net) == 3
    assert torch.allclose(net(shifted), torch.roll(net(x), shifts=(2, 3), dims=(2, 3)), atol=1e-5)
    # 和一开始就按环绕补边构造的卷积结果一致
    fresh = torch.nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular")
    fresh.load_state_dict(net[0].state_dict())
    assert torch.allclose(fresh(x), net[0](x), atol=1e-6)
