"""贴图和观测的绑定：平均色要落到底色上，纹理的相对变化要留下。"""
import numpy as np
import pytest
from PIL import Image

from gwm.bench import appearance


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([90, 140, 60], dtype=np.float64)           # 偏绿的贴图
    stripes = (np.arange(64)[None, :, None] % 8 < 4) * 40.0     # 竖条纹
    return np.clip(base + stripes + rng.normal(0, 3, (64, 64, 3)), 0, 255).astype(np.uint8)


def _mean_linear(image):
    return appearance.srgb_to_linear(image.astype(np.float64) / 255).reshape(-1, 3).mean(axis=0)


def test_mean_colour_lands_on_base_colour():
    out = appearance.bind_colour(_texture(), "#c8964f")
    want = appearance.srgb_to_linear(appearance.parse_hex("#c8964f"))
    assert _mean_linear(out) == pytest.approx(want, rel=0.03)


def test_pattern_survives_and_strength_scales_it():
    image = _texture()
    full = appearance.bind_colour(image, "#808080", strength=1.0).astype(float)
    half = appearance.bind_colour(image, "#808080", strength=0.5).astype(float)
    flat = appearance.bind_colour(image, "#808080", strength=0.0).astype(float)
    bright, dark = full[:, 0:4].mean(), full[:, 4:8].mean()
    assert bright > dark + 10                                   # 条纹还在
    assert half[:, 0:4].mean() - half[:, 4:8].mean() < bright - dark
    assert flat.std() < 1.0                                     # 强度为 0 就是纯色


def test_bad_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        appearance.bind_colour(_texture(), "red")
    with pytest.raises(ValueError):
        appearance.bind_colour(_texture(), "#808080", strength=1.5)
    with pytest.raises(FileNotFoundError):
        appearance.bind_textures({"a": {"base_color": "#808080"}}, tmp_path, tmp_path / "out")


def test_bind_textures_copies_entries_without_base_colour(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    for name in ("a", "b"):
        Image.fromarray(_texture()).save(raw / f"{name}.png")
    bound = appearance.bind_textures({"a": {"base_color": "#202020"}, "b": {}}, raw, tmp_path / "out")
    assert bound == ["a"]
    assert np.array_equal(np.asarray(Image.open(tmp_path / "out" / "b.png")), _texture())
    assert np.asarray(Image.open(tmp_path / "out" / "a.png")).mean() < 60
