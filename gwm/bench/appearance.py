"""把生成的贴图和观测绑在一起：纹理来自生成模型，颜色来自照片。

只凭一句文字，生成模型画得出「木地板的纹理」，画不准「这个房间里这块地板的颜色」——
同一句 "cream painted wall" 出来可能是灰的。而颜色恰恰是最容易从照片里量出来的东西：
智能体在 materials.json 里给每种表面写了一个 `base_color`，就是它对着照片估的平均色。

所以这里把两者分工：

- 生成的贴图只留**相对变化**：每个像素除以整张图的平均色，得到一张围绕 1 起伏的细节图。
- 平均色换成 `base_color`。

这样贴图渲染和纯色渲染在大面积上的颜色是一致的，贴图只负责加上纯色给不出来的那层细节。
`strength` 控制细节保留多少：1 是原样，0 就退化成纯色。生成模型偶尔会画出对比过强、
像微距特写的图，调低一点能压住。

运算在线性光下做（先把 sRGB 解码），不然暗部和亮部的比例会被伽马扭曲。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def parse_hex(colour: str) -> np.ndarray:
    if not isinstance(colour, str) or not _HEX.match(colour):
        raise ValueError(f"base_color 要写成 #rrggbb：{colour!r}")
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64) / 255.0


def bind_colour(image: np.ndarray, base_colour: str, strength: float = 1.0) -> np.ndarray:
    """image 是 HxWx3 的 uint8（sRGB）。返回同样形状的 uint8，平均色贴近 base_colour。"""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength 要在 0 到 1 之间")
    linear = srgb_to_linear(image.astype(np.float64) / 255.0)
    mean = linear.reshape(-1, 3).mean(axis=0)
    # 用亮度的相对变化，不用逐通道的：逐通道相除会把贴图自己的偏色也当成细节留下来
    luma = linear @ np.array([0.2126, 0.7152, 0.0722])
    relative = luma / max(float(mean @ np.array([0.2126, 0.7152, 0.0722])), 1e-6)
    detail = 1.0 + strength * (relative - 1.0)
    out = detail[..., None] * srgb_to_linear(parse_hex(base_colour))[None, None, :]
    return np.round(linear_to_srgb(out) * 255.0).astype(np.uint8)


def bind_textures(materials: dict, raw_dir: Path, out_dir: Path, strength: float = 1.0) -> list[str]:
    """对 materials 里每一条，把 raw_dir/<名字>.png 绑上它的 base_color，写到 out_dir。

    没写 base_color 的条目原样拷过去。返回实际改过颜色的材质名。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bound = []
    for name, spec in materials.items():
        source = raw_dir / f"{name}.png"
        if not source.is_file():
            raise FileNotFoundError(f"缺生成的贴图 {source}")
        image = np.asarray(Image.open(source).convert("RGB"))
        if spec.get("base_color"):
            image = bind_colour(image, spec["base_color"], strength)
            bound.append(name)
        Image.fromarray(image).save(out_dir / f"{name}.png")
    return bound
