"""样式那一半的验收指标：渲出来的颜色和视频里量到的颜色差多少。

为什么需要它。重渲染相似度（`likeness.py`，DINOv2 嵌入）**几乎看不见颜色**——实测把
场景里每个物体的色相旋转半圈，六条片段的相似度中位数只从 0.5635 变到 0.5568，其中
两条反而涨了，幅度和片段间噪声同量级。所以样式那一半目前没有能验收的数：#52 要的
「样式恢复得准不准」答不上来，#55 的渲染反馈闭环做出来也说不出它起没起作用。

这里补的就是那个数。口径是闭环的：**两边用同一种方法量**——掩码内像素逐通道取中位数。
观测侧的掩码来自分割，渲染侧的掩码来自内核的 ID 通道，两者都是「这个物体占的像素」。

量的是残差而不是绝对正确性，这是有意的：真实视频里没有材质真值，唯一能自洽闭合的
问题是「按这份程序渲出来，和输入视频里看到的一样吗」。合成数据上另有真值可比，
那是 `probe_colour_headroom.py` 的事。
"""
from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np

from ..compiler.ids import decode_id_png, registry_order
from .render import render


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _median_colour(rgb: np.ndarray, mask: np.ndarray, min_pixels: int) -> tuple[int, int, int] | None:
    """掩码内逐通道取中位数。和感知侧 observe_colors 的口径一致。"""
    if mask.shape != rgb.shape[:2] or int(mask.sum()) < min_pixels:
        return None
    pixels = rgb[mask]
    return tuple(int(np.median(pixels[:, c])) for c in range(3))


def colour_residual(game_dir: Path, program: dict, evidence: dict, out_dir: Path,
                    times: list[float], width: int = 640, height: int = 360,
                    min_pixels: int = 16) -> dict:
    """渲几帧，逐物体比「渲出来的颜色」和「视频里量到的颜色」。

    返回每通道绝对误差的中位数（0 到 255）、亮度比值的中位数，以及参与比较的物体数。
    亮度比值小于 1 说明渲出来比视频暗——那是把像素值当反照率用的典型症状：像素是
    反照率乘光照，再渲一次等于又乘了一遍。
    """
    from ..taxonomy import slug
    observed: dict[str, tuple[int, int, int]] = {}
    for obj in evidence.get("objects", []):
        rgb = _hex_to_rgb(((obj.get("appearance") or {}).get("base_color")))
        if rgb:
            observed[slug(obj["id"])] = rgb
    if not observed:
        return {"ok": False, "reason": "证据里没有观测颜色"}

    try:
        index = render(game_dir, times, out_dir, width, height, passes=("rgb", "id"))
    except (RuntimeError, OSError, ValueError) as error:
        return {"ok": False, "reason": f"{type(error).__name__}: {error}"[:200]}

    reg = registry_order(program)
    errors: list[float] = []
    ratios: list[float] = []
    per_object: dict[str, dict] = {}
    for frame in index.get("frames", []):
        files = frame.get("files") or {}
        if "rgb" not in files or "id" not in files:
            continue
        from PIL import Image
        rgb = np.asarray(Image.open(Path(out_dir) / files["rgb"]).convert("RGB"))
        masks = decode_id_png(Path(out_dir) / files["id"], reg)
        for node_id, mask in masks.items():
            want = observed.get(node_id)
            if want is None:
                continue
            got = _median_colour(rgb, mask, min_pixels)
            if got is None:
                continue
            error = float(np.mean([abs(a - b) for a, b in zip(got, want)]))
            want_luma = max(sum(want) / 3.0, 1e-6)
            ratio = (sum(got) / 3.0) / want_luma
            errors.append(error)
            ratios.append(ratio)
            row = per_object.setdefault(node_id, {"errors": [], "ratios": []})
            row["errors"].append(error)
            row["ratios"].append(ratio)
    if not errors:
        return {"ok": False, "reason": "没有一个物体在渲染里量得到颜色"}
    return {
        "ok": True,
        "objects": len(per_object),
        "samples": len(errors),
        # 每通道绝对误差，0 到 255。感知侧往返测过一次是 63/255，约 25%
        "channel_error_median": round(statistics.median(errors), 2),
        # 渲出来的亮度除以视频里量到的亮度。小于 1 = 渲得偏暗
        "luma_ratio_median": round(statistics.median(ratios), 3),
        "per_object": {k: {"channel_error": round(statistics.median(v["errors"]), 2),
                           "luma_ratio": round(statistics.median(v["ratios"]), 3)}
                       for k, v in sorted(per_object.items())},
    }
