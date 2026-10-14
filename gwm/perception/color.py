"""Observed display color, not illumination-corrected material/albedo."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .masks import resize_mask

def _hex(rgb) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in np.floor(np.asarray(rgb) + 0.5))

def measured_color(appearance) -> str | None:
    """Only consume explicitly measured, well-formed display color."""
    if not isinstance(appearance, dict) or not appearance.get("samples"):
        return None
    value = appearance.get("base_color")
    return value if appearance.get("color_space") == "sRGB" and isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else None

def observe_color(frames: list[dict], masks: dict, frame_size: tuple[int, int],
                  object_id: str, min_pixels: int = 4) -> dict:
    """Read matching full-frame masks; each valid frame contributes one RGB median.

    Masks may be full-frame downsampled (including cached masks). Nearest-neighbor
    resizing keeps them binary; cropped/aspect-incompatible masks are rejected.
    No bbox, class color, absent frame, or numeric confidence is substituted.
    """
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int) or min_pixels < 1:
        raise ValueError("min_pixels must be a positive integer")
    result = {"base_color": "unknown", "color_space": "sRGB",
              "method": "median_of_frame_mask_medians", "confidence": "unknown",
              "samples": [], "issues": []}
    medians = []
    for fi, mask in sorted(masks.items()):
        reason = None
        if type(fi) is not int or not 0 <= fi < len(frames) or frames[fi].get("index") != fi:
            result["issues"].append({"frame_index": fi, "reason": "frame_reference_mismatch"})
            continue
        frame = frames[fi]
        m = np.asarray(mask)
        if m.ndim != 2 or m.size == 0 or m.dtype != np.bool_:
            reason = "invalid_binary_mask"
        elif abs(m.shape[0] - round(m.shape[1] * frame_size[1] / frame_size[0])) > 1:
            reason = "mask_aspect_mismatch"
        elif not m.any():
            reason = "empty_mask"
        if reason:
            result["issues"].append({"frame_index": fi, "reason": reason})
            continue
        try:
            with Image.open(frame["file"]) as image:
                image.load()
                if abs(image.height - round(image.width * frame_size[1] / frame_size[0])) > 1:
                    result["issues"].append({"frame_index": fi, "reason": "image_aspect_mismatch"})
                    continue
                rgb = np.asarray(image.convert("RGB"))
                selected = resize_mask(m, image.size)
            pixels = int(selected.sum())
            if pixels < min_pixels:
                result["issues"].append({"frame_index": fi, "reason": "insufficient_pixels"})
                continue
            median = np.median(rgb[selected], axis=0)
        except (OSError, ValueError, KeyError) as error:
            result["issues"].append({"frame_index": fi, "reason": "image_unavailable",
                                     "error_type": type(error).__name__})
            continue
        medians.append(median)
        result["samples"].append({"frame_index": fi, "t": float(frame["t"]),
            "image_ref": Path(frame["file"]).name, "mask_ref": f"masks.npz#{object_id}__{fi}",
            "pixels": pixels, "base_color": _hex(median)})
    if medians:
        result["base_color"] = _hex(np.median(medians, axis=0))
    elif not result["issues"]:
        result["issues"].append({"reason": "missing_mask"})
    return result
