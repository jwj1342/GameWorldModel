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

def observe_colors(frames: list[dict], object_masks: dict, frame_size: tuple[int, int],
                   min_pixels: int = 4, aspect_tolerance: float = 0.02) -> dict:
    """Measure all objects frame-first, decoding each requested image at most once.

    Relative aspect tolerance allows geometry patch-size rounding, not arbitrary
    crops. Only one decoded RGB frame is retained at a time. No albedo is inferred.
    """
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int) or min_pixels < 1:
        raise ValueError("min_pixels must be a positive integer")
    if (isinstance(aspect_tolerance, bool) or not isinstance(aspect_tolerance, (int, float))
            or not np.isfinite(aspect_tolerance) or not 0 <= aspect_tolerance < 1):
        raise ValueError("aspect_tolerance must be a finite ratio in [0, 1)")
    if len(frame_size) != 2 or any(not np.isfinite(v) or v <= 0 for v in frame_size):
        raise ValueError("frame_size must contain positive finite dimensions")
    ratio = frame_size[0] / frame_size[1]
    def compatible(width, height):
        return abs((width / height) / ratio - 1) <= aspect_tolerance + 1e-12

    results = {oid: {"base_color": "unknown", "color_space": "sRGB",
               "method": "median_of_frame_mask_medians", "confidence": "unknown",
               "samples": [], "issues": []} for oid in object_masks}
    medians = {oid: [] for oid in object_masks}
    by_frame = {}
    for oid, masks in object_masks.items():
        for fi, mask in sorted(masks.items(), key=lambda item: repr(item[0])):
            if type(fi) is not int or not 0 <= fi < len(frames) or frames[fi].get("index") != fi:
                results[oid]["issues"].append({"frame_index": fi, "reason": "frame_reference_mismatch"})
                continue
            by_frame.setdefault(fi, []).append((oid, mask))
    for fi, entries in sorted(by_frame.items()):
        frame = frames[fi]
        valid = []
        for oid, mask in entries:
            m = np.asarray(mask)
            reason = None
            if m.ndim != 2 or m.size == 0 or m.dtype != np.bool_:
                reason = "invalid_binary_mask"
            elif not compatible(m.shape[1], m.shape[0]):
                reason = "mask_aspect_mismatch"
            elif not m.any():
                reason = "empty_mask"
            if reason:
                results[oid]["issues"].append({"frame_index": fi, "reason": reason})
            else:
                valid.append((oid, m))
        if not valid:
            continue
        try:
            with Image.open(frame["file"]) as image:
                image.load()
                if not compatible(image.width, image.height):
                    for oid, _ in valid:
                        results[oid]["issues"].append({"frame_index": fi, "reason": "image_aspect_mismatch"})
                    continue
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, ValueError, KeyError) as error:
            for oid, _ in valid:
                results[oid]["issues"].append({"frame_index": fi, "reason": "image_unavailable",
                                             "error_type": type(error).__name__})
            continue
        for oid, m in valid:
            selected = resize_mask(m, (rgb.shape[1], rgb.shape[0]))
            pixels = int(selected.sum())
            if pixels < min_pixels:
                results[oid]["issues"].append({"frame_index": fi, "reason": "insufficient_pixels"})
                continue
            median = np.median(rgb[selected], axis=0)
            medians[oid].append(median)
            results[oid]["samples"].append({"frame_index": fi, "t": float(frame["t"]),
                "image_ref": Path(frame["file"]).name, "mask_ref": f"masks.npz#{oid}__{fi}",
                "pixels": pixels, "base_color": _hex(median)})
    for oid, result in results.items():
        if medians[oid]:
            result["base_color"] = _hex(np.median(medians[oid], axis=0))
        elif not result["issues"]:
            result["issues"].append({"reason": "missing_mask"})
    return results


def observe_color(frames: list[dict], masks: dict, frame_size: tuple[int, int],
                  object_id: str, min_pixels: int = 4, aspect_tolerance: float = 0.02) -> dict:
    """Backward-compatible single-object wrapper around the batch measurement."""
    return observe_colors(frames, {object_id: masks}, frame_size,
                          min_pixels, aspect_tolerance)[object_id]
