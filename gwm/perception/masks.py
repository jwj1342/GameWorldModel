"""Small mask helpers shared by perception, feedback and the critic."""
from __future__ import annotations
import numpy as np
from PIL import Image

def resize_mask(m: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if m.shape[1] == size_wh[0] and m.shape[0] == size_wh[1]: return m
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size_wh, Image.NEAREST)) > 127

def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 0.0

def centroid(m: np.ndarray):
    ys, xs = np.where(m); return (float(xs.mean()), float(ys.mean())) if len(xs) else None
