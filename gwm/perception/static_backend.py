"""Fallback geometry backend: static camera assumption + monocular depth (Depth Anything V2 Small via transformers)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .base import Geometry

class StaticDepthBackend:
    name = "static"
    def __init__(self, weights_dir: str | Path, width: int = 518):
        self.weights_dir = Path(weights_dir); self.width = width
    def estimate(self, frame_files: list[str], frame_indices: list[int], cfg: dict) -> Geometry:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(str(self.weights_dir)); model = AutoModelForDepthEstimation.from_pretrained(str(self.weights_dir)).to(device).eval()
        depths, intr = [], []
        for f in frame_files:
            im = Image.open(f).convert("RGB"); W = self.width; H = int(round(im.height * W / im.width / 14) * 14)
            im = im.resize((W, H))
            with torch.no_grad():
                out = model(**proc(images=im, return_tensors="pt").to(device))
                pred = torch.nn.functional.interpolate(out.predicted_depth[None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            rel = pred.float().cpu().numpy()             # relative inverse depth (DA2 outputs disparity-like)
            depth = 1.0 / np.clip(rel, 1e-3, None); depth = depth / np.median(depth) * 3.0   # arbitrary: median 3 units
            depths.append(depth.astype(np.float32))
            f_px = 0.9 * W  # ~58 deg horizontal fov
            intr.append(np.array([[f_px, 0, W / 2], [0, f_px, H / 2], [0, 0, 1]], dtype=np.float64))
        c2w = [np.eye(4) for _ in frame_files]
        return Geometry(frame_indices=list(frame_indices), intrinsics=intr, cam_to_world=c2w, depth=depths, scale="relative", backend=self.name, notes="static camera assumption + Depth Anything V2 small (relative)")
