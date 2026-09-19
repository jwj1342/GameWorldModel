"""VGGT geometry backend: camera intrinsics/extrinsics + depth for a set of frames (relative scale)."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from .base import Geometry

class VGGTBackend:
    name = "vggt"
    def __init__(self, weights_dir: str | Path):
        self.weights_dir = Path(weights_dir)
    def estimate(self, frame_files: list[str], frame_indices: list[int], cfg: dict) -> Geometry:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from safetensors.torch import load_file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model = VGGT()
        sd_path = self.weights_dir / "model.safetensors"
        model.load_state_dict(load_file(str(sd_path)))
        model = model.to(device).eval()
        images = load_and_preprocess_images(frame_files).to(device)   # (N,3,H,W), W=518
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                aggregated_tokens_list, ps_idx = model.aggregator(images[None])
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images[None], ps_idx)
        ext = extrinsic[0].float().cpu().numpy()      # (N,3,4) world->cam (OpenCV)
        intr = intrinsic[0].float().cpu().numpy()     # (N,3,3)
        depth = depth_map[0, ..., 0].float().cpu().numpy()  # (N,H,W)
        conf = depth_conf[0].float().cpu().numpy()
        c2w = []
        for e in ext:
            M = np.eye(4); M[:3, :4] = e
            c2w.append(np.linalg.inv(M))
        del model; torch.cuda.empty_cache()
        return Geometry(frame_indices=list(frame_indices), intrinsics=[k for k in intr], cam_to_world=c2w,
                        depth=[d.astype(np.float32) for d in depth], depth_conf=[c.astype(np.float32) for c in conf],
                        scale="relative", backend=self.name, notes=f"vggt on {len(frame_files)} frames at {images.shape[-1]}x{images.shape[-2]}")
