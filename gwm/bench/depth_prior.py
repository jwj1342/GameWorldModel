"""输入消融用的稠密点：单目深度估计对齐到 COLMAP 稀疏点之后反投影。

对给智能体看的每张训练照片，用 Depth Anything V2 估一张相对视差图。它只在一个仿射变换下是对的，
所以拿这张照片里 COLMAP 实际观测到的稀疏点去拟合：视差 ≈ a / 深度 + b，最小二乘之后把残差最大的
两成点剔掉再拟合一次。然后按每 8 个像素取一个点反投影到场景包的坐标系里，所有视角合在一起，
按 2 厘米的体素去重。结果写在场景包的 agent/dense_points.json，agent prepare --inputs dense 时并进 points.json。

    python -m gwm.bench.depth_prior --config configs/bench/room.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import geometry
from .data import normalise, read_colmap, read_observations
from .run import REPO, load_config, scene_out

MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
STRIDE = 8
VOXEL_M = 0.02
MAX_DEPTH_M = 15.0


def fit_disparity(pred: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """pred ≈ a / depth + b。拟合两遍，第二遍去掉残差最大的 20%。"""
    x = 1.0 / depth
    keep = np.ones(len(x), dtype=bool)
    for _ in range(2):
        a, b = np.polyfit(x[keep], pred[keep], 1)
        resid = np.abs(a * x + b - pred)
        keep = resid <= np.quantile(resid, 0.8)
    return float(a), float(b)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    d = cfg["data"]
    cameras, points, _ = read_colmap(REPO / d["root"] / "sparse" / "0")
    r, scale, t, _ = normalise(cameras, points, camera_height_m=d["camera_height_m"], floor_band=d["floor_band"])
    world = scale * points @ r.T + t
    observed = read_observations(REPO / d["root"] / "sparse" / "0")
    agent_dir = scene_out(cfg) / "pack" / "agent"
    views = json.loads((agent_dir / "cameras.json").read_text(encoding="utf-8"))["views"]

    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to("cuda").eval()
    xyz_all, rgb_all, fits = [], [], []
    for view in views:
        img = Image.open(agent_dir / view["file"]).convert("RGB")
        width, height = img.size
        with torch.no_grad():
            out = model(**processor(images=img, return_tensors="pt").to("cuda"))
        pred = processor.post_process_depth_estimation(out, target_sizes=[(height, width)])[0]["predicted_depth"]
        pred = pred.float().cpu().numpy()

        pts = world[observed[view["name"]]]
        u, v, z = geometry.project(view, pts, width, height)
        ok = (z > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        a, b = fit_disparity(pred[v[ok].astype(int), u[ok].astype(int)], z[ok])
        fits.append({"view": view["name"], "a": round(a, 4), "b": round(b, 4), "points": int(ok.sum())})

        vs, us = np.mgrid[STRIDE // 2:height:STRIDE, STRIDE // 2:width:STRIDE]
        disp = pred[vs, us] - b
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = a / disp
        good = (disp > 0) & (depth > 0.1) & (depth < MAX_DEPTH_M)
        us, vs, depth = us[good], vs[good], depth[good]
        f = (height / 2) / math.tan(math.radians(view["fov_deg"]) / 2)
        cam = np.stack([(us - width / 2) * depth / f, -(vs - height / 2) * depth / f, -depth], -1)
        xyz_all.append(cam @ geometry._quat_xyzw_to_matrix(view["quat"]).T + np.asarray(view["pos"]))
        rgb_all.append(np.asarray(img)[vs, us])

    xyz, rgb = np.concatenate(xyz_all), np.concatenate(rgb_all)
    _, first = np.unique(np.floor(xyz / VOXEL_M).astype(np.int64), axis=0, return_index=True)
    xyz, rgb = xyz[first], rgb[first]
    (agent_dir / "dense_points.json").write_text(json.dumps({
        "note": f"{MODEL} 的单目视差，按每张照片里 COLMAP 观测到的稀疏点拟合仿射之后反投影，{VOXEL_M} 米体素去重。",
        "fits": fits, "xyz": np.round(xyz, 3).tolist(), "rgb": rgb.tolist(),
    }), encoding="utf-8")
    print(f"dense points: {len(xyz)} from {len(views)} views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
