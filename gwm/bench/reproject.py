"""外观换成观测之后的上限：同一套几何，颜色直接从最近的训练照片投过来，能拿到多少分。

代码场景和三维高斯的保真度差了一大截（LPIPS 0.61 对 0.17），这里把差距拆成几何和外观两块。
对每个留出视角，取某一套几何渲染出来的深度，把每个像素反投影成三维点，再投到离它最近的 4 张
训练照片里取颜色（训练视角上同一套几何的深度要对得上，对不上说明被挡住了，不取），按视角相近
程度加权平均。取不到颜色的像素用这套几何自己的渲染图补。

同一个过程分别套在代码场景和三维高斯的几何上：两者的差只来自几何；代码场景投影之后和它自己
纯色渲染的差，是外观（贴图、光照）能补回来的上限。另外报一个曝光校正后的分数：代码场景的
纯色渲染按通道拟合一个线性颜色映射到照片上之后再评，看有多少差距只是整体亮度和色调不对。

这是分析用的，不是评测的一部分：把照片的像素投到场景上，正是评测里不许智能体做的事。

    python -m gwm.bench.reproject --config configs/bench/room.yaml --cand DIR --gs out/bench/room/baselines/<run>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import fidelity, geometry
from .data import normalise, read_colmap, split, three_camera
from .gs_baseline import SH_DEGREE, encode_depth, render, view_to_camera
from .run import REPO, _node, load_config, scene_out

NEIGHBOURS = 4
OCCLUSION_REL = 0.03     # 训练视角上的深度和投过去的深度差超过 3% 就当被挡住


def backproject(view: dict, depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    f = (height / 2) / math.tan(math.radians(view["fov_deg"]) / 2)
    vs, us = np.mgrid[0:height, 0:width]
    cam = np.stack([(us + 0.5 - width / 2) * depth / f, -(vs + 0.5 - height / 2) * depth / f, -depth], -1)
    return cam @ geometry._quat_xyzw_to_matrix(view["quat"]).T + np.asarray(view["pos"])


def sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """双线性取色，u、v 已经保证在图像内。"""
    h, w, _ = image.shape
    u0, v0 = np.clip(np.floor(u - 0.5).astype(int), 0, w - 2), np.clip(np.floor(v - 0.5).astype(int), 0, h - 2)
    du, dv = np.clip(u - 0.5 - u0, 0, 1)[:, None], np.clip(v - 0.5 - v0, 0, 1)[:, None]
    return (image[v0, u0] * (1 - du) * (1 - dv) + image[v0, u0 + 1] * du * (1 - dv)
            + image[v0 + 1, u0] * (1 - du) * dv + image[v0 + 1, u0 + 1] * du * dv)


def reproject_view(held: dict, depth: np.ndarray, fallback: np.ndarray, neighbours: list[tuple[dict, np.ndarray, np.ndarray]]
                   ) -> tuple[np.ndarray, float]:
    height, width = depth.shape
    out = fallback.astype(np.float64).copy()
    valid = ~np.isnan(depth)
    points = backproject(held, np.nan_to_num(depth))[valid]
    acc = np.zeros((len(points), 3)); wsum = np.zeros(len(points))
    held_dir = geometry._quat_xyzw_to_matrix(held["quat"])[:, 2]
    for view, photo, tdepth in neighbours:
        u, v, z = geometry.project(view, points, width, height)
        inside = (z > 0.05) & (u >= 0.5) & (u < width - 1.5) & (v >= 0.5) & (v < height - 1.5)
        idx = np.flatnonzero(inside)
        seen = tdepth[v[idx].astype(int), u[idx].astype(int)]
        idx = idx[~np.isnan(seen) & (np.abs(seen - z[idx]) <= OCCLUSION_REL * z[idx])]
        weight = max(float(geometry._quat_xyzw_to_matrix(view["quat"])[:, 2] @ held_dir), 0.05)
        acc[idx] += weight * sample(photo, u[idx], v[idx])
        wsum[idx] += weight
    got = wsum > 0
    colours = out[valid]
    colours[got] = acc[got] / wsum[got][:, None]
    out[valid] = colours
    return np.clip(out, 0, 255).astype(np.uint8), float(got.sum() / (height * width))


def colour_corrected(render: np.ndarray, photo: np.ndarray) -> np.ndarray:
    """按通道拟合 photo ≈ a * render + b，看整体亮度和色调能补回多少。"""
    out = np.empty_like(render, dtype=np.float64)
    for c in range(3):
        a, b = np.polyfit(render[..., c].ravel().astype(np.float64), photo[..., c].ravel().astype(np.float64), 1)
        out[..., c] = a * render[..., c] + b
    return np.clip(out, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cand", required=True, help="代码场景的候选目录（scene.js、materials.json）")
    parser.add_argument("--gs", required=True, help="gs_baseline 的产物目录，里面要有 gaussians.pt")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    d = cfg["data"]
    pack = scene_out(cfg) / "pack"
    out = Path(args.gs) / "reproject"
    out.mkdir(exist_ok=True)

    cameras, points, _ = read_colmap(REPO / d["root"] / "sparse" / "0")
    r, scale, t, _ = normalise(cameras, points, camera_height_m=d["camera_height_m"], floor_band=d["floor_band"])
    names = [c.name for c in cameras]
    train, _ = split(names)
    source = REPO / d["root"] / d["image_dir"]
    width, height = Image.open(source / names[0]).size
    held_views = json.loads((pack / "eval" / "cameras.json").read_text(encoding="utf-8"))["views"]
    train_views = [{**three_camera(cameras[i], r, scale, t), "width": width, "height": height} for i in train]
    (out / "train_cameras.json").write_text(json.dumps({"views": train_views}), encoding="utf-8")

    # 两套几何在留出视角和训练视角上的深度
    cand = out / "candidate"
    cand.mkdir(exist_ok=True)
    for file in ("scene.js", "materials.json"):
        (cand / file).write_bytes((Path(args.cand) / file).read_bytes())
    (cand / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
    code = {"held": out / "code_held_depth", "train": out / "code_train_depth", "flat": out / "code_held_flat"}
    _node("bench_render.mjs", "--cand", str(cand), "--cameras", str(pack / "eval" / "cameras.json"), "--out", str(code["held"]), "--mode", "depth")
    _node("bench_render.mjs", "--cand", str(cand), "--cameras", str(out / "train_cameras.json"), "--out", str(code["train"]), "--mode", "depth")
    _node("bench_render.mjs", "--cand", str(cand), "--cameras", str(pack / "eval" / "cameras.json"), "--out", str(code["flat"]), "--mode", "flat")
    gs = {"held": out / "gs_held_depth", "train": out / "gs_train_depth", "flat": Path(args.gs) / "renders" / "flat"}
    params = {k: v.to("cuda") for k, v in torch.load(Path(args.gs) / "gaussians.pt").items()}
    for key, views in (("held", held_views), ("train", train_views)):
        gs[key].mkdir(exist_ok=True)
        with torch.no_grad():
            for view in views:
                viewmat, k = view_to_camera(view, width, height)
                rgbd, alpha, _ = render(params, viewmat.cuda(), k.cuda(), width, height, SH_DEGREE, mode="RGB+ED")
                encode_depth(rgbd[..., 3].cpu().numpy(), alpha[..., 0].cpu().numpy(), gs[key] / f"{Path(view['name']).stem}.png")

    centres = np.array([v["pos"] for v in train_views])
    photos = {v["name"]: fidelity.load_rgb(source / v["name"]).astype(np.float64) for v in train_views}
    lpips_fn = fidelity.Lpips(REPO / cfg["paths"]["weights"], net=cfg["fidelity"]["lpips_net"])
    results = {}
    for label, geo in (("code", code), ("3dgs", gs)):
        target = out / f"{label}_reprojected"
        target.mkdir(exist_ok=True)
        coverage = []
        for held in held_views:
            stem = Path(held["name"]).stem
            near = np.argsort(np.linalg.norm(centres - np.asarray(held["pos"]), axis=1))[:NEIGHBOURS]
            neighbours = [(train_views[i], photos[train_views[i]["name"]],
                           geometry.decode_depth(geo["train"] / f"{Path(train_views[i]['name']).stem}.png")) for i in near]
            image, cov = reproject_view(held, geometry.decode_depth(geo["held"] / f"{stem}.png"),
                                        fidelity.load_rgb(geo["flat"] / f"{stem}.png"), neighbours)
            Image.fromarray(image).save(target / f"{stem}.png")
            coverage.append(cov)
        results[f"{label}_reprojected"] = {**fidelity.evaluate(pack / "eval" / "gt", target, held_views, lpips_fn)["mean"],
                                           "coverage": round(float(np.mean(coverage)), 4)}
        print(label, results[f"{label}_reprojected"], flush=True)

    target = out / "code_colour_corrected"
    target.mkdir(exist_ok=True)
    for held in held_views:
        stem = Path(held["name"]).stem
        Image.fromarray(colour_corrected(fidelity.load_rgb(code["flat"] / f"{stem}.png"),
                                         fidelity.load_rgb(pack / "eval" / "gt" / f"{stem}.png"))).save(target / f"{stem}.png")
    results["code_colour_corrected"] = fidelity.evaluate(pack / "eval" / "gt", target, held_views, lpips_fn)["mean"]
    results["code_flat"] = fidelity.evaluate(pack / "eval" / "gt", code["flat"], held_views, lpips_fn)["mean"]
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
