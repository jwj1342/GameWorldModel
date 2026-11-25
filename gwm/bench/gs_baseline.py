"""三维高斯基线：在本机、同一个分辨率、同一批留出视角上训练和评测，替掉引用的论文值。

为什么要自己跑：论文里 room 的 30.63 dB 是 images_2 上的数，我们的评测在 images_4 上；而且论文值没有
深度，几何那一列没法比。这里用 gsplat 的光栅化和默认致密化策略按三维高斯的标准设置训练
（30k 步，球谐升到 3 阶，L1 和 SSIM 按 0.8 和 0.2 加权），坐标系、相机和留出视角都和场景包一致，
评测直接用 fidelity.py 和 geometry.py，所以和代码场景的分数是同一把尺子量出来的。

相机按场景包的约定：竖直视场角、主点在图像中心（和代码场景的渲染一致）。

    python -m gwm.bench.gs_baseline --config configs/bench/room.yaml [--iters 30000]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image

from . import fidelity, geometry
from .data import read_colmap, normalise, split, three_camera
from .run import REPO, load_config, scene_out

SH_DEGREE = 3
C0 = 0.28209479177387814


def _quat_xyzw_to_matrix(q) -> np.ndarray:
    return geometry._quat_xyzw_to_matrix(q)


def view_to_camera(view: dict, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """three.js 相机（x 右、y 上、看向 -z）换成 gsplat 要的 OpenCV 世界到相机矩阵和内参。"""
    c2w = np.eye(4)
    c2w[:3, :3] = _quat_xyzw_to_matrix(view["quat"]) @ np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = view["pos"]
    f = (height / 2) / math.tan(math.radians(view["fov_deg"]) / 2)
    k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    return torch.tensor(np.linalg.inv(c2w), dtype=torch.float32), torch.tensor(k, dtype=torch.float32)


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """11x11 高斯窗的 SSIM，a、b 是 [1, 3, H, W]。"""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (g[:, None] @ g[None, :]) / g.sum() ** 2
    w = w.expand(3, 1, 11, 11)
    mu_a, mu_b = F.conv2d(a, w, padding=5, groups=3), F.conv2d(b, w, padding=5, groups=3)
    var_a = F.conv2d(a * a, w, padding=5, groups=3) - mu_a ** 2
    var_b = F.conv2d(b * b, w, padding=5, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean()


def render(params: dict, viewmat: torch.Tensor, k: torch.Tensor, width: int, height: int, sh_degree: int, mode="RGB"):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, viewmat[None], k[None], width, height,
        sh_degree=sh_degree, render_mode=mode, packed=False)
    return out[0], alpha[0], info


def encode_depth(depth: np.ndarray, alpha: np.ndarray, path: Path) -> None:
    """按 bench_view.js depth 模式的编码写 PNG：毫米拆成 24 位进 RGB，不透明度不到一半的像素算没有几何。"""
    mm = np.clip(np.round(depth * 1000), 0, 2 ** 24 - 1).astype(np.int64)
    mm[alpha < 0.5] = 0
    rgb = np.stack([mm // 65536, (mm // 256) % 256, mm % 256], -1).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iters", type=int, default=30000)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    d = cfg["data"]
    device = "cuda"
    torch.manual_seed(0)

    cameras, points, colours = read_colmap(REPO / d["root"] / "sparse" / "0")
    r, scale, t, _ = normalise(cameras, points, camera_height_m=d["camera_height_m"], floor_band=d["floor_band"])
    names = [c.name for c in cameras]
    train, held = split(names)
    pack = scene_out(cfg) / "pack"
    held_views = json.loads((pack / "eval" / "cameras.json").read_text(encoding="utf-8"))["views"]
    if [v["name"] for v in held_views] != [names[i] for i in held]:
        raise ValueError("场景包的留出视角和这里算出来的不一样，先重新 pack")

    source = REPO / d["root"] / d["image_dir"]
    width, height = Image.open(source / names[0]).size
    train_set = []
    for i in train:
        view = three_camera(cameras[i], r, scale, t)
        img = torch.tensor(np.asarray(Image.open(source / names[i]).convert("RGB")), dtype=torch.float32) / 255
        train_set.append((*view_to_camera(view, width, height), img))

    world = scale * points @ r.T + t
    centres = np.array([v[0].inverse()[:3, 3].numpy() for v in train_set])
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max() * 1.1)
    xyz = torch.tensor(world, dtype=torch.float32, device=device)
    rgb = torch.tensor(colours / 255.0, dtype=torch.float32, device=device)
    # 初始尺度取到最近三个邻居的平均距离，分块算免得 N×N 的距离矩阵放不下
    dist2 = torch.cat([torch.cdist(chunk, xyz).topk(4, largest=False).values[:, 1:].pow(2).mean(-1)
                       for chunk in xyz.split(4096)])
    n = len(xyz)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(xyz),
        "scales": torch.nn.Parameter(torch.log(dist2.clamp_min(1e-7).sqrt())[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=device))),
        "sh0": torch.nn.Parameter(((rgb - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, (SH_DEGREE + 1) ** 2 - 1, 3, device=device)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    means_decay = 0.01 ** (1.0 / args.iters)
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = scene_out(cfg) / "baselines" / f"{stamp}-3dgs"
    out.mkdir(parents=True)
    started = time.time()
    rng = np.random.default_rng(0)
    for step in range(args.iters):
        viewmat, k, gt = train_set[rng.integers(len(train_set))]
        sh_degree = min(step // 1000, SH_DEGREE)
        image, _, info = render(params, viewmat.to(device), k.to(device), width, height, sh_degree)
        gt = gt.to(device)
        loss = 0.8 * (image - gt).abs().mean() + 0.2 * (1 - ssim(image.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] *= means_decay
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.4f} gaussians {len(params['means'])} {time.time() - started:.0f}s", flush=True)
    train_seconds = time.time() - started
    torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "gaussians.pt")

    flat_dir, depth_dir = out / "renders" / "flat", out / "renders" / "depth"
    flat_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    with torch.no_grad():
        for view in held_views:
            viewmat, k = view_to_camera(view, width, height)
            out_rgbd, alpha, _ = render(params, viewmat.to(device), k.to(device), width, height, SH_DEGREE, mode="RGB+ED")
            stem = Path(view["name"]).stem
            Image.fromarray((out_rgbd[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(flat_dir / f"{stem}.png")
            encode_depth(out_rgbd[..., 3].cpu().numpy(), alpha[..., 0].cpu().numpy(), depth_dir / f"{stem}.png")
    lpips_fn = fidelity.Lpips(REPO / cfg["paths"]["weights"], net=cfg["fidelity"]["lpips_net"])
    fid = fidelity.evaluate(pack / "eval" / "gt", flat_dir, held_views, lpips_fn)
    geo = geometry.evaluate(pack / "eval" / "observed.json", depth_dir, held_views)
    fidelity.write(fid, out / "fidelity_flat.json")
    (out / "geometry.json").write_text(json.dumps(geo, indent=1), encoding="utf-8")
    summary = {"scene": cfg["scene"], "image_dir": d["image_dir"], "iters": args.iters, "gaussians": len(params["means"]),
               "train_seconds": round(train_seconds), "fidelity": fid["mean"], "geometry": geo["mean"],
               "gpu": torch.cuda.get_device_name(0)}
    (out / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
