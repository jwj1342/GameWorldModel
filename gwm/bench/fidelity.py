"""保真度那一轴：候选场景在留出视角下的渲染图，和同一视角的真实照片比。

三个数都是三维重建那边的标准口径，方便直接和高斯重建的结果放在一起看：

- PSNR：像素均方误差的对数，越高越好。
- SSIM：局部结构相似度，越高越好。
- LPIPS（VGG）：感知距离，越低越好。三维高斯的论文报的也是 VGG 版。

这里要说清楚一件事：代码写出来的场景配上生成的贴图，是「合理」而不是「相同」，
木纹的每一根线不会和照片对上，所以 PSNR 和 SSIM 天然吃亏。主看 LPIPS；另外两个
一起报，是为了能和别人的表对上，不是因为它们适合拿来排名。

留出视角的照片在生成场景的全过程里都没有给过智能体。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return math.inf if mse == 0 else -10.0 * math.log10(mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


class Lpips:
    """LPIPS 的薄封装。权重放到仓库的 models/ 下面，不往用户主目录里写。"""

    def __init__(self, weights_dir: Path, net: str = "vgg", device: str | None = None):
        os.environ.setdefault("TORCH_HOME", str(Path(weights_dir) / "torch"))
        import lpips
        import torch
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.net = net
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        torch = self.torch

        def prep(x: np.ndarray):
            return (torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1).to(self.device)

        with torch.no_grad():
            return float(self.model(prep(a), prep(b)).item())


def evaluate(gt_dir: Path, render_dir: Path, views: list[dict], lpips_fn=None) -> dict:
    """逐视角算三个指标，再取平均。任何一个视角缺图或者尺寸对不上都直接报错，不悄悄跳过。"""
    rows = []
    for view in views:
        stem = Path(view["name"]).stem
        gt_path, render_path = gt_dir / f"{stem}.png", render_dir / f"{stem}.png"
        if not render_path.is_file():
            raise FileNotFoundError(f"缺渲染图 {render_path}")
        gt, render = load_rgb(gt_path), load_rgb(render_path)
        if gt.shape != render.shape:
            raise ValueError(f"{stem} 的尺寸对不上：照片 {gt.shape}，渲染 {render.shape}")
        row = {"name": view["name"], "psnr": round(psnr(gt, render), 3), "ssim": round(ssim(gt, render), 4)}
        if lpips_fn is not None:
            row["lpips"] = round(lpips_fn(gt, render), 4)
        rows.append(row)
    keys = [k for k in ("psnr", "ssim", "lpips") if rows and k in rows[0]]
    mean = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    return {"views": len(rows), "mean": mean, "per_view": rows}


def write(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
