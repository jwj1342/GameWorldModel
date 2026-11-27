"""两轴评测里几何那一块：渲染出来的深度，和留出照片里 COLMAP 实际看到的点对不对得上。

PSNR、SSIM、LPIPS 把几何、光照和贴图混在一起量，「瓶颈在几何还是在光照」从它们身上看不出来。
这里只看几何：对每个留出视角，取 COLMAP 在这张照片里实际观测到的三维点（被挡住的、画面外的
都不算），投到渲染相机里，和同一个像素上渲染出来的深度比。

深度图由 kernel/bench_view.js 的 depth 模式渲染：视空间深度按毫米拆成 24 位写进 RGB，0 是没有几何。
相机投影和渲染用的是同一套 three.js 约定（竖直视场角、主点在图像中心），所以两边的像素是对齐的。

报四个数，都是逐视角算完再对视角取平均：
  coverage       投进画面的点里，渲染出来有几何的比例（没几何说明那里漏了东西）
  median_abs_m   有几何的点上，深度差的绝对值的中位数（米，尺度是场景包的先验尺度）
  within_10cm    深度差不超过 10 厘米的点占投进画面的点的比例（没几何的点算不合格）
  median_rel     深度差除以点的深度，取中位数
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

INLIER_M = 0.10


def decode_depth(path: Path) -> np.ndarray:
    """depth 模式的 PNG 解码成米；没有几何的地方是 nan。"""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    mm = rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
    depth = mm.astype(np.float64) / 1000.0
    depth[mm == 0] = np.nan
    return depth


def _quat_xyzw_to_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project(view: dict, xyz: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """three.js 相机（x 右、y 上、看向 -z）下的像素坐标和视空间深度。"""
    r = _quat_xyzw_to_matrix(view["quat"])
    cam = (xyz - np.asarray(view["pos"])) @ r          # 等于 R^T (p - pos)
    depth = -cam[:, 2]
    f = (height / 2) / math.tan(math.radians(view["fov_deg"]) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + f * cam[:, 0] / depth
        v = height / 2 - f * cam[:, 1] / depth
    return u, v, depth


def evaluate_view(view: dict, points: np.ndarray, depth_map: np.ndarray) -> dict:
    height, width = depth_map.shape
    u, v, z = project(view, points, width, height)
    inside = (z > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside].astype(int), v[inside].astype(int), z[inside]
    rendered = depth_map[v, u]
    has = ~np.isnan(rendered)
    err = np.abs(rendered[has] - z[has])
    return {
        "name": view["name"], "points": int(inside.sum()),
        "coverage": round(float(has.mean()), 4) if inside.any() else None,
        "median_abs_m": round(float(np.median(err)), 4) if has.any() else None,
        "within_10cm": round(float((err <= INLIER_M).sum() / inside.sum()), 4) if inside.any() else None,
        "median_rel": round(float(np.median(err / z[has])), 4) if has.any() else None,
    }


def evaluate(observed_file: Path, depth_dir: Path, views: list[dict]) -> dict:
    observed = json.loads(observed_file.read_text(encoding="utf-8"))["views"]
    rows = [evaluate_view(view, np.asarray(observed[view["name"]], dtype=np.float64),
                          decode_depth(depth_dir / f"{Path(view['name']).stem}.png"))
            for view in views]
    keys = ("coverage", "median_abs_m", "within_10cm", "median_rel")
    mean = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 4) for k in keys}
    return {"views": len(rows), "mean": mean, "per_view": rows}
