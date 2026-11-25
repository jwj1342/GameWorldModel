"""把一个带 COLMAP 位姿的多视角数据集整理成评测要用的「场景包」。

做三件事：

1. 读 COLMAP 的二进制稀疏重建（相机、位姿、稀疏点）。
2. 把 COLMAP 那个任意朝向、任意尺度的世界坐标，换成一个 y 轴朝上、地面在 y=0、
   尺度大致是米的坐标系。物理要重力方向和大致的米制尺度才有意义，而且写场景的
   智能体面对一个「地面是平的、墙大致沿坐标轴」的坐标系，才不用把精力花在摆正上。
3. 按三维高斯那边通用的约定划分训练视角和留出视角（文件名排序后每 8 张留 1 张）。
   留出视角的照片只进 `eval/`，不进交给智能体的 `agent/`。

尺度是这里唯一靠先验定的量：COLMAP 给不出绝对尺度，这里假定拍摄时相机离地高度的
中位数是 `camera_height_m`。这是一个人手持拍摄的粗略先验，不是量出来的，写进
`frame.json` 里跟着数据走。
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# COLMAP 相机模型编号 -> 参数个数。只列常见的几种。
_CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
# 三维高斯与 Mip-NeRF 360 通用的留出约定：排序后下标能被 8 整除的是测试视角。
HOLDOUT_EVERY = 8


@dataclass
class Camera:
    name: str            # 图片文件名
    width: int           # 原始分辨率
    height: int
    fx: float
    fy: float
    r_c2w: np.ndarray    # 3x3，COLMAP 相机系（x 右、y 下、z 前）到世界
    centre: np.ndarray   # 3，相机中心在世界里的位置


def _read(f, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("COLMAP 文件提前结束")
    return struct.unpack(fmt, data)


def _quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m: np.ndarray) -> list[float]:
    """3x3 旋转矩阵 -> 四元数，按 three.js 的 [x, y, z, w] 顺序。"""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def read_colmap(sparse_dir: Path) -> tuple[list[Camera], np.ndarray, np.ndarray]:
    """读 cameras.bin / images.bin / points3D.bin，返回（相机列表, 点 Nx3, 颜色 Nx3）。"""
    intrinsics: dict[int, tuple[int, int, float, float]] = {}
    with open(sparse_dir / "cameras.bin", "rb") as f:
        (count,) = _read(f, "<Q")
        for _ in range(count):
            cam_id, model, width, height = _read(f, "<iiQQ")
            if model not in _CAMERA_PARAMS:
                raise ValueError(f"不认识的 COLMAP 相机模型 {model}")
            params = _read(f, "<" + "d" * _CAMERA_PARAMS[model])
            # SIMPLE_* 和 RADIAL 只有一个焦距；PINHOLE 和 OPENCV 有 fx、fy
            fx, fy = (params[0], params[1]) if model in (1, 4, 5) else (params[0], params[0])
            intrinsics[cam_id] = (int(width), int(height), float(fx), float(fy))

    cameras: list[Camera] = []
    with open(sparse_dir / "images.bin", "rb") as f:
        (count,) = _read(f, "<Q")
        for _ in range(count):
            _image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(f, "<i7di")
            name = b""
            while (ch := f.read(1)) != b"\x00":
                if not ch:
                    raise ValueError("images.bin 里的文件名没有结束符")
                name += ch
            (n2d,) = _read(f, "<Q")
            f.seek(24 * n2d, 1)   # 每个二维点 x、y 两个 double 加一个 int64
            r_w2c = _quat_to_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            width, height, fx, fy = intrinsics[cam_id]
            cameras.append(Camera(name.decode(), width, height, fx, fy, r_w2c.T, -r_w2c.T @ t))
    cameras.sort(key=lambda c: c.name)

    xyz, rgb = [], []
    with open(sparse_dir / "points3D.bin", "rb") as f:
        (count,) = _read(f, "<Q")
        for _ in range(count):
            _pid, x, y, z, r, g, b, _err, track = _read(f, "<Q3d3BdQ")
            f.seek(8 * track, 1)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return cameras, np.array(xyz, dtype=np.float64), np.array(rgb, dtype=np.uint8)


def _rotation_to(up: np.ndarray) -> np.ndarray:
    """返回把单位向量 up 转到 +y 的旋转矩阵。"""
    up = up / np.linalg.norm(up)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    c = float(up @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def fit_floor(points: np.ndarray, up: np.ndarray, *, band: float = 0.15, iters: int = 400,
              tol: float | None = None, seed: int = 0) -> tuple[np.ndarray, float, int]:
    """在点云最低的一段里找一个法向接近 up 的平面，当作地面。

    返回（单位法向，朝上；平面偏移 d，使得 n·x = d；内点数）。只在最低的 `band`
    比例里采样，免得把桌面或者天花板拟合成地面；法向偏离 up 超过 25 度的候选直接丢掉。
    """
    up = up / np.linalg.norm(up)
    heights = points @ up
    low = points[heights <= np.quantile(heights, band)]
    if len(low) < 30:
        raise ValueError("最低一段的点太少，拟合不出地面")
    span = np.quantile(heights, 0.95) - np.quantile(heights, 0.05)
    tol = tol if tol is not None else 0.01 * span
    rng = np.random.default_rng(seed)
    best = (up, float(np.median(low @ up)), 0)
    for _ in range(iters):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        if n @ up < 0:
            n = -n
        if n @ up < math.cos(math.radians(25)):
            continue
        d = float(n @ a)
        inliers = int(np.sum(np.abs(low @ n - d) < tol))
        if inliers > best[2]:
            best = (n, d, inliers)
    n, d, _ = best
    mask = np.abs(low @ n - d) < tol
    if mask.sum() >= 3:   # 用全部内点再做一次最小二乘，比三点解稳
        centre = low[mask].mean(axis=0)
        _, _, vt = np.linalg.svd(low[mask] - centre)
        n2 = vt[-1] if vt[-1] @ up > 0 else -vt[-1]
        if n2 @ up >= math.cos(math.radians(25)):
            n, d = n2, float(n2 @ centre)
    return n, d, int(mask.sum())


def _best_yaw(xz: np.ndarray) -> float:
    """找一个绕 y 轴的转角，让点云在水平面上的包围矩形面积最小（墙大致沿坐标轴）。"""
    lo, hi = np.quantile(xz, 0.02, axis=0), np.quantile(xz, 0.98, axis=0)
    core = xz[np.all((xz >= lo) & (xz <= hi), axis=1)]
    best_angle, best_area = 0.0, math.inf
    for deg in np.arange(0.0, 90.0, 0.5):
        a = math.radians(deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = core @ rot.T
        area = float(np.prod(p.max(axis=0) - p.min(axis=0)))
        if area < best_area:
            best_angle, best_area = a, area
    return best_angle


def normalise(cameras: list[Camera], points: np.ndarray, *, camera_height_m: float
              ) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """算出 x_new = s * R @ x_old + t，使新坐标系 y 朝上、地面在 y=0、尺度大致是米。"""
    # 手持拍摄时相机大致是正着拿的：COLMAP 相机系 y 朝下，所以 -y 轴的平均就是朝上
    up0 = -np.mean([c.r_c2w[:, 1] for c in cameras], axis=0)
    n, d, inliers = fit_floor(points, up0)
    r_up = _rotation_to(n)
    centres = np.array([c.centre for c in cameras])
    cam_heights = centres @ n - d
    median_height = float(np.median(cam_heights))
    if median_height <= 0:
        raise ValueError("相机在拟合出来的地面以下，地面拟合错了")
    scale = camera_height_m / median_height

    rotated = (points - n * d) @ r_up.T          # 地面落到 y=0
    yaw = _best_yaw(rotated[:, [0, 2]])
    c, s_ = math.cos(yaw), math.sin(yaw)
    # 绕 y 轴的旋转，作用在 (x, z) 上和 _best_yaw 里的二维旋转一致
    r_yaw = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])
    r = r_yaw @ r_up
    moved = scale * (points - n * d) @ r.T
    origin_xz = np.median(moved[:, [0, 2]], axis=0)
    t = -scale * (r @ (n * d)) - np.array([origin_xz[0], 0.0, origin_xz[1]])
    info = {
        "floor_inliers": inliers,
        "up_vs_camera_up_deg": round(math.degrees(math.acos(float(np.clip(n @ up0 / np.linalg.norm(up0), -1, 1)))), 2),
        "median_camera_height_before": median_height,
        "scale": scale,
        "camera_height_m_assumed": camera_height_m,
        "camera_height_spread_m": [round(float(v), 3) for v in np.quantile(cam_heights * scale, [0.05, 0.5, 0.95])],
    }
    return r, scale, t, info


def three_camera(cam: Camera, r: np.ndarray, scale: float, t: np.ndarray) -> dict:
    """把一台 COLMAP 相机换成 three.js 相机的位姿：x 右、y 上、看向 -z。"""
    c2w = r @ cam.r_c2w @ np.diag([1.0, -1.0, -1.0])
    pos = scale * (r @ cam.centre) + t
    return {
        "name": cam.name,
        "pos": [round(float(v), 5) for v in pos],
        "quat": [round(v, 6) for v in matrix_to_quat(c2w)],
        "fov_deg": round(math.degrees(2 * math.atan(cam.height / (2 * cam.fy))), 4),
        "aspect": round(cam.width / cam.height, 6),
    }


def split(names: list[str]) -> tuple[list[int], list[int]]:
    """按通用约定划分：排序后下标能被 8 整除的留出，其余训练。"""
    order = sorted(range(len(names)), key=lambda i: names[i])
    held = [i for rank, i in enumerate(order) if rank % HOLDOUT_EVERY == 0]
    train = [i for rank, i in enumerate(order) if rank % HOLDOUT_EVERY != 0]
    return train, held


def _evenly(items: list, count: int) -> list:
    if count >= len(items):
        return list(items)
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def build_pack(scene_dir: Path, out_dir: Path, *, image_dir: str, camera_height_m: float,
               agent_views: int, agent_view_width: int, max_points: int, seed: int = 0) -> dict:
    """生成场景包。`agent/` 给写场景的智能体，`eval/` 只给评测用。"""
    cameras, points, colours = read_colmap(scene_dir / "sparse" / "0")
    r, scale, t, info = normalise(cameras, points, camera_height_m=camera_height_m)
    names = [c.name for c in cameras]
    train, held = split(names)
    source = scene_dir / image_dir
    missing = [n for n in names if not (source / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{source} 里缺 {len(missing)} 张图，例如 {missing[0]}")

    agent_dir, eval_dir = out_dir / "agent", out_dir / "eval"
    (agent_dir / "views").mkdir(parents=True, exist_ok=True)
    (eval_dir / "gt").mkdir(parents=True, exist_ok=True)

    def entry(i: int) -> dict:
        return three_camera(cameras[i], r, scale, t)

    # 全部训练相机都给，照片只给均匀抽出来的一部分：上下文有限，而且智能体可以按需再要
    shown = _evenly(train, agent_views)
    shown_entries = []
    for i in shown:
        e = entry(i)
        img = Image.open(source / names[i]).convert("RGB")
        w = min(agent_view_width, img.width)
        img = img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)
        file = Path(names[i]).stem + ".jpg"
        img.save(agent_dir / "views" / file, quality=90)
        shown_entries.append({**e, "file": f"views/{file}", "width": img.width, "height": img.height})
    (agent_dir / "cameras.json").write_text(json.dumps({
        "convention": "three.js：相机 x 右、y 上、看向 -z；quat 是 [x, y, z, w]；世界 y 朝上，地面在 y=0，单位大致是米",
        "views": shown_entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    world = scale * points @ r.T + t
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(world), min(max_points, len(world)), replace=False)
    (agent_dir / "points.json").write_text(json.dumps({
        "note": "COLMAP 稀疏点，已换到上面那个坐标系。xyz 单位大致是米，rgb 是 0-255。",
        "xyz": np.round(world[keep], 3).tolist(),
        "rgb": colours[keep].tolist(),
    }), encoding="utf-8")

    held_entries = []
    for i in held:
        e = entry(i)
        img = Image.open(source / names[i]).convert("RGB")
        file = Path(names[i]).stem + ".png"
        img.save(eval_dir / "gt" / file)
        held_entries.append({**e, "file": f"gt/{file}", "width": img.width, "height": img.height})
    (eval_dir / "cameras.json").write_text(json.dumps({"views": held_entries}, indent=1), encoding="utf-8")

    lo, hi = np.quantile(world, 0.02, axis=0), np.quantile(world, 0.98, axis=0)
    frame = {
        "scene": scene_dir.name,
        "image_dir": image_dir,
        "counts": {"all": len(names), "train": len(train), "held_out": len(held), "shown_to_agent": len(shown)},
        "split": f"文件名排序后每 {HOLDOUT_EVERY} 张留出 1 张",
        "bounds_2_98_percentile": {"min": np.round(lo, 2).tolist(), "max": np.round(hi, 2).tolist()},
        "normalisation": info,
    }
    (out_dir / "frame.json").write_text(json.dumps(frame, ensure_ascii=False, indent=1), encoding="utf-8")
    (agent_dir / "frame.json").write_text(json.dumps({
        "up": "+y", "floor": "y = 0", "units": "大致是米（按相机离地高度的先验定的尺度，不是量出来的）",
        "bounds_2_98_percentile": frame["bounds_2_98_percentile"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return frame
