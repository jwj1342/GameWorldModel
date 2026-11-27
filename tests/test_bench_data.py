"""两轴评测的数据整理：坐标归一化和相机换算必须是对的，否则留出视角的渲染和照片对不上，
保真度那一轴量出来的就是换算误差而不是场景好坏。"""
import math
import struct

import numpy as np
import pytest

from gwm.bench import data


def _look_at(centre, target, world_up):
    """造一台 COLMAP 约定的相机（x 右、y 下、z 前），返回 c2w 旋转。"""
    z = target - centre
    z = z / np.linalg.norm(z)
    x = np.cross(z, world_up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)          # y 朝下
    return np.stack([x, y, z], axis=1)


def _scene(seed=0):
    """一个斜着放、尺度任意的房间：地面一大片点，上面散一些家具的点，相机绕着拍。"""
    rng = np.random.default_rng(seed)
    scale, height = 2.4, 1.4
    tilt = data._rotation_to(np.array([0.3, 1.0, -0.2])).T      # 把 +y 转到一个歪的方向
    up = tilt @ np.array([0.0, 1.0, 0.0])
    floor = np.column_stack([rng.uniform(-3, 3, 4000), np.zeros(4000), rng.uniform(-2, 2, 4000)])
    stuff = np.column_stack([rng.uniform(-3, 3, 1500), rng.uniform(0.2, 2.2, 1500), rng.uniform(-2, 2, 1500)])
    points = (np.vstack([floor, stuff]) * scale) @ tilt.T + np.array([5.0, -3.0, 2.0])
    cameras = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        local = np.array([2.5 * math.cos(a), height + 0.1 * math.sin(3 * a), 1.5 * math.sin(a)])
        centre = (local * scale) @ tilt.T + np.array([5.0, -3.0, 2.0])
        target = (np.array([0.0, 0.8, 0.0]) * scale) @ tilt.T + np.array([5.0, -3.0, 2.0])
        cameras.append(data.Camera(f"img_{i:03d}.jpg", 800, 600, 700.0, 700.0, _look_at(centre, target, up), centre))
    return cameras, points, height


def test_floor_lands_on_zero_and_scale_follows_camera_height():
    cameras, points, height = _scene()
    r, scale, t, info = data.normalise(cameras, points, camera_height_m=height, floor_band=0.15)
    world = scale * points @ r.T + t
    floor = world[:4000]
    assert abs(float(np.median(floor[:, 1]))) < 0.01
    assert float(np.quantile(np.abs(floor[:, 1]), 0.99)) < 0.02
    heights = [data.three_camera(c, r, scale, t)["pos"][1] for c in cameras]
    assert float(np.median(heights)) == pytest.approx(height, abs=0.02)
    # 尺度是按相机高度定的，房间应该回到原来的大小：地面是 6 x 4 米
    extent = np.sort(floor[:, [0, 2]].max(axis=0) - floor[:, [0, 2]].min(axis=0))
    assert extent == pytest.approx([4.0, 6.0], abs=0.15)
    assert info["floor_inliers"] > 500


def test_three_camera_projects_like_colmap():
    """同一个三维点，按 COLMAP 的投影和按换算后的 three.js 相机投影，要落在同一个像素上。"""
    cameras, points, height = _scene(seed=1)
    r, scale, t, _ = data.normalise(cameras, points, camera_height_m=height, floor_band=0.15)
    cam = cameras[5]
    view = data.three_camera(cam, r, scale, t)
    x, y, z, w = view["quat"]
    c2w = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    f = (cam.height / 2) / math.tan(math.radians(view["fov_deg"]) / 2)
    checked = 0
    for p in points[4000:]:
        local = cam.r_c2w.T @ (p - cam.centre)                  # COLMAP 相机系
        if local[2] <= 0.1:
            continue
        u0, v0 = cam.fx * local[0] / local[2], cam.fy * local[1] / local[2]
        if abs(u0) > cam.width / 2 or abs(v0) > cam.height / 2:
            continue                                            # 只查落在画面里的点
        q = c2w.T @ (scale * (r @ p) + t - np.array(view["pos"]))  # three.js 相机系，看向 -z
        u1, v1 = f * q[0] / -q[2], -f * q[1] / -q[2]
        # 位姿和四元数各自存了五六位小数，换算回来差在亚像素
        assert (u1, v1) == pytest.approx((u0, v0), abs=0.05)
        checked += 1
    assert checked > 30


def test_split_holds_out_every_eighth_sorted_name():
    names = [f"img_{i:03d}.jpg" for i in range(20)][::-1]      # 故意倒序
    train, held = data.split(names)
    assert sorted(names[i] for i in held) == ["img_000.jpg", "img_008.jpg", "img_016.jpg"]
    assert len(train) == 17 and not set(train) & set(held)


def test_read_colmap_roundtrip(tmp_path):
    sparse = tmp_path
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, 800, 600))           # PINHOLE
        f.write(struct.pack("<4d", 700.0, 710.0, 400.0, 300.0))
    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 2))
        for image_id, name, t in ((1, b"b.jpg", (1.0, 2.0, 3.0)), (2, b"a.jpg", (0.0, 0.0, 0.0))):
            f.write(struct.pack("<i7di", image_id, 1.0, 0.0, 0.0, 0.0, *t, 1))
            f.write(name + b"\x00")
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ddq", 10.0, 20.0, -1))
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q3d3BdQ", 7, 0.5, -0.5, 2.0, 10, 20, 30, 0.1, 1))
        f.write(struct.pack("<ii", 1, 0))
    cameras, points, colours = data.read_colmap(sparse)
    assert [c.name for c in cameras] == ["a.jpg", "b.jpg"]       # 按文件名排序
    assert cameras[1].centre == pytest.approx([-1.0, -2.0, -3.0])
    assert (cameras[0].fx, cameras[0].fy) == (700.0, 710.0)
    assert points.tolist() == [[0.5, -0.5, 2.0]] and colours.tolist() == [[10, 20, 30]]
