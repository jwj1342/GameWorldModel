"""几何误差：投影和深度解码要和 bench_view.js 的 depth 模式对得上。"""
import numpy as np
from PIL import Image

from gwm.bench import geometry


def _depth_png(path, depth_m):
    mm = np.round(np.asarray(depth_m) * 1000).astype(np.int64)
    rgb = np.stack([mm // 65536, (mm // 256) % 256, mm % 256], -1).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def test_point_straight_ahead_lands_in_the_centre_with_its_depth():
    view = {"name": "a.JPG", "pos": [0, 1, 0], "quat": [0, 0, 0, 1], "fov_deg": 60}
    u, v, z = geometry.project(view, np.array([[0.0, 1.0, -2.0], [0.0, 2.0, -2.0]]), 100, 80)
    assert np.allclose([u[0], v[0], z[0]], [50, 40, 2.0])
    assert v[1] < 40          # 比相机高的点在图像上半部分


def test_depth_error_and_missing_geometry(tmp_path):
    depth = np.full((80, 100), 2.05)
    depth[:, :50] = 0                       # 左半边没有几何
    _depth_png(tmp_path / "a.png", depth)
    decoded = geometry.decode_depth(tmp_path / "a.png")
    assert np.isnan(decoded[0, 0]) and abs(decoded[0, 99] - 2.05) < 1e-9
    view = {"name": "a.JPG", "pos": [0, 0, 0], "quat": [0, 0, 0, 1], "fov_deg": 60}
    points = np.array([[0.5, 0.0, -2.0], [-0.5, 0.0, -2.0]])   # 一个落在右半边，一个落在左半边
    row = geometry.evaluate_view(view, points, decoded)
    assert row["points"] == 2 and row["coverage"] == 0.5
    assert abs(row["median_abs_m"] - 0.05) < 1e-6 and row["within_10cm"] == 0.5
