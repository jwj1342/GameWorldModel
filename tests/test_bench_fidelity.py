"""保真度指标的口径检查。不加载 LPIPS 权重，只查 PSNR、SSIM 和出错路径。"""
import math

import numpy as np
import pytest
from PIL import Image

from gwm.bench import fidelity


def _save(path, value, size=(32, 24)):
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(path)


def test_identical_images_are_perfect(tmp_path):
    (tmp_path / "gt").mkdir(); (tmp_path / "r").mkdir()
    _save(tmp_path / "gt" / "a.png", 120); _save(tmp_path / "r" / "a.png", 120)
    result = fidelity.evaluate(tmp_path / "gt", tmp_path / "r", [{"name": "a.JPG"}])
    assert math.isinf(result["per_view"][0]["psnr"]) and result["per_view"][0]["ssim"] == 1.0
    assert "lpips" not in result["mean"]


def test_psnr_matches_the_definition():
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.full((4, 4, 3), 0.1, dtype=np.float32)
    assert fidelity.psnr(a, b) == pytest.approx(20.0, abs=1e-4)


def test_size_mismatch_and_missing_render_raise(tmp_path):
    (tmp_path / "gt").mkdir(); (tmp_path / "r").mkdir()
    _save(tmp_path / "gt" / "a.png", 120)
    with pytest.raises(FileNotFoundError):
        fidelity.evaluate(tmp_path / "gt", tmp_path / "r", [{"name": "a.png"}])
    _save(tmp_path / "r" / "a.png", 120, size=(16, 12))
    with pytest.raises(ValueError):
        fidelity.evaluate(tmp_path / "gt", tmp_path / "r", [{"name": "a.png"}])
