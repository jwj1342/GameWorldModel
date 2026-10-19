"""GLB 顶点读取：自己写的，所以得有测试盯着它读的是不是真的几何。

为什么自己写而不是装 trimesh：集群 venv 里没有任何 glTF 库，而有两处需要资产的
真实几何——#7 要用完整资产网格当配准模板（断掉「ICP 把可见面换了解成物体转了」
那一环），#64 要用资产标称尺寸替掉不稳的「同类中位数」判据。

这里不造假的 GLB，直接拿资产库里的真文件验：库不在就跳过，不假装通过。
"""
import os
from pathlib import Path

import numpy as np
import pytest

from gwm.assets.mesh import load_points

ROOT = os.environ.get("GWM_ASSETS")
pytestmark = pytest.mark.skipif(not ROOT or not (Path(ROOT) / "3d").is_dir(),
                                reason="没有资产库（GWM_ASSETS）")


def some_assets(n=6):
    return sorted((Path(ROOT) / "3d").rglob("*.glb"))[:n]


def test_读得出点而且点数对得上():
    for glb in some_assets():
        pts = load_points(glb, samples=500)
        assert pts is not None, f"读不动 {glb}"
        assert pts.shape == (500, 3), f"{glb} 形状不对：{pts.shape}"
        assert np.isfinite(pts).all(), f"{glb} 有非有限值"


def test_包围盒是合理的实物尺寸():
    """Kenney 的资产都是米制的小物件。读错字节序或者漏了节点变换，尺寸会离谱。"""
    for glb in some_assets():
        pts = load_points(glb, samples=800)
        size = pts.max(0) - pts.min(0)
        assert (size > 1e-4).all(), f"{glb} 某一维是扁的：{size}"
        assert size.max() < 100.0, f"{glb} 大得离谱：{size}"


def test_采样是面积加权的不是顶点重复():
    """低多边形资产的顶点很少，直接拿顶点当模板会在大面上一个点都没有。

    判据：采到的不重复点数要远多于顶点数量级能给的——这里用「采 2000 个点，
    不重复的超过 1500 个」来盯住「别退化成从几十个顶点里反复抽」。
    """
    for glb in some_assets(3):
        pts = load_points(glb, samples=2000)
        unique = len(np.unique(np.round(pts, 6), axis=0))
        assert unique > 1500, f"{glb} 采样退化了，只有 {unique} 个不重复点"


def test_同一个种子读两次结果一样():
    glb = some_assets(1)[0]
    assert np.allclose(load_points(glb, samples=300), load_points(glb, samples=300))


def test_不是_glb_就返回_None(tmp_path):
    bad = tmp_path / "x.glb"
    bad.write_bytes(b"not a glb at all")
    assert load_points(bad) is None
