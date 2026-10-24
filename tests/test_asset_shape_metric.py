"""评测要看得见「挑了哪个资产」——现在所有指标都看不见。

实测：把资产检索的重排改掉，**选中的资产确实变了**（minigolf/ball-blue →
food/rice-ball、car/debris-door → castle/door），而**每一个现有指标都逐字节相同**：

  shape_kind_accuracy         资产节点一律记成 "mesh"，恒零
  shape_ok / scale_ok / shape_error   读的是节点 extent，而它来自**观测**不是网格
  object_recall / precision   只看位置
  class_accuracy              只看类名

没有任何一个指标看得见网格本身，而资产决定了场景实际长什么样。
`asset_shape_error` 补的就是这个缺口。
"""
import numpy as np

from gwm.feedback import gt_metrics
from gwm.feedback.gt_metrics import asset_shape_error


def gt_box(extent):
    return {"geom": {"kind": "primitive", "shape": "box", "extent": list(extent)}}


def asset_node(ref):
    return {"geom": {"kind": "asset", "query": "x", "ref": ref, "extent": [1, 1, 1]}}


def with_table(monkeypatch, table):
    monkeypatch.setattr(gt_metrics, "_ASSET_SHAPES", table)


def test_形状一致时误差接近零(monkeypatch):
    with_table(monkeypatch, {"kit/flat": [1.0, 0.5, 0.25]})
    err = asset_shape_error(gt_box([0.8, 0.4, 0.2]), asset_node("kit/flat"))
    assert err is not None and err < 1e-6


def test_形状差很多时误差大(monkeypatch):
    with_table(monkeypatch, {"kit/cube": [1.0, 1.0, 1.0]})
    err = asset_shape_error(gt_box([1.0, 0.2, 0.1]), asset_node("kit/cube"))
    assert err is not None and err > 0.4


def test_没解析到资产时返回None(monkeypatch):
    with_table(monkeypatch, {"kit/cube": [1.0, 1.0, 1.0]})
    node = {"geom": {"kind": "asset", "query": "x", "extent": [1, 1, 1]}}   # 没有 ref
    assert asset_shape_error(gt_box([1, 1, 1]), node) is None


def test_基本体节点返回None(monkeypatch):
    """这一项只问「挑的那个网格对不对」，基本体节点不在它的范围里。"""
    with_table(monkeypatch, {"kit/cube": [1.0, 1.0, 1.0]})
    assert asset_shape_error(gt_box([1, 1, 1]), gt_box([1, 1, 1])) is None


def test_缓存里没有这个资产时返回None(monkeypatch):
    with_table(monkeypatch, {})
    assert asset_shape_error(gt_box([1, 1, 1]), asset_node("kit/unknown")) is None


def test_对整体缩放不敏感(monkeypatch):
    """口径和 shape_scale_agreement 一致：比的是比例，不是绝对尺寸。"""
    with_table(monkeypatch, {"kit/flat": [1.0, 0.5, 0.25]})
    a = asset_shape_error(gt_box([4.0, 2.0, 1.0]), asset_node("kit/flat"))
    b = asset_shape_error(gt_box([0.4, 0.2, 0.1]), asset_node("kit/flat"))
    assert a is not None and b is not None and np.isclose(a, b)
