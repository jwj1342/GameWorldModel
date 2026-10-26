"""用观测到的边长比例挑资产——创新点一在管线里的落点。

为什么是这个落点：`shape_kind_accuracy` 在这条管线上**结构性恒零**
（`effective_shape()` 对解析到资产的节点返回 "mesh"，而真值全是基本体，93% 的物体
都解析到了资产），所以它量不出任何改进。真正有数的是 `shape_ok`——它比的正是
**三条边的比例**。而资产检索原来只看类名和图像，`Asset` 里一个几何字段都没有。
"""
from dataclasses import dataclass

import numpy as np

from gwm.assets.annotate import observed_edge_ratios, shape_rerank


@dataclass
class FakeAsset:
    id: str


def hits(*ids):
    return [(FakeAsset(i), 1.0) for i in ids]


def test_挑出比例最接近观测的那个():
    shapes = {"扁": [1.0, 0.9, 0.1], "方": [1.0, 0.95, 0.9], "细长": [1.0, 0.2, 0.15]}
    out = shape_rerank(hits("方", "扁", "细长"), [1.0, 0.9, 0.12], shapes, weight=0.9)
    assert out[0][0].id == "扁"


def test_观测拿不到比例时原样返回():
    shapes = {"a": [1.0, 0.5, 0.2]}
    h = hits("a", "b")
    assert shape_rerank(h, None, shapes) == h


def test_没有缓存时原样返回():
    h = hits("a", "b")
    assert shape_rerank(h, [1.0, 0.5, 0.2], {}) == h


def test_权重为零时不改变顺序():
    """weight=0 该完全让位给文字分，顺序不动——默认关的那一档必须真的不动。"""
    shapes = {"a": [1.0, 0.9, 0.9], "b": [1.0, 0.1, 0.1]}
    h = [(FakeAsset("a"), 0.9), (FakeAsset("b"), 0.4)]
    out = shape_rerank(h, [1.0, 0.1, 0.1], shapes, weight=0.0)
    assert [a.id for a, _ in out] == ["a", "b"]


def test_观测比例的口径和资产缓存一致():
    """两边都是「三边排序后除以最长边」。口径不一致的话这个重排就是在比噪声。"""
    obj = {"obb": [{"size": [0.4, 0.2, 0.1]}, {"size": [0.4, 0.2, 0.1]}]}
    r = observed_edge_ratios(obj)
    assert r is not None
    assert np.allclose(r, [1.0, 0.5, 0.25])


def test_没有obb时返回None():
    assert observed_edge_ratios({"obb": []}) is None
    assert observed_edge_ratios(None) is None


def test_choose_assets_only_reranks_text_eligible_candidates():
    from gwm.assets.annotate import choose_assets
    from gwm.assets.library import Asset
    catalog=[Asset('kit/cup','cup.glb','kit','kit','cup'),
             Asset('kit/plate','plate.glb','kit','cup','plate')]
    evidence={'objects':[{'id':'item','obb':[{'size':[1.,.1,.1]}]}]}
    for weight in (0,.5,.9,1):
        p={'objects':[{'id':'item','class':'cup','geom':{'kind':'asset','query':'cup','extent':[1.,.1,.1]}}]}
        result=choose_assets(p,evidence,[],{},catalog,min_score=.45,
            shapes={'kit/cup':[1.,1.,1.],'kit/plate':[1.,.1,.1]},shape_weight=weight)
        assert result[0]['chosen']=='kit/cup'
        assert p['objects'][0]['geom']['ref']=='kit/cup'


def test_no_text_eligible_candidate_still_falls_back():
    from gwm.assets.annotate import choose_assets
    from gwm.assets.library import Asset
    p={'objects':[{'id':'item','class':'cup','geom':{'kind':'asset','query':'cup','extent':[1.,.1,.1]}}]}
    evidence={'objects':[{'id':'item','obb':[{'size':[1.,.1,.1]}]}]}
    result=choose_assets(p,evidence,[],{},[Asset('kit/plate','plate.glb','kit','cup','plate')],
        min_score=.45,shapes={'kit/plate':[1.,.1,.1]},shape_weight=.9)
    assert result[0]['chosen'] is None
    assert 'ref' not in p['objects'][0]['geom']
