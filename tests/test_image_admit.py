"""让图像分高的资产**进候选集**，而不只是在候选里重排。

为什么需要这条。真值改用资产库搭之后第一次量到检索的真实状况：

    拿**真值类名**当查询词   top-1 命中 42/61 = 0.689、top-5 0.820
    管线端到端（检测器说的词）  **0/71 = 0.000**

排序本身是好的，瓶颈整个在查询词上——**正确的资产根本没进候选集**。
这种情况下任何「候选内重排」（图像分、形状分）都从原理上没有机会：实测形状重排
换掉 64% 的资产，命中率 0.000 → 0.000。
"""
from dataclasses import dataclass

from gwm.assets.retrieve import retrieve


@dataclass
class A:
    id: str
    path: str = ""
    kit: str = "kit"
    kit_title: str = "Kit"
    name: str = ""


def cat(*names):
    return [A(id=f"kit/{n}", name=n) for n in names]


def test_不开的时候文字分为零的资产进不来():
    """默认行为：文字分决定收不收。这是原来的契约，不能被悄悄改掉。"""
    catalog = cat("chair", "table", "lamp")
    hits = retrieve("chair", catalog, k=5, image_scores={"kit/lamp": 0.99})
    assert all(a.id != "kit/lamp" for a, _s in hits)


def test_开了之后图像分最高的能进候选():
    catalog = cat("chair", "table", "lamp")
    hits = retrieve("chair", catalog, k=5,
                    image_scores={"kit/lamp": 0.99, "kit/table": 0.01},
                    image_admit=1)
    assert any(a.id == "kit/lamp" for a, _s in hits)


def test_查询词完全不沾边时也能靠图像捞回来():
    """这正是管线里 0/71 的那种情形：检测器说的词和资产名对不上。"""
    catalog = cat("bed-bunk", "chair", "table")
    hits = retrieve("zzz-nonsense", catalog, k=5,
                    image_scores={"kit/bed-bunk": 0.9}, image_admit=2)
    assert any(a.id == "kit/bed-bunk" for a, _s in hits)


def test_进来的那些文字分记为零():
    """文字分是阈值判据，不能因为图像分高就伪造它——否则 min_score 那道闸会失效。"""
    catalog = cat("chair", "lamp")
    hits = retrieve("chair", catalog, k=5, image_scores={"kit/lamp": 0.99}, image_admit=1)
    lamp = [s for a, s in hits if a.id == "kit/lamp"]
    assert lamp and lamp[0] == 0.0


def test_没有图像分时完全不变():
    catalog = cat("chair", "table")
    assert retrieve("chair", catalog, k=5, image_admit=5) == \
        retrieve("chair", catalog, k=5)
