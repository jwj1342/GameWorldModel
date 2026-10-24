"""贴地接触点修正只排除**地面本身**，不排除所有结构件。

这条拆的是一个会反咬一口的耦合（#56）。

接触点修正（加上中心序列时间平滑）一共值 **+0.143 米**：拿 15 个真值结构件量，
产物里带后处理的首帧框心离真值 0.277 m（容差 0.45 内 0.867），现算的、不带后处理
的首帧是 0.420 m（0.533）。

而原来的守卫用的是 `is_static_class`，它在当前数据上只对 **5.2%** 的轨迹为真
（193 条里 10 条，全是 `ground` 和 `rail`）。所以几乎所有轨迹都吃到了这个修正，
**包括那 15 个真值结构件——它们恰恰是因为类名没被认出来才吃到的**。

于是会出现这种事：哪天把「结构件被归进 objects」补好，这些节点立刻失去 0.143 米
的修正，修好一个问题反而让另一个变差。所以先把耦合拆掉。

排除地面本身仍然对：地面就是地平面，拿「掩码最低点与地面的交点」当它的中心没有
意义。用 `is_ground_class`（完全相等）而不是 `is_static_class`（按词匹配），
强度正好对应这个更窄的意图。
"""
import inspect

from gwm.perception import evidence
from gwm.taxonomy import is_ground_class, is_static_class


def test_守卫用的是地面判据不是结构件判据():
    src = inspect.getsource(evidence.build_evidence)
    assert "if use_contact and not is_ground:" in src, \
        "接触点守卫要用地面判据；用结构件判据会让结构件一旦被认出来就失去这个修正"
    assert "if use_contact and not is_structure:" not in src


def test_两个判据的强度差别正是这里要的():
    """rail 是结构件但不是地面——它该拿到接触点修正。"""
    assert is_static_class("rail") and not is_ground_class("rail")
    assert is_static_class("ground") and is_ground_class("ground")
    # 按词匹配 vs 完全相等：这正是两者强度不同的地方
    assert is_static_class("wooden floor") and not is_ground_class("wooden floor")


def test_结构件判据仍然在别处用着():
    """只改接触点那一处守卫，别把 is_structure 整个删掉——它还有别的用途。"""
    src = inspect.getsource(evidence.build_evidence)
    assert "is_structure = is_static_class(" in src
