"""派生配对必须带上基准的全部设定，尤其是 `stage_scale`。

#52（样式-行为解耦度）整个指标建立在一个不变量上：`style_variant` 这组「同行为、
只换样式」。实测这个不变量**根本不成立**——六组里六组行为都不同，差的正好是
6.667 = 1/0.15 倍。

根因是 `with_style` / `with_behavior` 逐字段重建 SceneSpec，把 `stage_scale` 漏了，
于是退回默认 1.0：基准缩到了 0.15，派生出来的没缩。

逐字段重建是这类 bug 的温床——以后往 SceneSpec 加字段一样会被静默漏掉。所以改成
`dataclasses.replace`，并且这里用**反射**来盯：除了显式要改的那几个字段，其余
**全部**必须一致，这样新加字段自动受保护。
"""
import dataclasses

from gwm.dataset.generator import SceneSpec


def base():
    return SceneSpec(clip="test_00000", family="yard", behavior_seed=1, style_seed=2,
                     camera_seed=3, profile="test", duration=8.0, fps=30,
                     tags=("base",), stage_scale=0.15)


def test_换样式只动样式种子和标签():
    b = base()
    v = b.with_style(999, "test_00000_style")
    changed = {f.name for f in dataclasses.fields(SceneSpec)
               if getattr(b, f.name) != getattr(v, f.name)}
    assert changed == {"clip", "style_seed", "tags"}, \
        f"除了这三个字段，其余都该原样带过来，实际变了：{changed}"


def test_换行为只动行为种子和标签():
    b = base()
    v = b.with_behavior(999, "test_00000_behavior")
    changed = {f.name for f in dataclasses.fields(SceneSpec)
               if getattr(b, f.name) != getattr(v, f.name)}
    assert changed == {"clip", "behavior_seed", "tags"}, \
        f"除了这三个字段，其余都该原样带过来，实际变了：{changed}"


def test_stage_scale_必须带过去():
    """这就是当初漏掉的那个字段，单独钉一条。"""
    b = base()
    assert b.with_style(999, "s").stage_scale == 0.15
    assert b.with_behavior(999, "b").stage_scale == 0.15


def test_换样式保住行为种子_换行为保住样式种子():
    b = base()
    assert b.with_style(999, "s").behavior_seed == b.behavior_seed
    assert b.with_behavior(999, "b").style_seed == b.style_seed
    # 观测那一路两边都不该动
    assert b.with_style(999, "s").camera_seed == b.camera_seed
    assert b.with_behavior(999, "b").camera_seed == b.camera_seed


def test_标签是追加不是替换():
    b = base()
    assert b.with_style(999, "s").tags == ("base", "style_variant")
    assert b.with_behavior(999, "b").tags == ("base", "behavior_variant")
