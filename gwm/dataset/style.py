"""样式那一半：背景、光照、材质。和行为那一半共用场景，但走各自的随机源。

这个模块只读场景的几何事实（有哪些节点、各自多大），不读也不写任何运动字段。
行为那一半同理不碰这里的任何字段。两边的边界就是 program 里的 `style` 子树
加上每个节点的 `material`——核心主张第四条要求这两半能分开写、分开改，
生成器自己先守住这条，否则造出来的数据证明不了这件事。
"""
from __future__ import annotations
import random

# 和 schema 的 $defs.materialName 一一对应；kernel/scene.js 的 MATERIALS 是同一份表。
PALETTE = ["default", "grass", "stone", "wood", "metal", "concrete", "plastic", "rubber",
           "gold", "red", "blue", "green", "yellow", "white", "black", "lava", "water",
           "glass", "cardboard"]

# 材质三种写法的抽样比例。结构化那一档是 #44 新开的，真实视频里量出来的颜色只能走它，
# 所以训练数据必须有足够多的样本，不能全是调色板词。
MATERIAL_FORMS = (("palette", 0.45), ("hex", 0.20), ("surface", 0.35))

_SKY = ["#ffeedd", "#dfe9ff", "#fff6e0", "#e8f0ff", "#ffe9e4"]
_GROUND_TINT = ["#223344", "#2b2b22", "#1c2b1c", "#332a22", "#262633"]
_KEY = ["#fff2cc", "#ffffff", "#ffe0b0", "#e8f0ff", "#ffd9c0"]


# 结构件（地面、墙、平台、轨道）用低饱和色，物体才放开。三通道独立均匀取色会造出
# 洋红地板、荧光绿墙这种画面，和真实视频的差距是白白拉大的——合成到真实这道坎本来就够宽。
STRUCTURAL_PALETTE = ["stone", "concrete", "wood", "grass", "metal", "default", "white", "black"]


def _hex(rng: random.Random, muted: bool = False) -> str:
    if not muted:
        return "#%02x%02x%02x" % tuple(rng.randrange(24, 232) for _ in range(3))
    base = rng.randrange(70, 185)                       # 先定明度，再给每个通道小幅偏移
    return "#%02x%02x%02x" % tuple(min(235, max(30, base + rng.randrange(-20, 21))) for _ in range(3))


def sample_material(rng: random.Random, *, emissive_allowed: bool = True,
                    muted: bool = False) -> str | dict:
    """三种合法写法之一。结构化那条是给「颜色从视频里量出来」留的路。

    `muted` 给结构件用：偏向有自然名字的调色板词，取十六进制时也压住饱和度。
    """
    roll, acc = rng.random(), 0.0
    form = MATERIAL_FORMS[-1][0]
    for name, weight in MATERIAL_FORMS:
        acc += weight
        if roll < acc:
            form = name
            break
    if form == "palette":
        return rng.choice(STRUCTURAL_PALETTE if muted else PALETTE)
    if form == "hex":
        return _hex(rng, muted)
    surface: dict = {"base_color": _hex(rng, muted), "roughness": round(rng.uniform(0.15, 0.95), 3)}
    if rng.random() < 0.45:
        surface["metalness"] = round(rng.uniform(0.0, 0.9), 3)
    if emissive_allowed and rng.random() < 0.18:
        surface["emissive"] = _hex(rng)
        surface["emissive_intensity"] = round(rng.uniform(0.3, 2.0), 3)
    if rng.random() < 0.10:
        surface["opacity"] = round(rng.uniform(0.35, 0.95), 3)
    return surface


def sample_style(rng: random.Random) -> dict:
    """`style` 子树。缺省值和 kernel 写死的那一组逐项相同，所以不写也渲得出来；
    这里一律写满，是因为生成器要随机化光照，而真值程序必须记得住它（RP 5.8 缺口 14）。"""
    direction = [round(rng.uniform(-9, 9), 2), round(rng.uniform(5, 14), 2), round(rng.uniform(-9, 9), 2)]
    if sum(v * v for v in direction) < 1.0:          # 零向量会把灯摆在原点，语义校验直接拒
        direction = [4.0, 10.0, 6.0]
    return {
        "background": _hex(rng),
        "lighting": {
            "ambient": {"sky_color": rng.choice(_SKY), "ground_color": rng.choice(_GROUND_TINT),
                        "intensity": round(rng.uniform(0.25, 1.4), 3)},
            "key": {"color": rng.choice(_KEY), "intensity": round(rng.uniform(0.6, 3.2), 3),
                    "direction": direction, "shadow": rng.random() < 0.6},
        },
    }


def style_fields(program: dict) -> dict:
    """把一份程序的样式那一半摘出来，用于「同行为不同样式」这组配对的差异核对。"""
    return {
        "style": program.get("style"),
        "materials": {node["id"]: node.get("material")
                      for group in ("static", "objects") for node in program.get(group, [])},
    }
