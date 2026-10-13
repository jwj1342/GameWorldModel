"""行为那一半：场景结构、物体几何、运动脚本。不碰 style，也不碰 material。

场景家族分成互不相交的两组（RP 5.8 的跨生成器硬要求）：训练只用第一组，
测试集从第二组生成。两组的地面形状、围挡方式、物体摆放规律都不一样，
不这么分的话主表全是同分布结果，说明不了泛化。
"""
from __future__ import annotations
import math
import random

MOTION_TYPES = ["static", "trajectory", "revolute", "prismatic",
                "periodic_translate", "periodic_rotate", "spin", "dynamic"]

TRAIN_FAMILIES = ("room", "platform", "corridor")
TEST_FAMILIES = ("yard", "track")

# 测试集按可测性配：八类等权，保证表 1 的混淆结构每一格都有样本。
# 训练集按常见度配：真实场景里静止和周期运动多，自由落体少。两套必须分开说明。
MOTION_WEIGHTS = {
    "test": {k: 1.0 for k in MOTION_TYPES},
    "train": {"static": 2.4, "trajectory": 1.4, "revolute": 0.8, "prismatic": 0.8,
              "periodic_translate": 1.4, "periodic_rotate": 0.9, "spin": 1.2, "dynamic": 0.6},
}

# 类名池故意比绑定词表宽。#5 点名的坑是「DSL 表达不了的东西在数据里不会出现，
# 分数自然好看」，其中最具体的一条就是场景里的 lava_pit 和 coin 恰好命中绑定词表。
# 这里留一小部分命中、大部分不命中，并把命中比例写进 manifest，让它成为可报告的数而不是暗坑。
NEUTRAL_CLASSES = ["crate", "barrel", "pillar", "drum", "beacon", "pod", "canister", "bollard",
                   "slab", "totem", "planter", "cone_marker", "tank", "spool", "girder", "buoy"]
BINDING_CLASSES = ["coin", "gem", "star", "ball", "marble", "lava", "spike"]
BINDING_SHARE = 0.22

_SHAPES = ["box", "sphere", "cylinder", "cone"]


def _geom(rng: random.Random, shape: str) -> tuple[dict, float, float]:
    """返回 (geom, 半高, 水平半径)。半高用来把物体摆到地面以上，水平半径用来排开间距。"""
    if shape == "box":
        extent = [round(rng.uniform(0.4, 2.2), 3), round(rng.uniform(0.4, 2.0), 3), round(rng.uniform(0.4, 2.2), 3)]
        return {"kind": "primitive", "shape": "box", "extent": extent}, extent[1] / 2, max(extent[0], extent[2]) / 2
    if shape == "sphere":
        radius = round(rng.uniform(0.25, 1.1), 3)
        return {"kind": "primitive", "shape": "sphere", "radius": radius}, radius, radius
    radius, height = round(rng.uniform(0.25, 1.0), 3), round(rng.uniform(0.6, 2.4), 3)
    return ({"kind": "primitive", "shape": shape, "radius": radius, "height": height, "axis": "y"},
            height / 2, radius)


def _class_name(rng: random.Random) -> str:
    pool = BINDING_CLASSES if rng.random() < BINDING_SHARE else NEUTRAL_CLASSES
    return rng.choice(pool)


def sample_layout(rng: random.Random, family: str) -> tuple[list[dict], dict]:
    """地面与固定结构。地面顶面一律在 y=0，物体摆放只需加自己的半高。"""
    statics: list[dict] = []
    if family == "corridor":
        width, depth = round(rng.uniform(4.0, 7.0), 2), round(rng.uniform(18.0, 30.0), 2)
    elif family == "track":
        width, depth = round(rng.uniform(16.0, 26.0), 2), round(rng.uniform(16.0, 26.0), 2)
    else:
        width, depth = round(rng.uniform(10.0, 22.0), 2), round(rng.uniform(10.0, 22.0), 2)
    statics.append({"id": "ground", "class": "ground",
                    "geom": {"kind": "primitive", "shape": "box", "extent": [width, 0.4, depth]},
                    "pose": {"pos": [0.0, -0.2, 0.0]}})

    if family == "room":
        half_w, half_d, height = width / 2, depth / 2, round(rng.uniform(2.6, 4.2), 2)
        for name, pos, extent in (
                ("wall_n", [0.0, height / 2, -half_d], [width, height, 0.3]),
                ("wall_s", [0.0, height / 2, half_d], [width, height, 0.3]),
                ("wall_e", [half_w, height / 2, 0.0], [0.3, height, depth]),
                ("wall_w", [-half_w, height / 2, 0.0], [0.3, height, depth])):
            statics.append({"id": name, "class": "wall",
                            "geom": {"kind": "primitive", "shape": "box", "extent": extent},
                            "pose": {"pos": [round(v, 3) for v in pos]}})
    elif family == "corridor":
        half_w, half_d, height = width / 2, depth / 2, round(rng.uniform(2.4, 3.6), 2)
        for side, name in ((-1, "wall_w"), (1, "wall_e")):
            statics.append({"id": name, "class": "wall",
                            "geom": {"kind": "primitive", "shape": "box", "extent": [0.3, height, depth]},
                            "pose": {"pos": [round(side * half_w, 3), round(height / 2, 3), 0.0]}})
    elif family == "platform":
        for i in range(rng.randint(2, 4)):
            extent = [round(rng.uniform(2.0, 4.5), 2), round(rng.uniform(0.3, 0.6), 2), round(rng.uniform(2.0, 4.5), 2)]
            statics.append({"id": f"platform_{i + 1}", "class": "platform",
                            "geom": {"kind": "primitive", "shape": "box", "extent": extent},
                            "pose": {"pos": [round(rng.uniform(-width / 3, width / 3), 2),
                                             round(rng.uniform(0.8, 3.0), 2),
                                             round(rng.uniform(-depth / 3, depth / 3), 2)]}})
    elif family == "track":
        radius = round(min(width, depth) / 2 - 1.5, 2)
        for i in range(rng.randint(6, 10)):
            angle = 2 * math.pi * i / 8
            statics.append({"id": f"rail_{i + 1}", "class": "rail",
                            "geom": {"kind": "primitive", "shape": "box", "extent": [1.6, 0.25, 0.5]},
                            "pose": {"pos": [round(radius * math.cos(angle), 2), 0.125, round(radius * math.sin(angle), 2)]}})
    elif family == "yard":
        for i in range(rng.randint(1, 3)):
            extent = [round(rng.uniform(1.5, 3.5), 2), round(rng.uniform(1.0, 2.6), 2), round(rng.uniform(1.5, 3.5), 2)]
            statics.append({"id": f"block_{i + 1}", "class": "block",
                            "geom": {"kind": "primitive", "shape": "box", "extent": extent},
                            "pose": {"pos": [round(rng.uniform(-width / 3, width / 3), 2), round(extent[1] / 2, 3),
                                             round(rng.uniform(-depth / 3, depth / 3), 2)]}})
    return statics, {"width": width, "depth": depth, "family": family}


def sample_motion(rng: random.Random, kind: str, base: list[float], duration: float,
                  half_height: float, bounds: tuple[float, float] = (1e9, 1e9)) -> dict:
    """按类型产出内核认得的参数。每一类都给够驱动字段，不留「写了类型但不会动」的样本。

    `bounds` 是场地的半宽半深。脚本运动不走物理，写出界的关键帧不会被任何东西挡住——
    物体会直接飘出地面，或者从墙里穿过去。这种画面物理上不成立，学了就是学错，所以夹住。
    """
    limit_x, limit_z = bounds
    if kind == "static":
        return {"type": "static"}
    if kind == "trajectory":
        count = rng.randint(3, 5)
        keyframes = []
        x, z = base[0], base[2]
        for i in range(count):
            t = round(duration * i / (count - 1), 3)
            x = min(max(x + rng.uniform(-3.0, 3.0), -limit_x), limit_x)
            z = min(max(z + rng.uniform(-3.0, 3.0), -limit_z), limit_z)
            keyframes.append({"t": t, "pos": [round(x, 3), round(base[1], 3), round(z, 3)]})
        return {"type": "trajectory", "keyframes": keyframes,
                "interp": "catmull_rom" if rng.random() < 0.5 else "linear"}
    if kind == "revolute":
        # 绕竖轴转的门：转动中高度不变，不会穿进地面
        span = round(rng.uniform(45, 140), 1)
        pivot = [round(base[0] + rng.uniform(-1.2, 1.2), 3), round(base[1], 3), round(base[2] + rng.uniform(-1.2, 1.2), 3)]
        motion = {"type": "revolute", "axis": [0.0, 1.0, 0.0], "pivot": pivot,
                  "range_deg": [0.0, span], "from_deg": 0.0}
        if rng.random() < 0.6:
            motion["schedule"] = [{"t": round(rng.uniform(0.5, duration * 0.4), 2), "to_deg": span,
                                   "duration": round(rng.uniform(0.8, 2.5), 2)}]
        else:
            motion["rate_dps"] = round(span / max(duration, 1e-3), 2)
        return motion
    if kind == "prismatic":
        axis = rng.choice([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        room = {0: limit_x - abs(base[0]), 2: limit_z - abs(base[2])}.get(axis.index(1.0), 1e9)
        travel = round(min(rng.uniform(1.0, 4.0), max(room, 0.3)), 2)
        motion = {"type": "prismatic", "axis": axis, "from": 0.0}
        if rng.random() < 0.6:
            motion["schedule"] = [{"t": round(rng.uniform(0.3, duration * 0.35), 2), "to": travel,
                                   "duration": round(rng.uniform(1.0, 3.0), 2)}]
        else:
            motion["rate"] = round(travel / max(duration, 1e-3), 3)
        return motion
    if kind == "periodic_translate":
        axis = rng.choice([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        # 竖直振荡的振幅不能把物体压进地面：基准高度已经留够了，这里再卡一次
        amp = round(rng.uniform(0.5, 2.2), 2)
        if axis[1] == 1.0:
            amp = round(min(amp, max(base[1] - half_height, 0.2)), 2)
        return {"type": "periodic_translate", "axis": axis, "amp": max(amp, 0.2),
                "period": round(rng.uniform(1.5, 5.0), 2), "phase": round(rng.uniform(0, 6.28), 3)}
    if kind == "periodic_rotate":
        return {"type": "periodic_rotate", "axis": [0.0, 1.0, 0.0],
                "amp_deg": round(rng.uniform(15, 80), 1), "period": round(rng.uniform(1.5, 5.0), 2),
                "phase": round(rng.uniform(0, 6.28), 3)}
    if kind == "spin":
        return {"type": "spin", "axis": [0.0, 1.0, 0.0], "rate_dps": round(rng.choice([-1, 1]) * rng.uniform(30, 220), 1)}
    # dynamic：位姿由物理决定，脚本参数一律不写，否则语义校验会报 ignored_params
    return {"type": "dynamic", "mass": round(rng.uniform(0.5, 8.0), 2),
            "restitution": round(rng.uniform(0.05, 0.8), 2),
            "linear_velocity": [round(rng.uniform(-2.5, 2.5), 2), 0.0, round(rng.uniform(-2.5, 2.5), 2)]}


def sample_objects(rng: random.Random, layout: dict, profile: str, duration: float,
                   count: int) -> list[dict]:
    weights = MOTION_WEIGHTS[profile]
    half_w, half_d = layout["width"] / 2 - 1.5, layout["depth"] / 2 - 1.5
    objects, taken = [], []
    for i in range(count):
        shape = rng.choice(_SHAPES)
        geom, half_height, radius = _geom(rng, shape)
        for _ in range(24):                      # 简单排开：水平距离大于两者半径之和
            x, z = round(rng.uniform(-half_w, half_w), 2), round(rng.uniform(-half_d, half_d), 2)
            if all(math.hypot(x - px, z - pz) > radius + pr + 0.4 for px, pz, pr in taken):
                break
        taken.append((x, z, radius))
        kind = rng.choices(list(weights), weights=list(weights.values()))[0]
        lift = half_height + 0.02
        if kind == "dynamic":
            lift = half_height + round(rng.uniform(1.0, 4.0), 2)      # 落体要有下落的余量
        elif kind == "periodic_translate":
            lift = half_height + round(rng.uniform(0.8, 2.6), 2)      # 竖直振荡要留出下探空间
        elif rng.random() < 0.25:
            lift = half_height + round(rng.uniform(0.5, 2.5), 2)      # 一部分悬空物体，和贴地的分开
        pos = [x, round(lift, 3), z]
        objects.append({"id": f"obj_{i + 1}", "class": _class_name(rng), "geom": geom,
                        "pose": {"pos": pos},
                        "motion": sample_motion(rng, kind, pos, duration, half_height,
                                                bounds=(half_w, half_d))})
    return objects
