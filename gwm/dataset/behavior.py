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


# 绕自身竖轴转、而且位置不变的两类（spin、periodic_rotate）只有在形状**不是**绕该轴
# 对称时才看得见。球、圆柱、圆锥绕 y 轴转，画面上一帧都不会变——这种样本按构造就观测不到，
# 任何方法在它上面都只能瞎猜，会把表 1 整列变成噪声。RP 5.8 写明「测试集必须按可测性配」。
_ROTATION_IN_PLACE = {"spin", "periodic_rotate"}


def _geom(rng: random.Random, shape: str, distinct_xz: bool = False) -> tuple[dict, float, float]:
    """返回 (geom, 半高, 水平半径)。半高用来把物体摆到地面以上，水平半径用来排开间距。

    `distinct_xz` 要求俯视轮廓明显不是正方形，这样绕竖轴的转动才有可观测的姿态变化。
    """
    if shape == "box":
        x = round(rng.uniform(0.4, 2.2), 3)
        z = round(rng.uniform(0.4, 2.2), 3)
        if distinct_xz:                       # 长宽比拉开，俯视轮廓转起来才看得出朝向
            z = round(min(max(x * rng.uniform(2.0, 3.5), 0.3), 3.0), 3)
            if max(x, z) / min(x, z) < 1.6:
                z = round(x * 2.0, 3)
        extent = [x, round(rng.uniform(0.4, 2.0), 3), z]
        return {"kind": "primitive", "shape": "box", "extent": extent}, extent[1] / 2, max(extent[0], extent[2]) / 2
    if shape == "sphere":
        radius = round(rng.uniform(0.25, 1.1), 3)
        return {"kind": "primitive", "shape": "sphere", "radius": radius}, radius, radius
    radius, height = round(rng.uniform(0.25, 1.0), 3), round(rng.uniform(0.6, 2.4), 3)
    return ({"kind": "primitive", "shape": shape, "radius": radius, "height": height, "axis": "y"},
            height / 2, radius)


# —— 用资产库搭真值场景 ——
#
# 为什么要这条路（#82）。现在的真值是「从词表随机挑个名字 + 独立挑个基本体」，
# 于是两件事同时成立：
#
#   * **「该挑哪个资产」没有标准答案**——真值里压根没有资产，检索改进测不出来。
#     实测：把检索重排改掉，86 个资产节点换了 55 个（64%），而所有指标逐字节不动。
#   * **类名和形状脱钩**，所以 `class_accuracy` 恒等于噪声（`gt_metrics.py` 的注释
#     自己写明了「别拿它当成绩也别拿它调参」）。
#
# 真值改成从**我们检索的同一个资产库**里挑，两件事一起修好：资产 id 可以直接比对，
# 类名从资产名来所以和形状天然绑定。
#
# 默认关。打开要给 `assets.library`，库不在就自动退回基本体，生成照常跑。

_ASSET_POOL: list | None = None


def _asset_pool(cfg: dict | None = None) -> list:
    """(资产 id, 可读类名, 边长比例) 的列表。没有库就是空的。"""
    global _ASSET_POOL
    if _ASSET_POOL is None:
        _ASSET_POOL = []
        try:
            import json as _json
            import os
            from pathlib import Path as _Path
            from ..assets.library import library_root, load_catalog
            root = library_root(cfg or {})
            shapes_path = ((cfg or {}).get("assets") or {}).get("shape_cache") or \
                os.environ.get("GWM_ASSET_SHAPES")
            shapes = {}
            if shapes_path:
                try:
                    shapes = _json.loads(_Path(shapes_path).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    shapes = {}
            if root is not None:
                for a in load_catalog(root):
                    r = shapes.get(a.id)
                    if isinstance(r, list) and len(r) == 3:
                        # 类名取文件名，把分隔符换成空格——它会成为真值的 class，
                        # 检索那一侧拿到的就是这个词，所以名字和网格天然绑定。
                        name = str(a.name).replace("-", " ").replace("_", " ").strip().lower()
                        if name:
                            _ASSET_POOL.append((a.id, name, r))
        except Exception:                      # noqa: BLE001 — 没有库不是错误
            _ASSET_POOL = []
    return _ASSET_POOL


def _asset_geom(rng: random.Random, cfg: dict | None = None):
    """从资产库挑一个，返回 (geom, 半高, 半径, 类名)。库不可用时返回 None。

    尺寸按比例缩到和基本体那一路同一个量级，这样场景的尺度分布不变——
    只换「长什么样」，不换「多大」，否则两批数据连相机取景都没法比。
    """
    pool = _asset_pool(cfg)
    if not pool:
        return None
    asset_id, name, ratios = rng.choice(pool)
    longest = round(rng.uniform(0.5, 2.2), 3)          # 和 _geom 的盒子同量级
    ext = [round(longest * r, 3) for r in ratios]
    rng.shuffle(ext)                                   # 最长边不总是 x
    ext = [max(v, 0.05) for v in ext]
    # schema 的 assetGeom 要求 kind/query/extent 三样齐全（ref 可选）。
    # 第一版只给了 kind/ref/extent，12 条片段全部校验失败、写出 0 个程序。
    # query 给资产的可读名字——真值里它就是「这个物体叫什么」，
    # 和检索那一侧拿到的查询词是同一个口径。
    geom = {"kind": "asset", "query": name[:80], "ref": asset_id, "extent": ext}
    return geom, ext[1] / 2, max(ext[0], ext[2]) / 2, name


def _class_name(rng: random.Random) -> str:
    pool = BINDING_CLASSES if rng.random() < BINDING_SHARE else NEUTRAL_CLASSES
    return rng.choice(pool)


def sample_layout(rng: random.Random, family: str, stage_rng: random.Random) -> tuple[list[dict], dict]:
    """地面与固定结构。地面顶面一律在 y=0，物体摆放只需加自己的半高。

    场地的长宽由 `stage_rng`（观测那一路）抽，不走行为那一路。相机的取景距离是场地
    尺度的函数，两者同源才能做到「只换行为种子，相机一帧不动」——否则换个行为就连带
    换了机位，配对出来的两段视频差在哪就说不清了。场地之上摆什么、怎么动，仍然全在
    行为那一路。
    """
    statics: list[dict] = []
    if family == "corridor":
        width, depth = round(stage_rng.uniform(4.0, 7.0), 2), round(stage_rng.uniform(18.0, 30.0), 2)
    elif family == "track":
        width, depth = round(stage_rng.uniform(16.0, 26.0), 2), round(stage_rng.uniform(16.0, 26.0), 2)
    else:
        width, depth = round(stage_rng.uniform(10.0, 22.0), 2), round(stage_rng.uniform(10.0, 22.0), 2)
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
        rails = rng.randint(6, 10)
        for i in range(rails):
            # 角度按实际条数均分。写死除以 8 的话，9 条时 rail_9 会整个压在 rail_1 上，
            # 10 条时压两条——重复位姿既是坏几何，也会把评测的物体配对搞乱。
            angle = 2 * math.pi * i / rails
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
                  half_height: float, bounds: tuple[float, float] = (1e9, 1e9),
                  diag: float = 1.0) -> dict:
    """按类型产出内核认得的参数。每一类都给够驱动字段，不留「写了类型但不会动」的样本。

    `bounds` 是场地的半宽半深。脚本运动不走物理，写出界的关键帧不会被任何东西挡住——
    物体会直接飘出地面，或者从墙里穿过去。这种画面物理上不成立，学了就是学错，所以夹住。

    `diag` 是物体自身的对角线。平移类的位移必须大过它一截：位移比物体自己还小的时候，
    质心的那点变化在任何方法眼里都和静止没区别，这种样本只会把表 1 变成噪声。
    用物体自身尺寸作尺度是刻意的——不去对着评测器的阈值调，那样等于拿数据迁就指标。
    """
    limit_x, limit_z = bounds
    stride = max(0.6 * diag, 0.5)          # 一步至少走过自己大半个身位
    if kind == "static":
        return {"type": "static"}
    if kind == "trajectory":
        count = rng.randint(3, 5)
        keyframes = []
        x, z = base[0], base[2]
        heading = rng.uniform(0, 2 * math.pi)
        for i in range(count):
            t = round(duration * i / (count - 1), 3)
            heading += rng.uniform(-0.9, 0.9)           # 带转折但不来回抖，净位移才留得住
            step = stride * rng.uniform(1.0, 2.0)
            # 碰到边界要反弹，不能夹住。夹住的话靠边起步的物体会原地不动，
            # 写着 trajectory 其实和静止一样——又是一个观测不到的样本。
            for _ in range(4):
                nx, nz = x + step * math.cos(heading), z + step * math.sin(heading)
                if abs(nx) <= limit_x and abs(nz) <= limit_z:
                    break
                heading += math.pi / 2 + rng.uniform(-0.4, 0.4)
            else:
                nx, nz = min(max(nx, -limit_x), limit_x), min(max(nz, -limit_z), limit_z)
            x, z = nx, nz
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
        travel = round(min(rng.uniform(stride, stride + 3.0), max(room, stride)), 2)
        motion = {"type": "prismatic", "axis": axis, "from": 0.0}
        if rng.random() < 0.6:
            motion["schedule"] = [{"t": round(rng.uniform(0.3, duration * 0.35), 2), "to": travel,
                                   "duration": round(rng.uniform(1.0, 3.0), 2)}]
        else:
            motion["rate"] = round(travel / max(duration, 1e-3), 3)
        return motion
    if kind == "periodic_translate":
        amp = round(rng.uniform(stride * 0.7, stride * 0.7 + 1.6), 2)
        # 先定振幅再挑轴：竖直那一档只有在离地高度放得下这个振幅时才选，
        # 否则就被地面夹小成一个看不出来的抖动。宁可换个水平轴。
        axes = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        if base[1] - half_height >= amp:
            axes.append([0.0, 1.0, 0.0])
        return {"type": "periodic_translate", "axis": rng.choice(axes), "amp": amp,
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
                   count: int, cfg: dict | None = None) -> list[dict]:
    weights = MOTION_WEIGHTS[profile]
    # 留边按比例给，不是固定 1.5 米。走廊只有 4 米宽的时候，固定边距会把可活动范围
    # 压到半米，物体写着在动其实挪不开——又一类观测不到的样本。
    margin_w = min(1.5, layout["width"] * 0.15)
    margin_d = min(1.5, layout["depth"] * 0.15)
    half_w, half_d = layout["width"] / 2 - margin_w, layout["depth"] / 2 - margin_d
    objects, taken = [], []
    for i in range(count):
        # 先抽运动，再按运动挑形状。顺序反过来就会造出球体自转这种按构造观测不到的样本。
        kind = rng.choices(list(weights), weights=list(weights.values()))[0]
        # 真值用资产库搭：`dataset.use_assets` 打开之后，几何和类名都从库里来，
        # 于是「该挑哪个资产」有了标准答案、类名也和形状绑定（见 _asset_geom）。
        # 库不可用时自动退回基本体，生成照常跑。
        asset = (_asset_geom(rng, cfg)
                 if ((cfg or {}).get("dataset") or {}).get("use_assets") else None)
        if asset is not None:
            geom, half_height, radius, _asset_class = asset
        elif kind in _ROTATION_IN_PLACE:
            geom, half_height, radius = _geom(rng, "box", distinct_xz=True)
            _asset_class = None
        else:
            geom, half_height, radius = _geom(rng, rng.choice(_SHAPES))
            _asset_class = None
        extent = geom.get("extent") or [2 * radius, 2 * half_height, 2 * radius]
        diag = math.sqrt(sum(v * v for v in extent))
        for _ in range(24):                      # 简单排开：水平距离大于两者半径之和
            x, z = round(rng.uniform(-half_w, half_w), 2), round(rng.uniform(-half_d, half_d), 2)
            if all(math.hypot(x - px, z - pz) > radius + pr + 0.4 for px, pz, pr in taken):
                break
        taken.append((x, z, radius))
        lift = half_height + 0.02
        if kind == "dynamic":
            lift = half_height + round(rng.uniform(1.0, 4.0), 2)      # 落体要有下落的余量
        elif kind == "periodic_translate":
            lift = half_height + round(rng.uniform(0.8, 2.6), 2)      # 竖直振荡要留出下探空间
        elif rng.random() < 0.25:
            lift = half_height + round(rng.uniform(0.5, 2.5), 2)      # 一部分悬空物体，和贴地的分开
        pos = [x, round(lift, 3), z]
        objects.append({"id": f"obj_{i + 1}",
                        "class": _asset_class or _class_name(rng), "geom": geom,
                        "pose": {"pos": pos},
                        "motion": sample_motion(rng, kind, pos, duration, half_height,
                                                bounds=(half_w, half_d), diag=diag)})
    return objects
