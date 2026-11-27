"""铰接物体的状态差分样本：柜门、抽屉、箱盖、烤箱门，用基本体搭。

每条样本是一个家具：柜体当 static 节点，门、抽屉、盖子是 object 节点。部件里有一部分在 [0, T] 内
从关着打开（revolute 绕铰链转、prismatic 沿导轨滑），其余不动。相机固定，所以首末两帧只差部件的状态。

朝向只取四个正交方向，铰链坐标是整齐的数，难点留在「哪个部件动、铰链在哪条边、往哪开、开多大」上，
而不是坐标旋转的算术。

家族：train 有 cabinet（单门或对开门）、drawers（一列抽屉）、chest（顶盖）；
test 是训练里没有的构型：oven（门绕底边往下翻）、combo（对开门下面一个抽屉，门和抽屉混着动）。
"""
from __future__ import annotations

import math
import random

from .style import sample_style

TRAIN_FAMILIES = ("cabinet", "drawers", "chest")
TEST_FAMILIES = ("oven", "combo")
DURATION = 4.0
_MARGIN, _GAP, _THICK = 0.02, 0.01, 0.035
_BODY_COLORS = ["#c8b7a0", "#8a6a4a", "#d9d9d6", "#5a5f66", "#b9c4b0", "#e8dcc4", "#6f4e37", "#a3a8ad"]
_PART_COLORS = ["#c0392b", "#2e86c1", "#27ae60", "#f1c40f", "#8e44ad", "#e67e22", "#16a085", "#d35400", "#2c3e50"]
_NAMES = {"cabinet": "cabinet", "drawers": "chest of drawers", "chest": "chest", "oven": "oven", "combo": "cabinet"}
_NUM = ["zero", "one", "two", "three", "four", "five"]


def _rot_y(v, deg):
    a = math.radians(deg)
    x, y, z = v
    return [x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a)]


def _quat_y(deg):
    h = math.radians(deg) / 2
    return [0.0, round(math.sin(h), 6), 0.0, round(math.cos(h), 6)]


def _r(v):
    return [round(float(x), 4) for x in v]


def _box(extent):
    return {"kind": "primitive", "shape": "box", "extent": _r(extent)}


def _outward_axis(axis, pivot, free_point, outward):
    """铰链轴取哪个符号：让自由边在小的正转角下往外走（outward 是该部件张开的方向）。"""
    rel = [free_point[k] - pivot[k] for k in range(3)]
    cross = [axis[1] * rel[2] - axis[2] * rel[1], axis[2] * rel[0] - axis[0] * rel[2], axis[0] * rel[1] - axis[1] * rel[0]]
    return axis if sum(cross[k] * outward[k] for k in range(3)) > 0 else [-x for x in axis]


def _door(pid, cls, centre, extent, hinge_side, rng):
    """竖铰链的门：hinge_side 是 -1（左边）或 +1（右边），本地坐标里前面朝 +z。"""
    pivot = [centre[0] + hinge_side * extent[0] / 2, centre[1], centre[2] - extent[2] / 2]
    free = [centre[0] - hinge_side * extent[0] / 2, centre[1], centre[2]]
    axis = _outward_axis([0.0, 1.0, 0.0], pivot, free, [0.0, 0.0, 1.0])
    return {"id": pid, "class": cls, "centre": centre, "extent": extent, "joint": "revolute",
            "axis": axis, "pivot": pivot, "open": round(rng.uniform(55, 120), 1)}


def _flap(pid, cls, centre, extent, edge, rng, lo, hi):
    """横铰链：edge 是 'back_top'（箱盖，往上掀）或 'bottom_front'（烤箱门，往下翻）。"""
    if edge == "back_top":
        pivot = [centre[0], centre[1] - extent[1] / 2, centre[2] - extent[2] / 2]
        free = [centre[0], centre[1], centre[2] + extent[2] / 2]
        outward = [0.0, 1.0, 0.0]
    else:
        pivot = [centre[0], centre[1] - extent[1] / 2, centre[2] - extent[2] / 2]
        free = [centre[0], centre[1] + extent[1] / 2, centre[2]]
        outward = [0.0, 0.0, 1.0]
    axis = _outward_axis([1.0, 0.0, 0.0], pivot, free, outward)
    return {"id": pid, "class": cls, "centre": centre, "extent": extent, "joint": "revolute",
            "axis": axis, "pivot": pivot, "open": round(rng.uniform(lo, hi), 1)}


def _drawer(pid, centre, extent, depth, rng):
    return {"id": pid, "class": "drawer", "centre": centre, "extent": extent, "joint": "prismatic",
            "axis": [0.0, 0.0, 1.0], "open": round(rng.uniform(0.4, 0.85) * depth, 3)}


def _layout(family: str, rng: random.Random):
    """本地坐标（前面朝 +z、柜体底面在 y=0）下的柜体尺寸和部件列表。"""
    m, g, th = _MARGIN, _GAP, _THICK
    if family == "cabinet":
        W, H, D = rng.uniform(0.6, 1.4), rng.uniform(0.7, 2.0), rng.uniform(0.4, 0.7)
        if rng.random() < 0.4:
            parts = [_door("part_1", "door", [0.0, H / 2, D / 2 + th / 2], [W - 2 * m, H - 2 * m, th], rng.choice([-1, 1]), rng)]
        else:
            w = (W - 2 * m - g) / 2
            parts = [_door("part_1", "door", [-(g + w) / 2, H / 2, D / 2 + th / 2], [w, H - 2 * m, th], -1, rng),
                     _door("part_2", "door", [(g + w) / 2, H / 2, D / 2 + th / 2], [w, H - 2 * m, th], 1, rng)]
    elif family == "drawers":
        W, H, D = rng.uniform(0.5, 1.2), rng.uniform(0.6, 1.4), rng.uniform(0.4, 0.6)
        n = rng.choice([2, 3, 4])
        h = (H - 2 * m) / n
        parts = [_drawer(f"part_{i + 1}", [0.0, m + (n - 1 - i + 0.5) * h, 0.03], [W - 2 * m, h - g, D - 0.04], D, rng)
                 for i in range(n)]
    elif family == "chest":
        W, H, D = rng.uniform(0.6, 1.4), rng.uniform(0.35, 0.8), rng.uniform(0.4, 0.8)
        parts = [_flap("part_1", "lid", [0.0, H + 0.025, 0.0], [W, 0.05, D], "back_top", rng, 45, 110)]
    elif family == "oven":
        W, H, D = rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.7)
        hd = (H - 2 * m) * rng.uniform(0.65, 0.8)
        parts = [_flap("part_1", "door", [0.0, m + hd / 2, D / 2 + th / 2], [W - 2 * m, hd, th], "bottom_front", rng, 60, 88)]
    elif family == "combo":
        W, H, D = rng.uniform(0.8, 1.4), rng.uniform(1.0, 1.8), rng.uniform(0.45, 0.65)
        hdr = (H - 2 * m) * rng.uniform(0.18, 0.28)
        w = (W - 2 * m - g) / 2
        hdoor = H - 2 * m - hdr - g
        yd = m + hdr + g + hdoor / 2
        parts = [_door("part_1", "door", [-(g + w) / 2, yd, D / 2 + th / 2], [w, hdoor, th], -1, rng),
                 _door("part_2", "door", [(g + w) / 2, yd, D / 2 + th / 2], [w, hdoor, th], 1, rng),
                 _drawer("part_3", [0.0, m + hdr / 2, 0.03], [W - 2 * m, hdr - g, D - 0.04], D, rng)]
    else:
        raise ValueError(f"unknown family {family}")
    return (W, H, D), parts


def _sentence(family: str, parts: list[dict], moving: set[str]) -> str:
    """欠定的一句话：说是什么家具、有哪些部件、哪类部件动了几个，不说哪个、往哪、多大。"""
    name = _NAMES[family]
    counts: dict[str, int] = {}
    for p in parts:
        counts[p["class"]] = counts.get(p["class"], 0) + 1
    inv = " and ".join(f"{_NUM[c]} {k}{'s' if c > 1 else ''}" for k, c in counts.items())
    article = "An" if name[0] in "aeiou" else "A"
    head = f"{article} {name} with {inv}." if family != "chest" else "A chest with a lid."
    clauses = []
    for k, c in counts.items():
        n = sum(1 for p in parts if p["class"] == k and p["id"] in moving)
        verb = {"door": "swing" if family != "oven" else "fold", "drawer": "slide", "lid": "swing"}[k]
        tail = {"door": "open" if family != "oven" else "down", "drawer": "out", "lid": "open"}[k]
        if n == 0:
            continue
        if c == 1:
            clauses.append(f"The {k} {verb}s {tail}.")
        elif n == c:
            clauses.append(f"{'Both' if c == 2 else 'All ' + _NUM[c]} {k}s {verb} {tail}.")
        else:
            clauses.append(f"{_NUM[n].capitalize()} of the {k}s {verb}{'s' if n == 1 else ''} {tail}; the other{'s stay' if c - n > 1 else ' stays'} shut.")
    return head + " " + " ".join(clauses)


def generate_articulated(clip: str, family: str, seed: int, style_seed: int) -> tuple[dict, dict]:
    """返回 (program, facts)。facts 里有句子和每个部件的真值关节（世界坐标），给评测和描述用。"""
    rng = random.Random(seed)
    (W, H, D), parts = _layout(family, rng)
    k = rng.choice([1, 1, 2]) if len(parts) > 1 else 1
    moving = set(p["id"] for p in rng.sample(parts, min(k, len(parts))))
    yaw = rng.choice([0, 90, 180, 270])
    t_start = round(rng.uniform(0.3, 1.0), 2)
    dur = round(rng.uniform(1.5, DURATION - t_start - 0.3), 2)

    colours = rng.sample(_PART_COLORS, len(parts))
    objects, joints = [], {}
    for p, colour in zip(parts, colours):
        node = {"id": p["id"], "class": p["class"], "geom": _box(p["extent"]),
                "pose": {"pos": _r(_rot_y(p["centre"], yaw)), "quat": _quat_y(yaw)}, "material": colour}
        axis = _r(_rot_y(p["axis"], yaw))
        if p["id"] not in moving:
            node["motion"] = {"type": "static"}
        elif p["joint"] == "revolute":
            node["motion"] = {"type": "revolute", "axis": axis, "pivot": _r(_rot_y(p["pivot"], yaw)),
                              "range_deg": [0, p["open"]], "from_deg": 0,
                              "schedule": [{"t": t_start, "to_deg": p["open"], "duration": dur}]}
        else:
            node["motion"] = {"type": "prismatic", "axis": axis, "range": [0, p["open"]], "from": 0,
                              "schedule": [{"t": t_start, "to": p["open"], "duration": dur}]}
        joints[p["id"]] = {"class": p["class"], "moves": p["id"] in moving, "joint": p["joint"], "axis": axis,
                           "pivot": _r(_rot_y(p["pivot"], yaw)) if p["joint"] == "revolute" else None,
                           "open": p["open"], "extent": _r(p["extent"])}
        objects.append(node)

    # 相机：正面左右 55° 以内、仰角 12°~38°，距离让张开后的部件也在画面里。
    fov = 50.0
    reach = max(W, H, D + max([p["extent"][0] for p in parts if p["joint"] == "revolute"] + [0.0])
                + max([p["open"] for p in parts if p["joint"] == "prismatic"] + [0.0]))
    radius = 0.5 * math.sqrt(W * W + H * H + reach * reach)
    dist = radius / math.sin(math.radians(fov / 2)) * rng.uniform(1.0, 1.25)
    az, el = rng.uniform(-55, 55), rng.uniform(12, 38)
    look = [0.0, H / 2, D / 2]
    d = _rot_y([0.0, 0.0, 1.0], az)
    eye = [look[0] + dist * math.cos(math.radians(el)) * d[0], look[1] + dist * math.sin(math.radians(el)),
           look[2] + dist * math.cos(math.radians(el)) * d[2]]
    eye, look = _r(_rot_y(eye, yaw)), _r(_rot_y(look, yaw))

    srng = random.Random(style_seed)
    program = {
        "meta": {"clip": clip, "fps": 30, "duration": DURATION, "units": "m", "up": "y"},
        "style": sample_style(srng),
        "camera": {"intrinsics": {"fov_deg": fov, "aspect": 1.7778, "far": 100.0},
                   "keyframes": [{"t": 0.0, "pos": eye, "look_at": look}, {"t": DURATION, "pos": eye, "look_at": look}]},
        "static": [{"id": "ground", "class": "ground", "geom": _box([12.0, 0.4, 12.0]), "pose": {"pos": [0.0, -0.2, 0.0]},
                    "material": srng.choice(["concrete", "wood", "stone", "grass"])},
                   {"id": "body", "class": _NAMES[family].split()[-1], "geom": _box([W, H, D]),
                    "pose": {"pos": _r(_rot_y([0.0, H / 2, 0.0], yaw)), "quat": _quat_y(yaw)},
                    "material": srng.choice(_BODY_COLORS)}],
        "objects": objects,
    }
    facts = {"family": family, "yaw": yaw, "sentence": _sentence(family, parts, moving), "joints": joints}
    return program, facts
