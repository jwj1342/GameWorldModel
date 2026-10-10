"""按属性推断每个物体在游戏里扮演什么角色。

替代原来那份封闭词表。原来的做法是拿 class 名去匹配 coin/gem/key 和
lava/spike/water 这几个词，只在我们自己用游戏术语手写的合成场景上命中。
真实视频里的物体叫 toy_train、conveyor belt、cardboard box，一个都匹配不上，
所以两段真实视频的收集物和危险物列表全是空的，玩起来就是走到角落。

这里改成看物体本身的属性：多大、多高、顶面能不能站人、怎么运动、撑着什么。
这些量和物体叫什么名字无关，所以开放词表也能用。

阈值一律以玩家为基准，不以场景尺度为基准。绑定会把最大静态足迹缩放到 24 米的
游戏尺度，而玩家胶囊是固定大小不跟着缩，所以「能不能站上去」「能不能捡起来」
「够不够得着」都该拿玩家量。拿场景量会漂：同一扇门在大场景里占 5%、
在小场景里占 30%，但它能不能被捡起来跟场景多大没关系。

拿玩家量就得说清楚从哪儿量起。够不够得着要从玩家站的那一层算，不能拿世界
坐标里的绝对高度：同一个场景整体抬高十米，玩家和物体的相对关系一点没变，
但绝对高度全部超标，收集物会集体变成摆设。能不能站上去也要把姿态算进去，
局部 extent 看不出盒子是躺着还是立着。

一个诚实的边界：危险物（熔岩、水、火）光靠属性推不出来，它本质上是语义判断。
这里保留一个可配置的关键词信号，并把置信度标低，等以后接语义模型再改。
"""
from __future__ import annotations
import math

from ..compiler.geometry import rotation_matrix

ROLES = ("static_structure", "platform", "dynamic_platform", "collectible", "moving_obstacle", "hazard", "decoration")

# 玩家胶囊在 kernel/physics.js 的 createPlayer 里是半径 0.35、半高 0.55，
# 所以直径约 0.7 米、总高约 1.8 米。下面的阈值都从这两个数推出来。
PLAYER_DIAMETER_M = 0.7
PLAYER_HEIGHT_M = 1.8
# kernel/physics.js 的 KCC 设的是 setMaxSlopeClimbAngle(55 度)，再陡角色就爬不上去了。
PLAYER_MAX_SLOPE_DEG = 55.0

DEFAULTS = {
    "platform_min_top_m2": 0.45,                   # 顶面至少这么大才站得住人。玩家足迹约 0.49 平米，
                                                   # 角色控制器有自动上台阶和贴地，略小于足迹也站得住。
    "collectible_max_m": 2.0 * PLAYER_DIAMETER_M,  # 比玩家大一圈以内才可能被捡起来
    "collectible_min_chunkiness": 0.3,             # 第二长边至少占最长边这个比例。收集物应该是块状的，
                                                   # 条状的多半是结构件。硬币 0.5x0.5x0.06 算下来是 1.0，
                                                   # 门框 1.79x0.22x0.08 是 0.12，分得很开。
                                                   # 用长宽比做不到这点，它把扁片和长条混为一谈。
    "reach_max_m": 1.5 * PLAYER_HEIGHT_M,          # 高过这个就够不着了。从玩家站的那一层算起，不是绝对高度
    "max_slope_deg": PLAYER_MAX_SLOPE_DEG,         # 顶面倾斜超过这个角度就站不住，跟角色控制器的设置对齐
    "hazard_keywords": ["lava", "spike", "water", "fire", "acid", "熔岩", "尖刺", "岩浆"],
}


def _extent(node: dict) -> list[float] | None:
    """Local dimensions of the declared geometry, not a guessed small object.

    Cylinder height follows its axis; pose rotation and support-aware standing
    remain separate concerns. Primitive defaults match the renderer where used.
    """
    g = node.get("geom") or {}
    try:
        if g.get("extent") and g.get("shape") not in {"sphere", "cylinder", "cone"}:
            dims = [float(v) for v in g["extent"]]
        elif g.get("shape") == "sphere":
            dims = [2 * float(g["radius"])] * 3
        elif g.get("shape") == "cylinder":
            radius = g.get("radius", 0.5)
            r = max(float(g.get("radius_top", radius)), float(g.get("radius_bottom", radius)))
            dims = [2 * r] * 3
            dims[{"x": 0, "y": 1, "z": 2}[g.get("axis", "y")]] = float(g["height"])
        elif g.get("shape") == "cone":
            # The current renderer creates cones on the y axis.
            r = float(g["radius"])
            dims = [2 * r, float(g["height"]), 2 * r]
        elif g.get("size"):
            dims = [float(g["size"][0]), 0.1, float(g["size"][1])]
        elif g.get("kind") == "heightfield":
            dims = [float(v) for v in g["scale"]]
        else:
            return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return dims if len(dims) == 3 and all(math.isfinite(v) and v > 0 for v in dims) else None


def _poses(node: dict) -> list[dict]:
    """一个物体可以写成一个 pose，也可以写成一串 instances，两种都要看全。

    只看第一个会让判定跟着数组顺序走：同样五枚硬币，第一枚写在够得着的高度
    还是写在天花板上，整个物体的角色就不一样了。
    """
    out = ([node["pose"]] if node.get("pose") else []) + list(node.get("instances") or [])
    return out or [{"pos": [0.0, 0.0, 0.0]}]


def _y(pose: dict) -> float:
    pos = pose.get("pos") or [0.0, 0.0, 0.0]
    try:
        return float(pos[1])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _oriented(extent: list[float], quat) -> tuple[list[tuple[float, float]], float]:
    """按姿态把局部尺寸摆进世界。返回盒子三组对面各自的（水平投影面积，法线
    偏离竖直的余弦），以及物体在竖直方向上的半高。

    盒子是躺着还是立着，光看局部 extent 分不出来：一块 2 x 0.1 x 2 的板子绕
    x 轴转九十度就立起来了，extent 一个字没变，但顶上已经没法站人。

    三组面都要算，不能只挑法线最接近竖直的那一组。同一块板子倾斜五十度的时候，
    窄边比大面更接近水平，但人踩的是那块四平米的大面——它才五十度，角色爬得上去。
    """
    rotation = rotation_matrix([0.0, 0.0, 0.0, 1.0] if quat is None else quat)
    # 姿态读不出来就当没转。局部 y 朝上，跟原来的行为一致。
    vertical = [0.0, 1.0, 0.0] if rotation is None else [abs(rotation[1][k]) for k in range(3)]
    faces = []
    for k in range(3):
        i, j = (c for c in range(3) if c != k)
        faces.append((extent[i] * extent[j] * vertical[k], vertical[k]))
    return faces, sum(vertical[k] * extent[k] for k in range(3)) / 2.0


def _top_y(node: dict) -> float | None:
    """节点顶面在世界里的高度；读不出尺寸就返回 None。"""
    e = _extent(node)
    if e is None: return None
    pose = _poses(node)[0]
    return _y(pose) + _oriented(e, pose.get("quat"))[1]


def ground_level(program: dict) -> float:
    """玩家脚下那一层在哪。够不够得着从这一层量起。

    取静态几何里水平投影最大的那块的顶面，也就是地面。场景在世界坐标里被
    整体抬高多少，跟玩家够不够得着一枚硬币没有关系。
    """
    best_area, level = -1.0, 0.0
    for n in program.get("static", []):
        e = _extent(n)
        if e is None: continue
        pose = _poses(n)[0]
        area = max(a for a, _ in _oriented(e, pose.get("quat"))[0])
        if area > best_area:
            best_area, level = area, _top_y(n)
    return level


def classify(program: dict, settings: dict | None = None) -> dict[str, dict]:
    """返回 {id: {"role": ..., "reason": ...}}，每个判断都带理由，便于审查。"""
    s = {**DEFAULTS, **(settings or {})}
    ground = ground_level(program)
    min_cos_tilt = math.cos(math.radians(float(s["max_slope_deg"])))
    # support 指向谁，就从谁的顶面量起：三米高的台子上放一枚硬币，玩家爬上台子就够得着。
    tops = {n["id"]: t for n in list(program.get("static", [])) + list(program.get("objects", []))
            if (t := _top_y(n)) is not None}
    out: dict[str, dict] = {}

    # 静态几何只看危险物。哪些静态面能走由内核的 walkable: auto 决定，
    # 它直接看顶面朝向和高度，比在这里按包围盒猜准，不要重复判断。
    for n in program.get("static", []):
        name = ((n.get("class") or "") + " " + (n.get("material") or "")).lower()
        if any(k in name for k in s["hazard_keywords"]):
            out[n["id"]] = {"role": "hazard", "reason": f"类别或材质里出现了危险物关键词（{name.strip()}）；这是语义判断，置信度低"}
        else:
            out[n["id"]] = {"role": "static_structure", "reason": "静态几何，能不能走由内核的 walkable 判定"}

    for o in program.get("objects", []):
        e = _extent(o)
        if e is None:
            out[o["id"]] = {"role": "decoration", "reason": "几何尺寸缺失或无效，不自动推断收集或平台交互"}
            continue
        longest = max(e)
        moving = (o.get("motion") or {}).get("type", "static") != "static"
        name = ((o.get("class") or "") + " " + (o.get("material") or "")).lower()
        # 逐个实例算，再取最宽松的那个：实例之间姿态和高度可以不一样，
        # 只认第一个的话，数组换个顺序同一个物体就换了角色。
        poses = _poses(o)
        faces = [face for pose in poses for face in _oriented(e, pose.get("quat"))[0]]
        # 顶面要够大，而且得接近水平——立起来的板子局部 extent 一个字没变，但站不住人。
        # 高不高不影响能不能站在顶上，够不够得着由 reachable 管。
        standable = [area for area, cos_tilt in faces
                     if area >= s["platform_min_top_m2"] and cos_tilt >= min_cos_tilt]
        top_area = max(standable) if standable else max(area for area, _ in faces)
        srt = sorted(e, reverse=True)
        chunkiness = srt[1] / max(srt[0], 1e-6)
        small = longest <= s["collectible_max_m"] and chunkiness >= s["collectible_min_chunkiness"]
        base = tops.get(o.get("support"), ground)
        height = min(_y(pose) - base for pose in poses)
        reachable = height <= s["reach_max_m"]

        # 顺序要紧：站得住人的优先当平台。楼梯、台阶这些尺寸不大但是地形的一部分，
        # 如果先判收集物就会把一级级台阶变成金币，实测踩过这个坑。
        if any(k in name for k in s["hazard_keywords"]):
            role, why = "hazard", f"类别或材质里出现了危险物关键词（{name.strip()}）；这是语义判断，置信度低"
        elif standable and moving:
            role, why = "dynamic_platform", f"顶面水平投影 {top_area:.1f} 平米站得住人，而且在动，可以当会移动的平台"
        elif standable:
            role, why = "platform", f"顶面水平投影 {top_area:.1f} 平米站得住人，静止不动"
        elif small and reachable:
            role, why = "collectible", f"站不上去、最大边长 {longest:.2f} 米（玩家约 {PLAYER_DIAMETER_M} 米宽）、是块状不是条状（第二长边占 {chunkiness:.0%}）、离站立面 {height:.2f} 米够得着"
        elif moving and reachable:
            role, why = "moving_obstacle", f"在动、站不上去（顶面水平投影只有 {top_area:.1f} 平米）、离站立面 {height:.2f} 米够得着，会挡路"
        else:
            role, why = "decoration", f"不满足平台、收集物或移动障碍规则（顶面水平投影 {top_area:.1f} 平米，最大边长 {longest:.2f} 米，离站立面 {height:.1f} 米）"
        out[o["id"]] = {"role": role, "reason": why}

    return out


def slots_from_roles(roles: dict[str, dict]) -> dict[str, list[str]]:
    """把角色表整理成绑定需要的槽位列表。"""
    pick = lambda r: [i for i, v in roles.items() if v["role"] == r]
    return {"collectibles": pick("collectible"), "hazards": pick("hazard"),
            "dynamic_platforms": pick("dynamic_platform"), "moving_obstacles": pick("moving_obstacle")}
