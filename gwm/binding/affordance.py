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

一个诚实的边界：危险物（熔岩、水、火）光靠属性推不出来，它本质上是语义判断。
这里保留一个可配置的关键词信号，并把置信度标低，等以后接语义模型再改。
"""
from __future__ import annotations

ROLES = ("static_structure", "platform", "dynamic_platform", "collectible", "moving_obstacle", "hazard", "decoration")

# 玩家胶囊在 kernel/physics.js 的 createPlayer 里是半径 0.35、半高 0.55，
# 所以直径约 0.7 米、总高约 1.8 米。下面的阈值都从这两个数推出来。
PLAYER_DIAMETER_M = 0.7
PLAYER_HEIGHT_M = 1.8

DEFAULTS = {
    "platform_min_top_m2": 0.45,                   # 顶面至少这么大才站得住人。玩家足迹约 0.49 平米，
                                                   # 角色控制器有自动上台阶和贴地，略小于足迹也站得住。
    "collectible_max_m": 2.0 * PLAYER_DIAMETER_M,  # 比玩家大一圈以内才可能被捡起来
    "collectible_min_chunkiness": 0.3,             # 第二长边至少占最长边这个比例。收集物应该是块状的，
                                                   # 条状的多半是结构件。硬币 0.5x0.5x0.06 算下来是 1.0，
                                                   # 门框 1.79x0.22x0.08 是 0.12，分得很开。
                                                   # 用长宽比做不到这点，它把扁片和长条混为一谈。
    "reach_max_m": 1.5 * PLAYER_HEIGHT_M,          # 高过这个就够不着了
    "hazard_keywords": ["lava", "spike", "water", "fire", "acid", "熔岩", "尖刺", "岩浆"],
}


def _extent(node: dict) -> list[float]:
    g = node.get("geom") or {}
    if g.get("extent"): return [float(v) for v in g["extent"]]
    if g.get("radius"): r = float(g["radius"]); return [2 * r, 2 * r, 2 * r]
    if g.get("size"): s = g["size"]; return [float(s[0]), 0.1, float(s[1])]
    return [0.1, 0.1, 0.1]


def _pos(node: dict) -> list[float]:
    if node.get("pose"): return [float(v) for v in node["pose"]["pos"]]
    inst = node.get("instances") or []
    return [float(v) for v in inst[0]["pos"]] if inst else [0.0, 0.0, 0.0]


def scene_scale(program: dict) -> float:
    """场景尺度取静态几何里最大的水平尺寸，所有相对阈值都以它为准。"""
    best = 1.0
    for n in program.get("static", []):
        e = _extent(n); best = max(best, e[0], e[2])
    return best


def classify(program: dict, settings: dict | None = None) -> dict[str, dict]:
    """返回 {id: {"role": ..., "reason": ...}}，每个判断都带理由，便于审查。"""
    s = {**DEFAULTS, **(settings or {})}
    scale = scene_scale(program)
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
        e, p = _extent(o), _pos(o)
        top_area, longest = e[0] * e[2], max(e)
        moving = (o.get("motion") or {}).get("type", "static") != "static"
        name = ((o.get("class") or "") + " " + (o.get("material") or "")).lower()
        # 只看顶面够不够大。高不高不影响能不能站在顶上，够不够得着由 reachable 管
        standable = top_area >= s["platform_min_top_m2"]
        srt = sorted(e, reverse=True)
        chunkiness = srt[1] / max(srt[0], 1e-6)
        small = longest <= s["collectible_max_m"] and chunkiness >= s["collectible_min_chunkiness"]
        reachable = p[1] <= s["reach_max_m"]

        # 顺序要紧：站得住人的优先当平台。楼梯、台阶这些尺寸不大但是地形的一部分，
        # 如果先判收集物就会把一级级台阶变成金币，实测踩过这个坑。
        if any(k in name for k in s["hazard_keywords"]):
            role, why = "hazard", f"类别或材质里出现了危险物关键词（{name.strip()}）；这是语义判断，置信度低"
        elif standable and moving:
            role, why = "dynamic_platform", f"顶面 {top_area:.1f} 平米站得住人，而且在动，可以当会移动的平台"
        elif standable:
            role, why = "platform", f"顶面 {top_area:.1f} 平米站得住人，静止不动"
        elif small and reachable:
            role, why = "collectible", f"站不上去、最大边长 {longest:.2f} 米（玩家约 {PLAYER_DIAMETER_M} 米宽）、是块状不是条状（第二长边占 {chunkiness:.0%}）、高度够得着"
        elif moving and reachable:
            role, why = "moving_obstacle", f"在动、站不上去（顶面只有 {top_area:.1f} 平米）、又在够得着的高度，会挡路"
        else:
            role, why = "decoration", f"站不上去也够不着（顶面 {top_area:.1f} 平米，高度 {p[1]:.1f} 米）"
        out[o["id"]] = {"role": role, "reason": why}

    return out


def slots_from_roles(roles: dict[str, dict]) -> dict[str, list[str]]:
    """把角色表整理成绑定需要的槽位列表。"""
    pick = lambda r: [i for i, v in roles.items() if v["role"] == r]
    return {"collectibles": pick("collectible"), "hazards": pick("hazard"),
            "dynamic_platforms": pick("dynamic_platform"), "moving_obstacles": pick("moving_obstacle")}
