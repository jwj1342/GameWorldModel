"""把物体放到支撑面上。

直译出来的位姿直接取自感知给的包围盒中心，而包围盒是拟合到**可见表面**的，加上地面
法向本身就有估计误差，于是经常有物体陷进地里。陷进去不只是难看：拖动的时候物体会从
地下被拽出来，物理也解不稳，所以可交互性先要求物体本来就站在该站的地方。

这一步和 `gwm/compiler/repair.py` 是两件事，不要混。那边是**保守修复**，只在证据支持
时才动位姿，动不了就如实报错；这里是**装配**，明确承认「让物体落到支撑面上」是我们加
的先验，不是从视频里测出来的，所以每次移动都记下来，谁用谁自己判断要不要信。
"""
from __future__ import annotations

from ..compiler.geometry import _aabb, _GROUND_CLASSES

# 小于这个量就不动了：和校验器的接触容差一致，免得改完仍然踩线
TOLERANCE_M = 0.01


def ground_top(program: dict) -> float | None:
    """地面的顶面高度。多块地面取最高的那块。"""
    tops = []
    for node in program.get("static", []):
        if str(node.get("class", "")).lower() not in _GROUND_CLASSES:
            continue
        bounds = _aabb(node, node.get("pose") or {})
        if bounds is not None:
            tops.append(bounds[1][1])
    return max(tops) if tops else None


def settle_on_ground(program: dict, max_lift_m: float = 2.0) -> list[dict]:
    """把陷进地面的物体抬到地面上，就地修改程序。返回每次移动的记录。

    抬升量超过 `max_lift_m` 的不动：那种情况多半不是「稍微陷进去一点」，而是这个物体的
    尺寸或者地面本身估错了，硬抬会把一个大错误变成一个看不出来的大错误。
    """
    top = ground_top(program)
    if top is None:
        return []
    moves = []
    for node in program.get("objects", []):
        pose = node.get("pose") or {}
        bounds = _aabb(node, pose)
        if bounds is None:
            continue
        lift = top - bounds[0][1]
        if lift <= TOLERANCE_M:
            continue
        if lift > max_lift_m:
            moves.append({"id": node["id"], "lift_m": round(lift, 4), "applied": False,
                          "reason": "lift_exceeds_limit"})
            continue
        pose.setdefault("pos", [0.0, 0.0, 0.0])
        pose["pos"][1] = float(pose["pos"][1]) + lift
        moves.append({"id": node["id"], "lift_m": round(lift, 4), "applied": True})
    return moves


def drop_sunk(program: dict, below_height_mult: float = 2.0) -> list[dict]:
    """丢掉**接触装配之后仍然**陷在地面以下很深的物体。

    一个底面在地平面下方好几倍自身高度的「物体」，要么是深度把一片背景推到了地下，
    要么是掩码糊到了地面以外，两种都不是可以拿起来的东西。

    **顺序是这条规则的全部要害，必须跑在 `settle_on_ground` 之后。** 我第一版把它
    放在 `direct.py` 里程序刚生成的时候，离线估算说「砍掉 6 个、全是假阳性、一个
    命中不丢、精确率 0.214 -> 0.227」，实际接进链路跑出来是**砍掉 57 个、召回
    0.500 -> 0.268**。

    原因是顺序：接触装配本来就负责把陷在地里的物体抬上来，而我在它之前把那些物体
    丢掉了——**丢的正是它要修的那一批**。而我离线估算时用的是**最终程序**（装配
    已经跑过、物体都抬上来了），所以只看到还剩 6 个。

    教训很具体：**拿「处理后的数据」去估算「处理前的闸」会砍掉多少，估的是另一个
    分布。** 要估一个闸的代价，得在它实际会看到的那个管线位置上取数据。
    """
    import numpy as np

    dropped = []
    keep = []
    for node in program.get("objects") or []:
        ext = (node.get("geom") or {}).get("extent")
        pos = (node.get("pose") or {}).get("pos")
        if not (isinstance(ext, list) and len(ext) == 3 and isinstance(pos, list) and len(pos) == 3):
            keep.append(node); continue
        height = float(ext[1])
        bottom = float(pos[1]) - height / 2
        if height > 1e-6 and bottom < -below_height_mult * height:
            dropped.append({"id": node["id"], "class": node.get("class"),
                            "bottom_over_height": round(bottom / max(height, 1e-6), 2)})
        else:
            keep.append(node)
    if dropped:
        program["objects"] = keep
        # 留痕放 meta：程序 schema 顶层是 additionalProperties: false，加顶层键会让
        # 整个程序编译不过（这个坑踩过一次）。meta 的 additionalProperties 是 true。
        program.setdefault("meta", {})["dropped_sunk"] = dropped
    return dropped
