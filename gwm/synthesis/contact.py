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
