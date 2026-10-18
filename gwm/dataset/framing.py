"""取景约定：把生成出来的场景整体缩放到目标输入的尺度。

为什么需要这个。整条链路把相对尺度的重建标定到米，靠的是一个假设：相机离地
`assumed_camera_height_m`（1.2 米，手持手机站着拍）。而生成器造的是平台跳跃关卡——
走廊 4 到 7 米宽、18 到 30 米长，墙高 2.4 到 3.6 米，相机高度是场地尺度的函数，实测
中位数 7.385 米，最高 26 米。两者差六到十二倍，后果见 #68：真值评测里物体配对的容差
是 3 米，场景被压小之后一百零二个真值物体只有三十七个配得上。

修法是整体**相似**缩放，不是重写场景族。几何相似正是需要的：链路对场景的唯一尺度
假设就是相机高度，把整个场景（含相机轨迹）等比缩小，相机高度就跟着落到手持的范围，
而场景内部的相对结构一点不变。

缩放因子取多少由调用方定，默认不缩放，所以现有数据和行为不受影响。
"""
from __future__ import annotations

import json

# 这些字段带长度量纲，缩放时必须一起走，漏掉任何一个都会让场景内部自相矛盾
_VECTOR_KEYS = ("extent", "size", "pos", "offset", "linear_velocity", "amplitude", "pivot")
_SCALAR_KEYS = ("radius", "radius_top", "radius_bottom", "height", "speed", "distance", "travel")


def _scale_vec(container: dict, key: str, k: float) -> None:
    value = container.get(key)
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        container[key] = [float(v) * k for v in value]


def _scale_num(container: dict, key: str, k: float) -> None:
    value = container.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        container[key] = float(value) * k


def scale_program(program: dict, k: float) -> dict:
    """把整个场景乘上 k，返回新的程序，不改原来的。

    位置、尺寸、相机轨迹、运动参数都要跟着走。只缩放带长度量纲的量——角度、角速度、
    时间、质量、弹性系数都不动。
    """
    if k == 1.0:
        return json.loads(json.dumps(program))
    out = json.loads(json.dumps(program))
    for node in (out.get("objects") or []) + (out.get("static") or []):
        for holder in (node.get("pose") or {}, node.get("geom") or {},
                       node.get("motion") or {}, node.get("interaction") or {}):
            for key in _VECTOR_KEYS:
                _scale_vec(holder, key, k)
            for key in _SCALAR_KEYS:
                _scale_num(holder, key, k)
        for part in (node.get("geom") or {}).get("parts") or []:
            for key in _VECTOR_KEYS:
                _scale_vec(part, key, k)
            for key in _SCALAR_KEYS:
                _scale_num(part, key, k)
        for inst in node.get("instances") or []:
            _scale_vec(inst, "pos", k)
        # 滑轨行程和铰链限位：前者是长度要缩，后者是角度不能缩
        interaction = node.get("interaction") or {}
        mode = interaction.get("grab")
        if isinstance(interaction.get("limit"), list) and mode == "slide":
            interaction["limit"] = [float(v) * k for v in interaction["limit"]]
    camera = out.get("camera") or {}
    for frame in camera.get("keyframes") or []:
        _scale_vec(frame, "pos", k)
    intrinsics = camera.get("intrinsics") or {}
    _scale_num(intrinsics, "far", k)
    binding = (out.get("binding") or {}).get("slots") or {}
    _scale_vec(binding, "player_spawn", k)
    goal = binding.get("goal_volume")
    if isinstance(goal, dict):
        _scale_vec(goal, "pos", k)
        _scale_vec(goal, "extent", k)
    out.setdefault("meta", {})["stage_scale"] = float(k)
    return out
