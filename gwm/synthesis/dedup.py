"""把同一个物体的重复检出合掉。

#56 里量过：手持取景那批片段上，一百二十四个没配上真值的预测物体里，**四十一个
（38%）离最近的真阳性不到一个物体尺度**，其中十六个连类别都相同——它们是同一个东西
被检出了两遍。剩下 62% 离得更远，是真的无中生有，去重碰不到它们。

所以这一步的上限是算得出来的：假设去重做到完美、一个真物体都不误伤，精确率从
26/150 = 0.173 升到 26/109 = 0.239。值得做，但不解决过检本身。

判据用的是和当初刻画它时**完全相同**的统计量：两个物体中心的距离，除以两者包围盒
对角线长度的平均值。疑似重复那一群的中位数是 0.74，无中生有那一群是 1.74。阈值要
保守——0.74 离 1.0 不远，而堆叠的箱子、紧挨的栏杆本来就该是两个物体。
"""
from __future__ import annotations

import math


def _centre(node: dict) -> list[float] | None:
    pos = (node.get("pose") or {}).get("pos")
    if isinstance(pos, list) and len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos):
        return [float(v) for v in pos]
    return None


def _diagonal(node: dict) -> float:
    geom = node.get("geom") or {}
    extent = geom.get("extent") or geom.get("size")
    if isinstance(extent, list) and extent:
        return math.sqrt(sum(float(v) ** 2 for v in extent))
    radius = geom.get("radius")
    height = geom.get("height")
    if isinstance(radius, (int, float)):
        return math.sqrt((2 * float(radius)) ** 2 * 2 + float(height or 2 * radius) ** 2)
    return 0.0


def _support(node: dict, evidence_by_id: dict) -> float:
    """留哪个、丢哪个。证据多的留下——被看到的帧数优先，其次置信度。"""
    obj = evidence_by_id.get(node.get("id")) or {}
    frames = len(obj.get("obb") or [])
    conf = obj.get("confidence")
    return (frames, float(conf) if isinstance(conf, (int, float)) else 0.0)


def merge_duplicates(program: dict, max_relative_distance: float = 0.5,
                     same_class_only: bool = True,
                     evidence: dict | None = None) -> list[dict]:
    """就地删掉重复的物体节点，返回每次合并的记录。

    `max_relative_distance` 是中心距离除以两者对角线的平均值。0 关掉这一步。
    `same_class_only` 默认开着：类别不同的两个框挨得近，更可能是两个真的物体
    （杯子放在托盘上），不是重复。
    """
    if max_relative_distance <= 0:
        return []
    objects = program.get("objects") or []
    if len(objects) < 2:
        return []
    from ..taxonomy import slug
    by_id = {}
    for obj in (evidence or {}).get("objects", []):
        by_id[obj["id"]] = obj
        by_id[slug(obj["id"])] = obj

    keep: list[dict] = []
    merged: list[dict] = []
    # 证据多的先进，所以留下的总是支持更强的那个
    for node in sorted(objects, key=lambda n: _support(n, by_id), reverse=True):
        centre, diag = _centre(node), _diagonal(node)
        duplicate_of = None
        if centre is not None and diag > 1e-6:
            for other in keep:
                if same_class_only and str(other.get("class", "")).lower() != str(node.get("class", "")).lower():
                    continue
                other_centre, other_diag = _centre(other), _diagonal(other)
                if other_centre is None or other_diag <= 1e-6:
                    continue
                distance = math.dist(centre, other_centre)
                scale = (diag + other_diag) / 2
                if distance / scale <= max_relative_distance:
                    duplicate_of = (other, round(distance / scale, 3))
                    break
        if duplicate_of is None:
            keep.append(node)
        else:
            other, relative = duplicate_of
            merged.append({"dropped": node.get("id"), "kept": other.get("id"),
                           "class": node.get("class"), "relative_distance": relative})
    if merged:
        kept_ids = {n.get("id") for n in keep}
        program["objects"] = [n for n in objects if n.get("id") in kept_ids]
    return merged
