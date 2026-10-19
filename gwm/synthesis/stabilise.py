"""把物理稳定之后的位姿写回程序。

`harness/settle.mjs` 负责跑仿真，这里只负责决定哪些结果该采纳。分开是因为两件事的
性质不同：仿真是确定性的测量，采纳与否是带阈值的判断，后者需要能单独测试。

位移超过上限的不采纳。那种情况多半不是「稍微悬空了一点」，而是这个物体根本没有支撑，
或者尺寸估错了——让它掉下去只会把一个看得见的错误变成一个看不见的错误。
"""
from __future__ import annotations

DEFAULT_MAX_SHIFT_M = 0.8
MIN_SHIFT_M = 0.01          # 小于一厘米就不算动过，免得记一堆噪声


def apply_settled(program: dict, report: dict,
                  max_shift_m: float = DEFAULT_MAX_SHIFT_M) -> list[dict]:
    """就地更新物体位姿，返回每条的采纳结果。"""
    by_id = {node["id"]: node for node in program.get("objects", [])}
    applied = []
    for row in (report or {}).get("settled", []):
        node = by_id.get(row.get("id"))
        moved = float(row.get("moved", 0.0))
        if node is None or moved < MIN_SHIFT_M:
            continue
        record = {"id": row["id"], "moved_m": round(moved, 4), "applied": moved <= max_shift_m}
        if not record["applied"]:
            record["reason"] = "shift_exceeds_limit"
        else:
            pose = node.setdefault("pose", {})
            pose["pos"] = [float(v) for v in row["pos"]]
            if row.get("quat"):
                pose["quat"] = [float(v) for v in row["quat"]]
        applied.append(record)
    return applied
