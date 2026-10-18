"""物体类别名的共用判断。以前这些散在感知和生成两边各写一遍，规则还不一致。"""
from __future__ import annotations
import re

DEFAULT_STATIC_CLASSES = ["wall", "floor", "ground", "table", "track", "rail", "belt", "conveyor", "road", "lawn", "grass"]

GROUND_CLASSES = {"floor", "ground", "terrain"}


def is_ground_class(name: str) -> bool:
    """地面、地板、地形。

    和 is_static_class 分开，是因为两者的强度不一样。桌子、轨道「通常」不动，但判出
    它在动未必是错的——轨道可能真的是传送带，所以那一类还要看运动假设。地面不一样：
    地面在动一定是估计错了，是相机动了而配准把相机的运动算到了地面头上。所以这一类
    不看运动假设，一律按固定结构处理。

    实测的后果：有片段把 1.9 x 0.4 x 1.5 米的「ground」建成了一个沿轨迹运动的可拖动
    物体，它横在开场视角的最前面，后面十一个物体一个都点不到。
    """
    return (name or "").strip().lower() in GROUND_CLASSES


def is_static_class(name: str, cfg: dict | None = None) -> bool:
    """这个类别是固定结构（墙、地板、轨道、传送带）还是可以动的物体。"""
    words = (cfg or {}).get("perception", {}).get("static_classes") or DEFAULT_STATIC_CLASSES
    low = (name or "").lower()
    return any(w in low for w in words)

def slug(text: str, max_len: int = 40) -> str:
    """把任意名字变成合法的程序 id：字母开头，只含字母数字下划线（和 schema 的 pattern 一致）。"""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", text or "").strip("_")
    if not s or not s[0].isalpha(): s = "o_" + s
    return s[:max_len]
