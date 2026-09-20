"""物体类别名的共用判断。以前这些散在感知和生成两边各写一遍，规则还不一致。"""
from __future__ import annotations
import re

DEFAULT_STATIC_CLASSES = ["wall", "floor", "ground", "table", "track", "rail", "belt", "conveyor", "road", "lawn", "grass"]

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
