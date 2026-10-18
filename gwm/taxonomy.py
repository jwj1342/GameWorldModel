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

    刻意用**完全相等**而不是子串或者分词匹配。子串会把 playground 一起吃掉；分词看着
    更聪明（能认出「wooden floor」），但同样会把「floor lamp」判成地面，而落地灯是个
    不折不扣的物体。这条规则的强度很高——它不看运动假设直接改归属——所以宁可漏不可错。
    """
    return (name or "").strip().lower() in GROUND_CLASSES


def is_static_class(name: str, cfg: dict | None = None) -> bool:
    """这个类别是固定结构（墙、地板、轨道、传送带）还是可以动的物体。

    按**词**匹配，不是按子串。`"ground" in "playground"` 是 True，`"table" in
    "tablet"` 也是——子串匹配会把游乐场滑梯和平板电脑判成固定结构。

    这个 bug 一直潜伏着，因为调用方原来要求「运动分类**肯定地说**它静止」才生效，
    而运动分类 85% 的时候返回 unknown，等于把这条规则大部分时候关掉了。把那个与
    条件换成「没有证据表明它会动」之后它立刻咬人：`playground slide` 被判成了固定
    结构（有测试专门盯这个）。

    比 `is_ground_class` 松一档是有意的：那一条用完全相等，因为它不看运动假设直接
    改归属，强度太高宁可漏不可错；这一条还要配合「有没有证据会动」一起用，所以
    允许「wooden floor」「train track」这种带修饰词的写法。
    """
    words = (cfg or {}).get("perception", {}).get("static_classes") or DEFAULT_STATIC_CLASSES
    tokens = {t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t}
    return any(w in tokens for w in words)

def slug(text: str, max_len: int = 40) -> str:
    """把任意名字变成合法的程序 id：字母开头，只含字母数字下划线（和 schema 的 pattern 一致）。"""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", text or "").strip("_")
    if not s or not s[0].isalpha(): s = "o_" + s
    return s[:max_len]
