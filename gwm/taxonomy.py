"""物体类别名的共用判断。以前这些散在感知和生成两边各写一遍，规则还不一致。"""
from __future__ import annotations
import re

# **必须和 `configs/default.yaml` 的 `perception.static_classes` 保持一致。**
# 生产路径读 yaml，而任何不带配置的调用（测试、直接调 `is_static_class(cls, {})`）
# 读这里的兜底——两边不一致的时候，同一个类名会得到两个答案，而且不会报错。
# 之前就是这样：yaml 里有 shelf/counter/desk/carpet/mat/base，这里没有。
DEFAULT_STATIC_CLASSES = ["wall", "floor", "ground", "table", "track", "rail",
                          "block", "slab", "beam", "belt", "conveyor", "road",
                          "shelf", "counter", "desk", "lawn", "grass",
                          "carpet", "mat", "base"]

GROUND_CLASSES = {"floor", "ground", "terrain"}

# **本来就不该动的结构件**。和 static_classes 的区别在于：传送带、桌子「通常不动
# 但可以动」，而墙、地板、轨道、方块是**建筑的一部分**，动了基本就是感知错了。
#
# 为什么要分这一层：判结构件原来是 `is_static_class(cls) and not has_motion`，
# 而 has_motion 里那个 `moves` 标志在**静止轨道上假阳率 88.2%**（60/68，真值里
# rail 全是 static）——多半是 ICP 在相机运动下无中生有解出转动。于是结构件规则
# 被假阳整个关掉，rail 进 objects、标成 asset、送进资产检索，而真值里结构件是
# primitive，**根本没有「该挑哪个」这个答案**。实测 273 个资产决策里 151 个
# （55.3%）是这么来的，命中恒为 0。
#
# 位移也救不了：实测 rail 的「位移/对角线」中位 1.81，而真物体是 1.93，
# **分不开**（包围盒中心本来就在漂）。所以只能靠类名。
#
# 对这一档，二元的 `moves` 不算证据，只有**明确的运动类型**才算。
# `belt`/`conveyor`/`table` 不在这一档，行为不变。
RIGID_STRUCTURE_CLASSES = {"wall", "floor", "ground", "terrain", "rail", "track",
                           "road", "block", "slab", "beam", "lawn", "grass",
                           "carpet", "mat"}


def is_rigid_structure(name: str) -> bool:
    """类名是不是「建筑的一部分」。按词匹配，和 is_static_class 一致。"""
    tokens = {t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t}
    return any(w in tokens for w in RIGID_STRUCTURE_CLASSES)



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
