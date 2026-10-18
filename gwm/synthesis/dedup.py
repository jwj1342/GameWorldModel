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


def merge_fragment_chains(program: dict, max_gap_diagonals: float = 2.5,
                          min_chain: int = 2, min_elongation: float = 3.0,
                          same_class_only: bool = False) -> list[dict]:
    """把**排成一条线的同类小碎块**合成一个长条。

    为什么需要这一步（#56）：真值里一根轨道是 6.4 比 1 的长条，但掩码只覆盖它的
    一小段，于是一根轨道被切成好几个小块、每块都成了一个独立「物体」。实测十一条
    片段上，落在真值 static 上的 25 个检出只对应 **11 个**真值结构件（平均 2.27 次，
    其中一个被切成 6 块），碎片尺寸中位只有真值的 **0.208**。

    这解释了为什么二十条「把假的滤掉」的信号全部失败：**那些东西不是假的**，
    是真实结构件的碎片，有深度、有掩码、有一致的跨帧跟踪。没有任何信号能把一截
    真实的轨道判成假的。

    `merge_duplicates` 碰不到它们：碎片沿着轨道排开，相对距离远超它的 0.5 阈值
    （同类对的相对距离中位是 2.76）。而把那个阈值调大会把真正挨近的不同物体并掉
    （阈值 2.0 时 283 对里有 86 对会被合）。

    所以这里用一个碎片有、而真实物体没有的特征：**它们排成一条线**。一排小的、
    共线的东西是一个长条；两个碰巧挨近的不同物体不会连成链，也不会细长。

      `max_gap_diagonals`  只连**相邻**的
      `min_elongation`     合出来必须是**细长**的——这是安全阀：三个堆叠的箱子也会
                           连通，但合出来是方的不是长条，所以不合
      `min_chain`          至少几块才算链

    **两个前提我第一版设错了，量完才改：**

    `same_class_only` 默认**关着**。第一版按类名分组，实测八组真碎片里只有 **3 组**
    类名一致——同一根地面的 6 块碎片被贴了 `lava` 和 `pod` 两个名字。而这个数据集的
    类名本来就无意义（生成器从词表挑名字再独立配基本体）。按类名分组等于用一个
    已被证伪的信号去拆散正确的分组，端到端跑下来一条链都没合上。

    `min_chain` 从 3 降到 2。实测八组里只有 **3 组**有三块以上，五组是两块。降到 2
    不会抢 `merge_duplicates` 的活，因为 `min_elongation` 仍然守着：两块挨近且合起来
    细长的是碎片，合起来方的是两个物体。

    （实测的间距中位是 0.68 个对角线，所以 `max_gap_diagonals=2.5` 是宽松的，
    22 个碎片里 17 个能连上，这一条不是瓶颈。）

    **这个判据有结构缺陷，默认关着。** 单变量对照（同一快照、只切开关、十二条片段）：
    配对物体 24 -> 17、召回 0.464 -> 0.268、精确率 0.226 -> 0.200、可交互性
    0.884 -> 0.802。它确实执行了（三条片段各合掉 10、4、14 个节点），是生效了并且有害。

    机制：**连通分量没有长度上限**。碎片本身很小（尺寸中位只有真值的 0.208），
    小物体的对角线小，`max_gap_diagonals` 折算出来是个很短的绝对距离——但只要场景里
    有一排小东西，它们就会一个接一个连通成链，一条链吞掉 10 个节点（而真碎片组平均
    只有 2.27 块、最大一组 6 块）。`min_elongation` 挡不住：串得越长越细长，这个条件
    反而**鼓励**长链。

    调阈值救不了：`max_gap` 调小会让真碎片也连不上（实测碎片间距中位 0.68 个对角线，
    真物体之间也可能是这个量级）。要修得给链加**长度上限**（比如不超过场景尺度的
    若干分之一），或者换一个不靠连通分量的分组方式（比如先拟合直线再看谁落在线上）。

    返回每次合并的记录。合不成链的一个都不动。
    """
    import numpy as np

    objects = program.get("objects") or []
    if len(objects) < min_chain:
        return []

    usable = []
    for node in objects:
        centre, diag = _centre(node), _diagonal(node)
        if centre is not None and diag > 1e-6:
            usable.append((node, np.asarray(centre, float), diag))
    by_class: dict[str, list] = {}
    for item in usable:
        key = str(item[0].get("class", "")).lower() if same_class_only else ""
        by_class.setdefault(key, []).append(item)

    merged: list[dict] = []
    drop: set[int] = set()
    for cls, items in by_class.items():
        if len(items) < min_chain:
            continue
        # 相邻就连边，连通分量就是候选链
        n = len(items)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                scale = (items[i][2] + items[j][2]) / 2
                if float(np.linalg.norm(items[i][1] - items[j][1])) / scale <= max_gap_diagonals:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for members in groups.values():
            if len(members) < min_chain:
                continue
            pts = np.stack([items[i][1] for i in members])
            halves = np.stack([np.asarray((items[i][0].get("geom") or {}).get("extent")
                                          or [items[i][2] / 3] * 3, float) / 2 for i in members])
            lo, hi = (pts - halves).min(0), (pts + halves).max(0)
            extent = hi - lo
            elong = float(extent.max() / max(extent.min(), 1e-6))
            if elong < min_elongation:
                continue                       # 挤成一团而不是连成一条，不动它
            head = items[members[0]][0]
            head.setdefault("geom", {})["extent"] = [float(v) for v in extent]
            head.setdefault("pose", {})["pos"] = [float(v) for v in (lo + hi) / 2]
            head["notes"] = (f"{len(members)} 块同类碎片连成一条线，合成一个长条"
                             f"（细长比 {elong:.1f}）")[:200]
            for i in members[1:]:
                drop.add(id(items[i][0]))
            merged.append({"kept": head["id"], "absorbed": [items[i][0]["id"] for i in members[1:]],
                           "elongation": round(elong, 2), "class": cls})
    if drop:
        program["objects"] = [n for n in objects if id(n) not in drop]
    return merged
