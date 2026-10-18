"""场景完整度：两轴分开报。

现有的反馈分数只对**已配对上的物体**求平均，所以场景里少了三十个物体一分都不扣。
这就是为什么重建出五个盒子也能一路通过门禁——没有任何指标在衡量「像不像」。

这里按两轴报，对应论文的两条主张：

* **还原程度**：画面里该有的东西有没有、尺寸对不对、有多少是真网格而不是占位盒子。
* **可交互程度**：有多少物体能上手动、动起来之后是不是真有行为。

两轴不合成一个总分。合成之后，一个「什么都没重建出来但剩下那个能拖」的场景会拿到
不低的分数，而那恰恰是我们现在最该避免的失败模式。
"""
from __future__ import annotations

import math


def _longest(node: dict) -> float | None:
    geom = node.get("geom") or {}
    for key in ("extent", "size", "scale"):
        values = geom.get(key)
        if isinstance(values, (list, tuple)) and values:
            longest = max(float(v) for v in values)
            if longest > 1e-6:
                return longest
    lengths = [float(geom[k]) * (2 if "radius" in k else 1)
               for k in ("radius", "radius_top", "radius_bottom", "height")
               if isinstance(geom.get(k), (int, float))]
    return max(lengths) if lengths and max(lengths) > 1e-6 else None


def _evidence_longest(obj: dict) -> float | None:
    boxes = obj.get("obb") or []
    sizes = [max(float(v) for v in b["size"]) for b in boxes
             if isinstance(b.get("size"), (list, tuple)) and len(b["size"]) == 3]
    return max(sizes) if sizes else None


def fidelity(program: dict, evidence: dict | None = None,
             asset_manifest: dict | None = None,
             masks: dict | None = None) -> dict:
    """还原程度。不需要渲染，所以每次生成都能顺手算一遍。"""
    objects = program.get("objects") or []
    statics = program.get("static") or []
    nodes = objects + statics
    out: dict = {"objects": len(objects), "static": len(statics)}

    # 有多少物体身上挂的是真网格，而不是占位的盒子
    requests = (asset_manifest or {}).get("requests") or []
    if requests:
        resolved = sum(1 for r in requests if str(r.get("resolved", "")).startswith("library"))
        out["asset_coverage"] = round(resolved / len(requests), 3)
        out["asset_requests"] = len(requests)

    if evidence is not None:
        ev_objects = evidence.get("objects") or []
        program_ids = {n.get("id") for n in nodes}
        from ..taxonomy import slug
        matched = [o for o in ev_objects if o.get("id") in program_ids or slug(o.get("id", "")) in program_ids]
        out["evidence_objects"] = len(ev_objects)
        out["object_recall"] = round(len(matched) / len(ev_objects), 3) if ev_objects else None

        # 尺寸一致性：程序里写的大小和证据量到的大小差几倍，取绝对对数误差的中位数。
        # 用对数是因为「大了一倍」和「小了一半」应该一样严重。
        by_id = {o["id"]: o for o in ev_objects}
        by_id.update({slug(o["id"]): o for o in ev_objects if o.get("id")})
        errors = []
        for node in objects:
            observed = _evidence_longest(by_id.get(node.get("id"), {}))
            written = _longest(node)
            if observed and written:
                errors.append(abs(math.log(written / observed)))
        if errors:
            errors.sort()
            out["size_log_error_median"] = round(errors[len(errors) // 2], 3)

    # 同一条片段里同类物体的尺寸差多少。判据准但覆盖小，所以**只报不改**：
    # 三倍门槛下标出来的六个全是假阳性（一个真物体都没误伤），但只占一百二十个里的
    # 六个，丢掉它们精确率只从 0.133 到 0.140。接进链路去删物体不值得多一道会误伤
    # 的闸，当诊断却有用——一条片段里出现三倍以上的同类尺寸差，基本可以断定某个环节
    # 出了问题，适合挑出来人工看。见 #64。
    outliers = size_outliers(objects)
    if outliers:
        out["size_outliers"] = len(outliers)
        out["size_outlier_ids"] = [row["id"] for row in outliers[:8]]

    # 画面被解释了多少。这一项是前面几项的必要补充：只看 object_recall 的话，
    # 一个塞满东西的房间只检出九个物体、九个全都重建了出来，也会得到满分，
    # 而那恰恰是「重建出来不像」最主要的来源。
    if masks:
        # 要按帧分别算再平均，不能只看某一帧。多关键帧检测出来的物体，掩码是从它被
        # 检到的那一帧才开始有的；只统计第一帧的话，物体数从 9 涨到 62 这个指标都不动，
        # 而那恰恰是这项指标存在的意义。
        import numpy as np
        per_frame: dict = {}
        for mask_set in masks.values():
            for frame_index, mask in mask_set.items():
                array = np.asarray(mask).astype(bool)
                if frame_index not in per_frame:
                    per_frame[frame_index] = np.zeros_like(array)
                if per_frame[frame_index].shape == array.shape:
                    per_frame[frame_index] |= array
        fractions = sorted(float(u.mean()) for u in per_frame.values() if u.size)
        if fractions:
            out["evidence_pixel_coverage"] = round(sum(fractions) / len(fractions), 3)
            out["evidence_pixel_coverage_best"] = round(fractions[-1], 3)
            out["covered_frames"] = len(fractions)
    return out


def size_outliers(objects: list[dict], ratio: float = 3.0,
                  min_peers: int = 3) -> list[dict]:
    """同类里尺寸离谱的那些。只报不改。

    参照的是**同一条片段里同类其它实例的尺寸中位数**，不是外部先验——后者是
    `synthesis.size_regularisation` 做的事，它默认关着，因为缩物体却不改相机距离
    会让取景整个变样（见 configs/default.yaml 的注释）。这里只问「这一个和它的同类
    比是不是离谱」，不引入「这类东西一般多大」。

    至少要 `min_peers` 个同类实例中位数才稳：只有两个实例且其中一个是错的时候，
    中位数落在两者中间，倍数判据会把好的那个一起冤枉。
    """
    import statistics
    from collections import defaultdict
    by_class: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for node in objects:
        longest = _longest(node)
        if longest:
            by_class[str(node.get("class", "")).lower()].append((node.get("id", ""), longest))
    found: list[dict] = []
    for cls, items in by_class.items():
        if len(items) < min_peers:
            continue
        median = statistics.median(size for _, size in items)
        if median <= 1e-9:
            continue
        for oid, size in items:
            factor = size / median
            if factor >= ratio or factor <= 1.0 / ratio:
                found.append({"id": oid, "class": cls, "longest_m": round(size, 3),
                              "class_median_m": round(median, 3), "ratio": round(factor, 2)})
    found.sort(key=lambda row: -abs(math.log(row["ratio"])))
    return found


def interactivity(program: dict) -> dict:
    """可交互程度。四档各报一个比例，对应 kernel/manipulate.js 里写的那张表。

    注意这里数的是程序**声明**了什么，不是实际能不能动。声明和能用是两回事：物体
    可能和桌面穿插着生成、碰撞体退化、或者整个被挡在别的东西后面，这些情况下下面
    每一项都照样是满的。真的去抓一遍、拖一遍再报的那组数在 interaction_probe.py，
    键名带 measured_ 前缀，两组并排看，差距本身才是有信息的那一项。
    """
    objects = program.get("objects") or []
    total = len(objects)
    if not total:
        return {"objects": 0}
    grabbable = sum(1 for o in objects
                    if (o.get("interaction") or {}).get("grab") not in (False, "none"))
    motions = [(o.get("motion") or {}).get("type", "static") for o in objects]
    scripted = sum(1 for t in motions if t not in ("static", "dynamic"))
    # 关节有两个来源：视频里观测到的铰接运动，以及程序声明的可操纵关节
    joint_modes = {"slide", "hinge"}
    articulated = sum(1 for o, t in zip(objects, motions)
                      if t in ("revolute", "prismatic")
                      or (o.get("interaction") or {}).get("grab") in joint_modes)
    simulated = sum(1 for t in motions if t == "dynamic")
    return {
        "objects": total,
        # 直接操纵：能用鼠标抓起来的比例。静态结构不算，它们本来就不该被拖走
        "manipulable_fraction": round(grabbable / total, 3),
        # 观测运动：从视频里恢复出了行为的比例
        "scripted_motion_fraction": round(scripted / total, 3),
        # 关节交互：抽屉、柜门这一档
        "articulated_fraction": round(articulated / total, 3),
        # 物理驱动：交给仿真自己算的比例
        "simulated_fraction": round(simulated / total, 3),
    }


def assess(program: dict, evidence: dict | None = None,
           asset_manifest: dict | None = None, masks: dict | None = None) -> dict:
    """两轴一起算。刻意不给总分，理由见模块开头。"""
    return {"fidelity": fidelity(program, evidence, asset_manifest, masks),
            "interactivity": interactivity(program)}


def format_table(report: dict) -> str:
    """给报告和日志用的一张小表。"""
    lines = ["还原程度:"]
    for key, value in report["fidelity"].items():
        lines.append(f"  {key:<26} {value}")
    lines.append("可交互程度:")
    for key, value in report["interactivity"].items():
        lines.append(f"  {key:<26} {value}")
    return "\n".join(lines)
