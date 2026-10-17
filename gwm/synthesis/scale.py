"""用物体的真实尺寸把场景的度量尺度定下来。

单目视频恢复不出绝对尺度，所以总得有个锚。现在用的锚是「相机离地 1.2 米」
（`gwm/perception/evidence.py` 里的 `assumed_camera_height_m`）。这个假设在站着拍房间
时还行，但只要是俯拍桌面——而桌面恰好会被当成地面——相机其实只高出 0.5 米左右，
整个场景就会被放大两到三倍。

更稳的锚是物体本身：杯子就是十厘米上下，桌子一米四上下。做法是拿每个认得出类别的
物体，比一比「它该多大」和「我们重建出它多大」，取中位数当全局改正。用中位数而不是
均值，是因为包围盒拟合到可见表面，个别物体会错得很离谱。

尺寸先验有两个来源，优先用前者：

1. **视觉语言模型给的**。命名那一步本来就问过每个物体的 `size_m`，答案一直存在
   `namer.json` 里，但从来没有被用过。这个模型看过画面，它的估计比按类名查表准。
2. **下面这张表**。没有模型时的兜底，也用于模型没提到的类别。

改正只作用于程序，不改证据：证据记的是「我们量到了什么」，这里做的是「我们认为它
实际该多大」，两件事不该混在一份记录里。改正倍数会写进 `meta.metric_scale` 备查。
"""
from __future__ import annotations

import statistics

# 常见物体的最长边，单位米。只收日常场景里尺寸稳定、不容易有歧义的类别。
# 多词键优先匹配：「toy train」是玩具不是火车，必须排在「train」前面被选中。
SIZE_PRIORS_M: dict[str, float] = {
    "toy train": 0.20, "toy car": 0.12, "toy": 0.20,
    "cup": 0.10, "mug": 0.10, "glass": 0.12, "bottle": 0.25, "bowl": 0.18, "plate": 0.25,
    "book": 0.25, "laptop": 0.35, "keyboard": 0.45, "monitor": 0.60, "phone": 0.15,
    "lamp": 0.45, "pillow": 0.60, "blanket": 1.60, "towel": 0.80,
    "chair": 0.90, "stool": 0.45, "table": 1.40, "desk": 1.40, "sofa": 2.00, "couch": 2.00,
    "bed": 2.00, "bookshelf": 1.80, "shelf": 1.00, "cabinet": 1.20, "drawer": 0.50,
    "door": 2.00, "window": 1.20, "rug": 2.00, "carpet": 2.50,
    "box": 0.40, "carton": 0.40, "crate": 0.50, "basket": 0.40, "bag": 0.40,
    "ball": 0.22, "plant": 0.60, "bicycle": 1.70, "person": 1.70,
    "conveyor belt": 3.00, "railroad track": 0.30, "train track": 0.30,
}

# 改正倍数超出这个区间就不信：那说明不是尺度差一点，而是哪一层估错了，
# 硬改会把一个看得出来的错误变成一个看不出来的错误。
MIN_FACTOR, MAX_FACTOR = 0.05, 20.0

# 全局尺度只采信长边不小于这个值的物体，小物体的包围盒被深度噪声撑大得太厉害
LARGE_OBJECT_M = 0.5

# 单个物体的尺寸允许偏离先验多少倍。超出就按先验收回来——留着一个比火车还大的
# 积木，画面上一眼就是错的，而这种错误没有任何下游环节能纠正
MAX_OBJECT_DEVIATION = 2.0


def size_prior_for(class_name: str) -> float | None:
    """按类名查最长边先验。多词键优先，所以「toy train」不会被「train」抢走。"""
    text = (class_name or "").lower()
    best_key = None
    for key in SIZE_PRIORS_M:
        if key in text and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return SIZE_PRIORS_M[best_key] if best_key else None


def _longest_side(node: dict) -> float | None:
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
    longest = max(lengths, default=0.0)
    return longest if longest > 1e-6 else None


def estimate_metric_scale(program: dict, vlm_sizes: dict[str, list[float]] | None = None,
                          min_objects: int = 2) -> tuple[float, dict]:
    """返回 (改正倍数, 诊断)。认不出足够多的物体时返回 1.0，并说明原因。"""
    report: dict = {"samples": [], "factor": 1.0, "source": "none"}
    sizes = {str(k).lower(): v for k, v in (vlm_sizes or {}).items()}
    ratios = []
    for node in program.get("objects", []):
        observed = _longest_side(node)
        if not observed:
            continue
        class_name = str(node.get("class") or "")
        prior, source = None, None
        vlm = sizes.get(class_name.lower())
        if isinstance(vlm, (list, tuple)) and vlm:
            numbers = [float(v) for v in vlm if isinstance(v, (int, float)) and v > 0]
            if numbers:
                prior, source = max(numbers), "vlm"
        if prior is None:
            prior = size_prior_for(class_name)
            source = "table" if prior is not None else None
        if prior is None:
            continue
        ratios.append(prior / observed)
        report["samples"].append({"id": node.get("id"), "class": class_name, "source": source,
                                  "prior_m": round(prior, 3), "observed_m": round(observed, 3),
                                  "ratio": round(prior / observed, 3)})
    if len(ratios) < min_objects:
        report["reason"] = f"only {len(ratios)} objects had a size prior, need {min_objects}"
        return 1.0, report
    # 全局尺度只听大物体的。包围盒拟合的是可见表面，深度噪声是个绝对量，物体越小
    # 它占的比例越大——实测同一段视频里，桌子和书架的比例在 0.8~1.2 之间，玩具火车
    # 却被量成了一米多。拿小物体一起投票，等于让噪声最大的样本决定全场尺度。
    large = [s["ratio"] for s in report["samples"] if s["observed_m"] >= LARGE_OBJECT_M]
    if len(large) >= min_objects:
        ratios, report["anchor"] = large, f"objects >= {LARGE_OBJECT_M} m"
    else:
        report["anchor"] = "all objects with a prior"
    factor = float(statistics.median(ratios))
    report["spread"] = round(max(ratios) / min(ratios), 2) if min(ratios) > 0 else None
    if not MIN_FACTOR <= factor <= MAX_FACTOR:
        report["reason"] = f"factor {factor:.3f} outside [{MIN_FACTOR}, {MAX_FACTOR}]"
        return 1.0, report
    report["factor"] = round(factor, 4)
    report["source"] = "vlm" if any(s["source"] == "vlm" for s in report["samples"]) else "table"
    return factor, report


def regularize_object_sizes(program: dict, vlm_sizes: dict[str, list[float]] | None = None,
                            max_deviation: float = MAX_OBJECT_DEVIATION) -> list[dict]:
    """把尺寸明显不合理的物体按先验收回来，就地修改程序，返回改动记录。

    这是**先验压过测量**，和全局尺度那一步性质不同，所以单独做、单独记。理由是实测下来
    小物体的包围盒错得没有规律可言：同一段视频里，一个玩具火车被量成一米一，而旁边的
    桌子只差百分之十几。留着这种物体，画面上一眼就是错的，下游也没有任何环节能纠正它。

    只改尺寸，不改中心、不改朝向、不改运动。倍数在容许范围内的物体一个字节都不动——
    这一步应该是少数情况下的兜底，不是默认把所有东西都换成先验。
    """
    sizes = {str(k).lower(): v for k, v in (vlm_sizes or {}).items()}
    changes = []
    for node in program.get("objects", []):
        observed = _longest_side(node)
        if not observed:
            continue
        class_name = str(node.get("class") or "")
        vlm = sizes.get(class_name.lower())
        prior = None
        if isinstance(vlm, (list, tuple)):
            numbers = [float(v) for v in vlm if isinstance(v, (int, float)) and v > 0]
            prior = max(numbers) if numbers else None
        if prior is None:
            prior = size_prior_for(class_name)
        if prior is None:
            continue
        deviation = observed / prior
        if 1 / max_deviation <= deviation <= max_deviation:
            continue
        factor = prior / observed
        geom = node["geom"]
        for key in ("extent", "size", "scale"):
            if isinstance(geom.get(key), (list, tuple)):
                geom[key] = [float(x) * factor for x in geom[key]]
        for key in ("radius", "radius_top", "radius_bottom", "height"):
            if isinstance(geom.get(key), (int, float)):
                geom[key] = float(geom[key]) * factor
        changes.append({"id": node.get("id"), "class": class_name,
                        "observed_m": round(observed, 3), "prior_m": round(prior, 3),
                        "factor": round(factor, 3)})
    return changes
