#!/usr/bin/env python
"""凭什么把一个节点判成固定结构——现在靠类名，而类名在这个数据集上没用。

量到的事实（#56）：位置配上的 39 对里，真值是结构件的有 16 个，我们只把 **1 个**
归进了 static，另外 15 个全写成了可动物体。

```
object->object   22
static->object   15      <- 真值是结构件，我们写成了可动物体
object->static    1
static->static    1      <- 只有这一个归对了
```

机制很清楚。`direct.py` 的判据是「类名像结构件（is_static_class）**并且**没有证据
表明它会动」。而 `is_static_class` 吃的是**预测的**类名，这个链路的类名准确率只有
12.5%——真值里的 `rail`（45 个）和 `ground`（12 个）本来都在静态词表里，但它们被
检测器叫成了别的名字，于是这条规则压根没触发。**判据挂在一个 12% 准确的信号上。**

## 所以换一个不靠类名的信号

真值这边两个候选，直接量：

| | 真值 static（69 个）| 真值 objects（59 个）|
|---|---|---|
| 扁平度 = y 边 / 水平最长边 | **0.156**（四分位全是 0.156）| **0.591**（0.311 – 0.822）|
| 中心高度 y | 0.019 | 0.240 |
| y < 0.1 的比例 | 0.841 | 0.254 |

扁平度在真值上几乎完全分开。**但那是真值。** 这个脚本要回答的是另一个问题：

  **我们自己拟合出来的包围盒，还保留着这个区分度吗？**

不一定。包围盒是在部分可见的点云上拟合的，看不见的那一维会被系统性低估，而结构件
通常只能看到朝上的一面。所以必须在**预测节点**上量，按它配上的真值类别分组。

还有一条要先说清楚：真值 static 的扁平度四分位**全是 0.156**，说明这个生成器对
所有 `rail` 用的是同一个扁片形状。在它上面调出来的阈值是**对着这个生成器过拟合**
的，换数据集很可能不成立。这个脚本量的是「有没有信号」，不是「阈值定在哪」。

用法:
  python -m scripts.probe_structure_signal --report <产物目录> --eval <eval.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.feedback.gt_metrics import node_extent   # noqa: E402


def flatness(node: dict) -> float | None:
    e = node_extent(node)
    if e is None or len(e) != 3 or not np.isfinite(e).all():
        return None
    horizontal = float(max(e[0], e[2]))
    return float(e[1]) / horizontal if horizontal > 1e-6 else None


def describe(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name:<22} （没有样本）"
    a = np.sort(np.asarray(values, float))
    return (f"  {name:<22} n={len(a):<4} 中位 {np.median(a):.3f}  "
            f"四分位 {np.percentile(a, 25):.3f} / {np.percentile(a, 75):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gt-root", default=None,
                    help="给了就逐对比预测和真值的边长，分清「撑厚」和「切短」")
    a = ap.parse_args(argv)

    rows = json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", [])
    buckets: dict[str, list[float]] = {"真值是结构件": [], "真值是可动物体": [], "没配上": []}
    heights: dict[str, list[float]] = {k: [] for k in buckets}
    dims: list[tuple[float, float]] = []       # (y 边之比, 水平最长边之比)，只对结构件

    for r in rows:
        scene = ((r.get("metrics") or {}).get("full") or {}).get("scene") or {}
        pairs = scene.get("pairs") or {}
        prog_path = Path(a.report) / r["clip"] / "program.json"
        if not prog_path.is_file():
            continue
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        gt_nodes: dict[str, dict] = {}
        if a.gt_root:
            gtp = Path(a.gt_root) / r["clip"] / "program.json"
            if gtp.is_file():
                g = json.loads(gtp.read_text(encoding="utf-8"))
                gt_nodes = {f"static::{n['id']}": n for n in (g.get("static") or [])}
                gt_nodes.update({n["id"]: n for n in (g.get("objects") or [])})
        nodes = {n["id"]: n for n in (prog.get("objects") or [])}
        nodes.update({f"static::{n['id']}": n for n in (prog.get("static") or [])})
        claimed = set()
        for gt_key, pred_key in pairs.items():
            if pred_key is None or pred_key not in nodes:
                continue
            claimed.add(pred_key)
            bucket = "真值是结构件" if gt_key.startswith("static::") else "真值是可动物体"
            f = flatness(nodes[pred_key])
            if f is not None:
                buckets[bucket].append(f)
            pos = (nodes[pred_key].get("pose") or {}).get("pos")
            if isinstance(pos, list) and len(pos) == 3:
                heights[bucket].append(float(pos[1]))
            if bucket == "真值是结构件" and gt_key in gt_nodes:
                pe, ge = node_extent(nodes[pred_key]), node_extent(gt_nodes[gt_key])
                if pe is not None and ge is not None and len(pe) == 3 and len(ge) == 3:
                    gh, ph = float(max(ge[0], ge[2])), float(max(pe[0], pe[2]))
                    if ge[1] > 1e-6 and gh > 1e-6:
                        dims.append((float(pe[1]) / float(ge[1]), ph / gh))
        for key, node in nodes.items():
            if key in claimed:
                continue
            f = flatness(node)
            if f is not None:
                buckets["没配上"].append(f)
            pos = (node.get("pose") or {}).get("pos")
            if isinstance(pos, list) and len(pos) == 3:
                heights["没配上"].append(float(pos[1]))

    print("我们自己拟合出来的包围盒，按它配上的真值类别分组\n")
    if dims:
        print("先分清一件事：扁平度变大，可能是**薄的一维被撑厚**，也可能是")
        print("**长的一维被切短**（掩码只盖住了结构件的一段）。两者该做的事完全不同。")
        print("下面按配上的真值节点逐对比，分母是真值自己的边长。\n")
        for label, idx in (("薄的那一维（y 边）", 0), ("水平最长边", 1)):
            vals = [d[idx] for d in dims if np.isfinite(d[idx])]
            if vals:
                a = np.sort(np.asarray(vals))
                print(f"  {label:<18} n={len(a):<4} 预测/真值 中位 {np.median(a):.2f}  "
                      f"四分位 {np.percentile(a,25):.2f} / {np.percentile(a,75):.2f}")
        print("  中位大于 1 就是我们做大了，小于 1 就是做小了。\n")
    print("扁平度 = y 边 / 水平最长边（真值上 static 0.156、object 0.591，几乎不重叠）")
    for k in buckets:
        print(describe(k, buckets[k]))
    print("\n中心高度 y（真值上 static 0.019、object 0.240）")
    for k in heights:
        print(describe(k, heights[k]))

    s, o = buckets["真值是结构件"], buckets["真值是可动物体"]
    print()
    if s and o:
        gap = abs(float(np.median(s)) - float(np.median(o)))
        print(f"两组扁平度中位数相差 {gap:.3f}。真值上这个差是 0.435。")
        print("差要是塌到接近零，就说明包围盒把区分度磨没了——那这条路不是换个阈值")
        print("能救的，得回到包围盒怎么拟合，或者另找信号。")
    else:
        print("有一组没有样本，结论不成立。")
    print("\n注意：真值 static 的扁平度四分位全是 0.156，这个生成器对所有 rail 用的是")
    print("同一个扁片形状。在它上面调出来的阈值是对着这个生成器过拟合的。这里量的是")
    print("「有没有信号」，不是「阈值定在哪」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
