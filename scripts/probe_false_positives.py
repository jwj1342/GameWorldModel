#!/usr/bin/env python
"""过检的那些，到底是凭空捏的，还是**真东西放错了列表**。

背景（#56）：物体精确率 0.2 出头，而九条「按检测自身的属性过滤」的信号全部排除过了
——检测置信度、被看到的帧比例、掩码占画面、资产检索成功率、三维尺寸变异系数、
逐帧中心跳变、图像检索最高分、逐短语检测。机制也想清楚了：**这些信号都只描述检测
自身**，而真假之别在于它对不对应一个真实物体，那是检测自身不含的信息。

所以这个脚本换一个问法。真值程序把节点分成两个列表：

  `objects`  会动的、可交互的东西      <- 评测只拿这些当目标
  `static`   地面、轨道、墙这类结构件   <- 评测**不**拿它们当目标

如果我们把一根轨道检出来、摆对了位置，但写进了 `objects`，那它在评测里是个**假阳性**
——哪怕检测本身完全正确。这类错误的修法（分清结构件和物体）和「过滤假检测」
完全不同，所以得先知道占比。

**必须吃 `scripts/eval.py` 的产物，不能自己比坐标。** 重建的世界原点和朝向相对真值
差一个**观测不到的**偏航加水平平移（规范自由度），评测会先对齐掉再配对。第一版我
直接比世界坐标，于是报出精确率 0.015，而评测报的是 0.226——十五倍的差，全是口径。
实测某条片段的对齐偏航是 157 度，不对齐比坐标等于在比两个不同的坐标系。

用法:
  python -m scripts.probe_false_positives --eval <eval 产物.json> --gt-root <真值根>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def nodes_with_pos(program: dict, key: str) -> list[dict]:
    out = []
    for n in program.get(key, []):
        pos = ((n.get("pose") or {}).get("pos"))
        if isinstance(pos, list) and len(pos) == 3:
            out.append({"id": n["id"], "class": str(n.get("class", "")).lower(),
                        "pos": np.asarray(pos, float)})
    return out


def gauge_of(metrics: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """把评测记下来的规范对齐取出来：预测 -> 真值的偏航和水平平移。"""
    g = ((metrics or {}).get("full") or {}).get("gauge_alignment") or {}
    yaw, txz = g.get("yaw_deg"), g.get("translation_xz_m")
    if not isinstance(yaw, (int, float)) or not isinstance(txz, list) or len(txz) != 2:
        return None
    a = np.deg2rad(float(yaw))
    R0 = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    return R0, np.array([float(txz[0]), 0.0, float(txz[1])])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="scripts/eval.py 的输出 JSON")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--tol", type=float, default=0.45, help="和 eval 的 --max-dist 对齐")
    a = ap.parse_args(argv)

    rows = json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", [])
    tot_pred = hit = on_static = orphan = 0
    static_classes, orphan_classes = Counter(), Counter()
    skipped = []
    for row in rows:
        metrics = row.get("metrics") or {}
        pairs = (metrics.get("matching") or {}).get("pairs") or {}
        gauge = gauge_of(metrics)
        run = Path(row["run"])
        prog_path = run / "program.json"
        gt_path = Path(a.gt_root) / row["clip"] / "program.json"
        if gauge is None or not prog_path.is_file() or not gt_path.is_file():
            skipped.append(row.get("clip")); continue
        R0, t0 = gauge
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        pred = json.loads(prog_path.read_text(encoding="utf-8"))
        gt_static = nodes_with_pos(gt, "static")
        matched_pred = {v for v in pairs.values() if v}
        for p in nodes_with_pos(pred, "objects"):
            tot_pred += 1
            if p["id"] in matched_pred:          # 评测说它配上了真值 object
                hit += 1
                continue
            aligned = p["pos"] @ R0.T + t0       # 和评测用同一个对齐
            near = min((float(np.linalg.norm(aligned - g["pos"])) for g in gt_static),
                       default=float("inf"))
            if near <= a.tol:
                on_static += 1; static_classes[p["class"]] += 1
            else:
                orphan += 1; orphan_classes[p["class"]] += 1

    if not tot_pred:
        print("没有可用的评测行"); return 1
    print(f"预测出来的物体 {tot_pred} 个（{len(rows) - len(skipped)} 条片段，容差 {a.tol} m）")
    if skipped:
        print(f"（跳过 {len(skipped)} 条：没有对齐信息或产物不全 {skipped}）")
    print()
    print(f"  评测判定配上真值 objects   {hit:4d}（{hit / tot_pred:.1%}）  <- 真正的命中")
    print(f"  落在真值 static 上         {on_static:4d}（{on_static / tot_pred:.1%}）"
          f"  <- **真东西，放错了列表**")
    print(f"  附近什么都没有             {orphan:4d}（{orphan / tot_pred:.1%}）  <- 凭空捏的")
    print(f"\n  现在的精确率 {hit / tot_pred:.3f}")
    print(f"  「结构件写进 objects」这类全修掉的话，精确率上限 "
          f"{hit / max(tot_pred - on_static, 1):.3f}")
    print("\n落在真值 static 上的，类名分布（分不清结构件和物体）：")
    for c, n in static_classes.most_common(12):
        print(f"    {c or '（空）':<24} {n}")
    print("\n附近什么都没有的，类名分布（这些才是真的过检）：")
    for c, n in orphan_classes.most_common(12):
        print(f"    {c or '（空）':<24} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
