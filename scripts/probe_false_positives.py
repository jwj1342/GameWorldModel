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

用法:
  python -m scripts.probe_false_positives --report <产物目录> --gt-root <真值根>
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
            ext = (n.get("geom") or {}).get("extent") or [0.3, 0.3, 0.3]
            out.append({"id": n["id"], "class": str(n.get("class", "")).lower(),
                        "pos": np.asarray(pos, float),
                        "diag": float(np.linalg.norm(np.asarray(ext, float)))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--tol", type=float, default=0.45,
                    help="配对容差（米），和 scripts/eval.py 的 --max-dist 对齐")
    a = ap.parse_args(argv)

    tot_pred = tot_matched_obj = tot_matched_static = tot_orphan = 0
    static_classes, orphan_classes = Counter(), Counter()
    per_clip = []
    for prog_path in sorted(Path(a.report).glob("*/program.json")):
        clip = prog_path.parent.name
        gt_path = Path(a.gt_root) / clip / "program.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        pred = json.loads(prog_path.read_text(encoding="utf-8"))
        gt_obj = nodes_with_pos(gt, "objects")
        gt_sta = nodes_with_pos(gt, "static")
        preds = nodes_with_pos(pred, "objects")

        # 贪心配对：先配真值 objects，配不上的再看是不是落在某个真值 static 上。
        # 这里不追求和 eval.py 的匈牙利配对完全一致——要回答的是「这个假阳性旁边
        # 有没有一个真值结构件」，粗配对足够。
        used = set()
        m_obj = m_sta = orphan = 0
        for p in preds:
            best, bd = None, float("inf")
            for g in gt_obj:
                if g["id"] in used:
                    continue
                d = float(np.linalg.norm(p["pos"] - g["pos"]))
                if d < bd:
                    best, bd = g, d
            if best is not None and bd <= a.tol:
                used.add(best["id"]); m_obj += 1; continue
            near = min((float(np.linalg.norm(p["pos"] - g["pos"])) for g in gt_sta),
                       default=float("inf"))
            if near <= a.tol:
                m_sta += 1
                static_classes[p["class"]] += 1
            else:
                orphan += 1
                orphan_classes[p["class"]] += 1
        tot_pred += len(preds); tot_matched_obj += m_obj
        tot_matched_static += m_sta; tot_orphan += orphan
        per_clip.append((clip, len(preds), m_obj, m_sta, orphan))

    if not tot_pred:
        print("没有产物"); return 1
    print(f"预测出来的物体 {tot_pred} 个（{len(per_clip)} 条片段，容差 {a.tol} m）\n")
    print(f"  配上真值 objects      {tot_matched_obj:4d}（{tot_matched_obj / tot_pred:.1%}）  <- 真正的命中")
    print(f"  落在真值 static 上    {tot_matched_static:4d}（{tot_matched_static / tot_pred:.1%}）"
          f"  <- **真东西，放错了列表**")
    print(f"  附近什么都没有        {tot_orphan:4d}（{tot_orphan / tot_pred:.1%}）  <- 凭空捏的")
    print(f"\n  现在的精确率 {tot_matched_obj / tot_pred:.3f}")
    print(f"  如果「结构件写进 objects」这类错误全修掉，精确率上限 "
          f"{tot_matched_obj / max(tot_pred - tot_matched_static, 1):.3f}")
    print("\n落在真值 static 上的，类名分布（这些是分不清结构件和物体）：")
    for c, n in static_classes.most_common(12):
        print(f"    {c or '（空）':<24} {n}")
    print("\n附近什么都没有的，类名分布（这些才是真的过检）：")
    for c, n in orphan_classes.most_common(12):
        print(f"    {c or '（空）':<24} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
