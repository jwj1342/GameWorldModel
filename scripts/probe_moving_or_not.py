#!/usr/bin/env python
"""「会不会动」这个二分，信息到底在不在候选分数里。

背景（#7）：运动这一级现在连二分都不成立——八个配置里六个「说会动」比瞎猜还不准，
查全只有 0.11–0.24。但那是**最终写进程序的结论**，而最终结论要先过两道闸：
`min_primary_confidence` 和 `min_hypothesis_margin`。过不去就是 `unknown`，
而 `unknown` 到了合成那边等于 static。

所以有一个问题还没问过：**信息是不是其实在候选分数里，只是被闸挡掉了？**

这个脚本不重跑感知，直接读证据里的 `candidate_details`，对每个物体比较

    静止候选的置信度    对上    非静止候选里最高的那个

然后看这个比较本身（不经过任何阈值）能不能把真会动的和真静止的分开。分得开，
修法就是「别让二分跟着七分类一起被闸掉」；分不开，就得换信号。

用法:
  python -m scripts.probe_moving_or_not --report $GWM_SCRATCH/dataset/r2_report \
      --gt-root $GWM_SCRATCH/dataset/ab015
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def best_scores(motion_guess: dict) -> tuple[float | None, float | None]:
    """(静止候选的置信度, 非静止候选里最高的置信度)。缺一个就给 None。"""
    static_score = moving_score = None
    for cand in motion_guess.get("candidate_details") or []:
        conf = cand.get("confidence")
        if not isinstance(conf, (int, float)):
            continue
        if cand.get("type") == "static":
            static_score = float(conf) if static_score is None else max(static_score, float(conf))
        else:
            moving_score = float(conf) if moving_score is None else max(moving_score, float(conf))
    return static_score, moving_score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="产物目录，每条片段一个子目录")
    ap.add_argument("--gt-root", required=True, help="真值程序的根目录")
    a = ap.parse_args(argv)

    rows = []
    for ev_path in sorted(Path(a.report).glob("*/perception/evidence.json")):
        clip = ev_path.parent.parent.name
        gt_path = Path(a.gt_root) / clip / "program.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        kinds = {n["id"]: (n.get("motion") or {}).get("type", "static") for n in gt.get("objects", [])}
        kinds.update({n["id"]: "static" for n in gt.get("static", [])})
        for o in json.loads(ev_path.read_text(encoding="utf-8")).get("objects", []):
            truth = kinds.get(o["id"])
            if truth is None:
                continue
            s, m = best_scores(o.get("motion_guess") or {})
            rows.append({"clip": clip, "id": o["id"], "moving_truth": truth != "static",
                         "static_score": s, "moving_score": m,
                         "final": (o.get("motion_guess") or {}).get("type", "unknown")})
    if not rows:
        print("没有样本"); return 1

    have = [r for r in rows if r["static_score"] is not None and r["moving_score"] is not None]
    truth_moving = sum(r["moving_truth"] for r in rows)
    print(f"物体 {len(rows)}，真值会动的 {truth_moving}（{truth_moving / len(rows):.1%}）")
    print(f"两个候选都在的 {len(have)}——只有这些才谈得上比较\n")
    if not have:
        print("没有一个物体同时有静止候选和非静止候选，这本身就是结论")
        return 0

    # 不过任何阈值，直接比两个候选谁高
    hit = sum(1 for r in have if (r["moving_score"] > r["static_score"]) == r["moving_truth"])
    said_moving = [r for r in have if r["moving_score"] > r["static_score"]]
    prec = sum(r["moving_truth"] for r in said_moving) / len(said_moving) if said_moving else float("nan")
    chance = sum(r["moving_truth"] for r in have) / len(have)
    recall = (sum(r["moving_truth"] for r in said_moving)
              / max(sum(r["moving_truth"] for r in have), 1))
    print("直接比候选分数（不过任何阈值）：")
    print(f"  二分判对        {hit}/{len(have)} = {hit / len(have):.3f}")
    print(f"  说会动的        {len(said_moving)} 个，命中率 {prec:.3f}  瞎猜 {chance:.3f}"
          f"  差 {prec - chance:+.3f}")
    print(f"  会动查全        {recall:.3f}")

    print("\n对比：链路最终写出来的结论（过了闸之后）")
    final_moving = [r for r in rows if r["final"] not in ("static", "unknown")]
    fprec = (sum(r["moving_truth"] for r in final_moving) / len(final_moving)
             if final_moving else float("nan"))
    fchance = truth_moving / len(rows)
    print(f"  说会动的        {len(final_moving)} 个，命中率 {fprec:.3f}  瞎猜 {fchance:.3f}"
          f"  差 {fprec - fchance:+.3f}")
    print(f"  unknown 的      {sum(1 for r in rows if r['final'] == 'unknown')} 个")

    print("\n读法：上面那个差明显大于下面那个，说明信息本来在候选分数里、是被闸挡掉的")
    print("     ——那样修法就是别让二分跟着七分类一起被闸掉。两个差不多（或都不为正），")
    print("     就说明候选分数本身没有信息，得换信号。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
