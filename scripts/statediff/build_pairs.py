#!/usr/bin/env python
"""用内核的执行结果给同一输入的多份采样排序，组偏好对。

分数只看观测：把采样出的 motion 换进程序、渲染末帧，逐部件和观测到的末帧掩码比 IoU，取部件平均。
不用真值程序，所以没有程序标注的输入（只要有两端的部件掩码）也能组对。写不出合法程序的记 0。
每条输入取分最高和最低的一份，分差够大才成对。

用法:
  python scripts/statediff/build_pairs.py --rows EVAL/rows.jsonl --pred s0=P0.jsonl --pred s1=P1.jsonl ... --out pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def score(row: dict) -> float:
    vals = [p["iou_tT"] for p in row.get("per_part", []) if p["iou_tT"] is not None]
    return float(np.mean(vals)) if row["valid"] and vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", action="append", required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=predictions.jsonl，name 是评测时的来源名")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gap", type=float, default=0.15)
    a = ap.parse_args()

    texts = {}
    for spec in a.pred:
        name, path = spec.split("=", 1)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                texts[(name, r["id"])] = r["text"]
    by: dict[str, list[tuple[float, str, str]]] = {}
    for p in a.rows:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if (r["source"], r["id"]) in texts:
                    by.setdefault(r["id"], []).append((score(r), r["source"], texts[(r["source"], r["id"])]))
    pairs, gaps = [], []
    for sid, cands in sorted(by.items()):
        cands.sort(key=lambda c: c[0])
        lo, hi = cands[0], cands[-1]
        gaps.append(hi[0] - lo[0])
        if hi[0] - lo[0] >= a.min_gap:
            pairs.append({"id": sid, "chosen": hi[2], "rejected": lo[2], "chosen_score": round(hi[0], 4),
                          "rejected_score": round(lo[0], 4), "chosen_source": hi[1], "rejected_source": lo[1]})
    Path(a.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    print(json.dumps({"inputs": len(by), "pairs": len(pairs), "min_gap": a.min_gap,
                      "gap_median": float(np.median(gaps)) if gaps else None,
                      "chosen_score_mean": float(np.mean([p["chosen_score"] for p in pairs])) if pairs else None,
                      "rejected_score_mean": float(np.mean([p["rejected_score"] for p in pairs])) if pairs else None},
                     ensure_ascii=False))
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
