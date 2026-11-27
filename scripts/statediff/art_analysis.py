#!/usr/bin/env python
"""铰接探针的两张附表，都从 evaluate_art.py 的 rows.jsonl 算，不重新渲染。

pairs     只换样式的配对：同一个来源对 base 和 _style 两条写出的关节是否一致（行为半边该不随样式变）。
          一致 = 两条动的部件集合相同，且每个部件类型相同、轴夹角 < 20°、铰链线相距 < 0.15 倍部件尺寸、末态转差 < 15° 或行程差 < 0.3。
          另报两条都做对（success）的比例，免得「一致地错」被当成好事。
diversity 同一输入采样多次（来源名形如 lora_s0、lora_s1…）：每条输入成功与否是不是一边倒。
          成败混杂的输入才组得出偏好对；全对或全错的输入，偏好优化在它上面学不到东西。

用法:
  python scripts/statediff/art_analysis.py pairs --samples SET/samples.jsonl --rows OUT/rows.jsonl [--rows ...]
  python scripts/statediff/art_analysis.py diversity --rows OUT/rows.jsonl --prefix lora_s
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.statediff.art import extract_motions                                     # noqa: E402
from scripts.statediff.common import load_samples, read_json                          # noqa: E402
from scripts.statediff.evaluate_art import angle_at, dist_at, _ang, _rot, _unit       # noqa: E402


def _same_joint(a: dict, b: dict, extent, T: float) -> bool:
    ta, tb = a.get("type", "static"), b.get("type", "static")
    if ta != tb:
        return False
    if ta == "revolute":
        ax = _ang(a.get("axis", [0, 1, 0]), b.get("axis", [0, 1, 0]))
        rel = np.asarray(a.get("pivot", [0, 0, 0]), float) - np.asarray(b.get("pivot", [0, 0, 0]), float)
        piv = float(np.linalg.norm(np.cross(rel, _unit(b.get("axis", [0, 1, 0]))))) / max(extent)
        Ra, Rb = _rot(a.get("axis", [0, 1, 0]), angle_at(a, T)), _rot(b.get("axis", [0, 1, 0]), angle_at(b, T))
        rot = float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
        return min(ax, 180 - ax) < 20 and piv < 0.15 and rot < 15
    if ta == "prismatic":
        da = _unit(a.get("axis", [1, 0, 0])) * dist_at(a, T)
        db = _unit(b.get("axis", [1, 0, 0])) * dist_at(b, T)
        return float(np.linalg.norm(da - db)) < 0.3 * max(float(np.linalg.norm(da)), 1e-6)
    return True


def pairs(a) -> int:
    samples = {s["id"]: s for s in load_samples(a.samples)}
    rows = [json.loads(l) for p in a.rows for l in Path(p).read_text().splitlines() if l.strip()]
    preds = {}
    for p in a.pred:
        name, path = p.split("=", 1)
        for l in Path(path).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                preds[(name, r["id"])] = extract_motions(r.get("json"))
    success = {(r["source"], r["id"]): r["success"] for r in rows}
    out = {}
    for sid, s in samples.items():
        if "style_variant" not in s.get("tags", []):
            continue
        base = f"{s['set']}/{s['tags'][1]}"
        prog = read_json(s["program"])
        for name in {n for n, _ in preds}:
            pa, pb = preds.get((name, base)), preds.get((name, sid))
            same = (pa is not None and pb is not None and all(
                _same_joint(pa.get(o["id"], {"type": "static"}), pb.get(o["id"], {"type": "static"}), o["geom"]["extent"], float(s["duration"]))
                for o in prog["objects"]))
            e = out.setdefault(name, {"pairs": 0, "consistent": 0, "both_success": 0})
            e["pairs"] += 1
            e["consistent"] += int(same)
            e["both_success"] += int(bool(success.get((name, base))) and bool(success.get((name, sid))))
    print("| 来源 | 配对 | 行为一致 | 两条都成功 |\n|---|---:|---:|---:|")
    for name, e in sorted(out.items()):
        print(f"| {name} | {e['pairs']} | {e['consistent'] / e['pairs']:.2f} | {e['both_success'] / e['pairs']:.2f} |")
    return 0


def diversity(a) -> int:
    rows = [json.loads(l) for p in a.rows for l in Path(p).read_text().splitlines() if l.strip()]
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["source"].startswith(a.prefix):
            by.setdefault(r["id"], []).append(r)
    k = {len(v) for v in by.values()}
    succ = {i: [r["success"] for r in v] for i, v in by.items()}
    mixed = sum(1 for v in succ.values() if 0 < sum(v) < len(v))
    all_ok = sum(1 for v in succ.values() if sum(v) == len(v))
    spread = [max(r["iou_move_tT"] or 0 for r in v) - min(r["iou_move_tT"] or 0 for r in v) for v in by.values()]
    print(json.dumps({"inputs": len(by), "samples_per_input": sorted(k), "mixed_success": mixed, "all_success": all_ok,
                      "all_fail": len(by) - mixed - all_ok, "mean_success": float(np.mean([np.mean(v) for v in succ.values()])),
                      "best_of_k_success": float(np.mean([max(v) for v in succ.values()])),
                      "iou_move_spread_median": float(np.median(spread))}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--samples", required=True)
    p.add_argument("--rows", action="append", required=True)
    p.add_argument("--pred", action="append", required=True, help="name=predictions.jsonl，name 要和评测时一致")
    d = sub.add_parser("diversity")
    d.add_argument("--rows", action="append", required=True)
    d.add_argument("--prefix", default="lora_s")
    a = ap.parse_args()
    return pairs(a) if a.cmd == "pairs" else diversity(a)


if __name__ == "__main__":
    sys.exit(main())
