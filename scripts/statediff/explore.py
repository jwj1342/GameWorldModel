#!/usr/bin/env python
"""用内核在关节假设空间里搜：每个部件枚举一组关节假设，渲染末帧，和观测到的末帧掩码比，逐部件取最好的。

假设空间对每个部件都一样：不动；铰链沿盒子 12 条棱之一、两个转向、几个张角；滑轨沿盒子三根轴的正负、几档行程；
再加上模型自己的预测。第 k 份候选把每个部件都设成第 k 个假设，一次渲染就给所有部件各打一个分；
最后逐部件挑分最高的假设拼起来（和模型自己的预测打平时留模型的）。

只用观测（末帧部件掩码）打分，不看真值程序，所以可以在没有程序标注的输入上跑，产物当伪标签。

用法:
  python scripts/statediff/explore.py --samples POOL/samples.jsonl --pred MODEL.jsonl --out DIR --workers 48
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.statediff.art import extract_motions                       # noqa: E402
from scripts.statediff.common import load_samples, read_json            # noqa: E402
from scripts.statediff.evaluate_art import eval_one                      # noqa: E402

ANGLES = (75.0,)          # 第一轮只定棱和方向，张角留给第二轮细扫
TRAVEL = (0.6,)


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hypotheses(obj: dict, T: float) -> list[dict]:
    pos = np.asarray(obj["pose"]["pos"], float)
    R = _quat_mat(obj["pose"].get("quat", [0, 0, 0, 1]))
    h = np.asarray(obj["geom"]["extent"], float) / 2
    sched = lambda key, v: [{"t": 0.5, key: round(v, 4), "duration": round(T - 1.0, 3)}]
    out = [{"type": "static"}]
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        for sb in (-1, 1):
            for sc in (-1, 1):
                pivot = pos + sb * h[b] * R[:, b] + sc * h[c] * R[:, c]
                for sign in (-1, 1):
                    axis = sign * R[:, a]
                    for ang in ANGLES:
                        out.append({"type": "revolute", "axis": [round(float(v), 4) for v in axis],
                                    "pivot": [round(float(v), 4) for v in pivot], "range_deg": [0, ang], "from_deg": 0,
                                    "schedule": sched("to_deg", ang)})
    for a in range(3):
        for sign in (-1, 1):
            for frac in TRAVEL:
                d = frac * 2 * h[a]
                out.append({"type": "prismatic", "axis": [round(float(v), 4) for v in sign * R[:, a]], "range": [0, round(d, 4)],
                            "from": 0, "schedule": sched("to", d)})
    return out


FINE = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15)     # 张角按 100° 的倍数，行程按部件沿轴尺寸的倍数


def _with_amount(m: dict, obj: dict, T: float, i: int) -> dict:
    """同一个关节换一档张角或行程；不动的部件原样。"""
    m = json.loads(json.dumps(m))
    if m.get("type") == "revolute":
        ang = round(100.0 * FINE[i], 1)
        m.update(range_deg=[0, ang], from_deg=0, schedule=[{"t": 0.5, "to_deg": ang, "duration": round(T - 1.0, 3)}])
        m.pop("rate_dps", None)
    elif m.get("type") == "prismatic":
        R = _quat_mat(obj["pose"].get("quat", [0, 0, 0, 1]))
        ext = np.asarray(obj["geom"]["extent"], float)
        along = float(np.abs(R.T @ np.asarray(m.get("axis", [1, 0, 0]), float)) @ ext)
        d = round(along * FINE[i] * 0.9, 4)
        m.update(range=[0, d], schedule=[{"t": 0.5, "to": d, "duration": round(T - 1.0, 3)}], **{"from": 0})
        m.pop("rate", None)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sets", default="")
    ap.add_argument("--pred", required=True, help="模型的预测（贪心），当第 0 个假设，也是打平时的默认")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    samples = load_samples(a.samples, a.sets.split(",") if a.sets else None)
    model = {}
    for line in Path(a.pred).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            model[r["id"]] = extract_motions(r.get("json")) or {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    work = os.path.join(os.environ.get("SLURM_TMPDIR", str(out)), "explore_work")   # 两万多次渲染的中间文件放节点本地盘，打完分就删
    jobs, hyps = [], {}
    for s in samples:
        prog = read_json(s["program"])
        T = float(s["duration"])
        per = {o["id"]: [model[s["id"]].get(o["id"], {"type": "static"})] + hypotheses(o, T) for o in prog["objects"]}
        hyps[s["id"]] = per
        K = len(next(iter(per.values())))
        for k in range(K):
            jobs.append((f"h{k:03d}", s, {pid: per[pid][k] for pid in per}, work))
    print(f"{len(samples)} inputs, {len(jobs)} renders, {a.workers} workers", flush=True)
    t0 = time.time()
    scores: dict[str, dict[str, list[float]]] = {}
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(eval_one, jobs, chunksize=4), 1):
            shutil.rmtree(row.get("evaluation_dir", ""), ignore_errors=True)
            k = int(row["source"][1:])
            for p in row.get("per_part", []):
                if p["iou_tT"] is not None:
                    scores.setdefault(row["id"], {}).setdefault(p["id"], {})[k] = p["iou_tT"]
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    # 第一轮：逐部件挑最好的假设（和模型自己的预测打平时留模型的）
    first = {}
    for s in samples:
        for pid, cands in hyps[s["id"]].items():
            sc = scores.get(s["id"], {}).get(pid)
            if sc:
                k = max(sc, key=lambda k: (sc[k] + (0.02 if k == 0 else 0.0)))
                first[(s["id"], pid)] = (cands[k], sc[k], sc.get(0, 0.0))
    # 第二轮：铰链和滑轨定下来之后，细扫张角和行程
    jobs2, refine = [], {}
    for s in samples:
        prog = read_json(s["program"])
        T = float(s["duration"])
        per = {}
        for o in prog["objects"]:
            m = first.get((s["id"], o["id"]), ({"type": "static"},))[0]
            per[o["id"]] = [_with_amount(m, o, T, i) for i in range(len(FINE))]
        refine[s["id"]] = per
        for i in range(len(FINE)):
            jobs2.append((f"r{i:03d}", s, {pid: per[pid][i] for pid in per}, work))
    scores2: dict[str, dict[str, dict[int, float]]] = {}
    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(eval_one, jobs2, chunksize=4):
            shutil.rmtree(row.get("evaluation_dir", ""), ignore_errors=True)
            i = int(row["source"][1:])
            for p in row.get("per_part", []):
                if p["iou_tT"] is not None:
                    scores2.setdefault(row["id"], {}).setdefault(p["id"], {})[i] = p["iou_tT"]
    print(f"  refine {len(jobs2)} renders, {time.time() - t0:.0f}s", flush=True)

    recs, gains = [], []
    for s in samples:
        chosen, base_sc, best_sc = {}, [], []
        for pid, cands in hyps[s["id"]].items():
            if (s["id"], pid) not in first:     # 末帧看不见这个部件：没法打分，留模型的
                chosen[pid] = cands[0]
                continue
            m, sc1, sc0 = first[(s["id"], pid)]
            sc2 = scores2.get(s["id"], {}).get(pid, {})
            i = max(sc2, key=sc2.get) if sc2 else None
            if i is not None and sc2[i] > sc1:
                m, sc1 = refine[s["id"]][pid][i], sc2[i]
            chosen[pid] = m
            base_sc.append(sc0)
            best_sc.append(sc1)
        gains.append((float(np.mean(base_sc)) if base_sc else None, float(np.mean(best_sc)) if best_sc else None))
        recs.append({"id": s["id"], "set": s["set"], "json": {"motions": chosen},
                     "text": json.dumps({"motions": chosen}, separators=(",", ":"), ensure_ascii=False),
                     "model_score": gains[-1][0], "search_score": gains[-1][1], "source": "explore"})
    (out / "searched.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    b = [g[0] for g in gains if g[0] is not None]
    c = [g[1] for g in gains if g[1] is not None]
    print(json.dumps({"inputs": len(recs), "model_score_mean": float(np.mean(b)), "search_score_mean": float(np.mean(c)),
                      "search_score_ge_0.7": float(np.mean([x >= 0.7 for x in c])), "wall_s": round(time.time() - t0)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
