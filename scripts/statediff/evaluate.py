#!/usr/bin/env python
"""用内核给预测打分：合回已知部分 → 校验 → 编译 → 渲染两端 id 通道 → 逐物体和真值掩码配对。

每条样本的指标（都以真值物体为分母，所以无效的程序自然拿零分）：
  valid       过 schema 与语义校验且编译渲染成功
  iou_t0/tT   首帧 / 末帧上，每个真值物体和配到的预测物体的掩码 IoU，没配到记 0
  recall50    首帧 IoU ≥ 0.5 的真值物体占比；prec50 是预测物体里 IoU ≥ 0.5 的占比
  motion_acc  配到的物体运动类型八分类对不对；moving_acc 只问动没动
  axis_err    配到且双方都有轴的物体，轴夹角（度，不分正负）
  count_err   |预测物体数 − 真值物体数|
两个预言机当标尺：gt 把真值原样送进去（必须全 1，不然量具坏了）；static 把真值的运动全改成静止
（首帧完美检测、末帧不动，量的是「运动这一半」有多少分可拿）。

用法:
  python scripts/statediff/evaluate.py --samples SET/samples.jsonl --sets holdout_in,holdout_out \
      --pred prompt_glm=PRED1.jsonl --pred lora=PRED2.jsonl --oracles gt,static --work WORK --out OUT
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.compiler.validate import validate                                            # noqa: E402
from gwm.config import load_config                                                   # noqa: E402
from scripts.statediff.common import (axis_angle_deg, axis_of, context_of, extract_objects,  # noqa: E402
                                      iou, load_samples, match_masks, merge_prediction,
                                      object_masks, read_json, render_endpoints, target_objects)

METRICS = ("iou_t0", "iou_tT", "recall50", "prec50", "motion_acc", "moving_acc")


def _zero_row(row: dict, reason: str, n_pred: int) -> dict:
    row.update(valid=0, reason=reason, n_pred=n_pred, count_err=abs(n_pred - row["n_gt"]),
               axis_err=None, n_axis=0, per_object=[], **{m: 0.0 for m in METRICS})
    return row


def eval_one(job: tuple) -> dict:
    source, sample, objects, work_root = job
    gt_prog = read_json(sample["program"])
    gt_objs = {o["id"]: o for o in gt_prog.get("objects") or []}
    duration = float(sample["duration"])
    row = {"id": sample["id"], "set": sample["set"], "source": source, "n_gt": len(gt_objs),
           "tags": sample.get("tags", [])}
    if objects is None:
        return _zero_row(row, "no_json", 0)
    work = Path(work_root) / source / sample["id"].replace("/", "__")
    cache = work / "metrics.json"
    if cache.exists():
        return read_json(cache)
    program = merge_prediction(context_of(gt_prog), objects)
    report = validate(program)
    if not report["ok"]:
        row = _zero_row(row, "invalid:" + ",".join(e.get("code", "?") for e in report["errors"][:3]), len(objects))
    else:
        res = render_endpoints(program, work, [0.0, duration], load_config())
        if not res["ok"]:
            row = _zero_row(row, res["stage"], len(objects))
        else:
            gt0 = object_masks(gt_prog, sample["id0"])
            gt1 = object_masks(gt_prog, sample["id1"])
            p0 = object_masks(program, res["files"][0.0]["id"])
            p1 = object_masks(program, res["files"][round(duration, 3)]["id"])
            pred_objs = {o["id"]: o for o in objects if isinstance(o.get("id"), str)}
            match = match_masks(gt0, p0)
            per, axis_errs = [], []
            used = set()
            for g, (p, iou0) in match.items():
                gm = (gt_objs[g].get("motion") or {}).get("type", "static")
                pm = None
                iou1 = 0.0
                if p is not None:
                    used.add(p)
                    iou1 = iou(gt1[g], p1[p])
                    pm = ((pred_objs.get(p) or {}).get("motion") or {}).get("type", "static")
                    ga, pa = axis_of(gt_objs[g].get("motion")), axis_of((pred_objs.get(p) or {}).get("motion"))
                    if ga is not None and pa is not None:
                        axis_errs.append(axis_angle_deg(ga, pa))
                per.append({"gt": g, "pred": p, "iou_t0": round(iou0, 4), "iou_tT": round(iou1, 4),
                            "gt_motion": gm, "pred_motion": pm})
            n = max(len(gt_objs), 1)
            row.update(valid=1, reason="", n_pred=len(objects), count_err=abs(len(objects) - len(gt_objs)),
                       iou_t0=sum(r["iou_t0"] for r in per) / n, iou_tT=sum(r["iou_tT"] for r in per) / n,
                       recall50=sum(r["iou_t0"] >= 0.5 for r in per) / n,
                       prec50=(sum(r["iou_t0"] >= 0.5 for r in per) / len(objects)) if objects else 0.0,
                       motion_acc=sum(r["pred_motion"] == r["gt_motion"] for r in per) / n,
                       moving_acc=sum(r["pred"] is not None and (r["pred_motion"] != "static") == (r["gt_motion"] != "static") for r in per) / n,
                       axis_err=(float(np.mean(axis_errs)) if axis_errs else None), n_axis=len(axis_errs),
                       time_sampling=res.get("time_sampling"), per_object=per)
    work.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return row


def load_predictions(path: str) -> dict[str, list | None]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = extract_objects(rec.get("json"))
    return out


def static_oracle(program: dict) -> list[dict]:
    objs = copy.deepcopy(target_objects(program))
    for o in objs:
        o["motion"] = {"type": "static"}
        o.pop("events", None)
    return objs


def summarise(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["source"], r["set"]), []).append(r)
    summary = {}
    for (source, set_name), rs in sorted(groups.items()):
        n = len(rs)
        ax = [r["axis_err"] for r in rs if r.get("axis_err") is not None]
        entry = {"n": n, "valid": sum(r["valid"] for r in rs) / n,
                 **{m: sum(r[m] for r in rs) / n for m in METRICS},
                 "count_err": sum(r["count_err"] for r in rs) / n,
                 "axis_err": (float(np.mean(ax)) if ax else None), "n_axis": int(sum(r["n_axis"] for r in rs)),
                 "reasons": {}}
        for r in rs:
            if r["reason"]:
                key = r["reason"].split(":")[0]
                entry["reasons"][key] = entry["reasons"].get(key, 0) + 1
        conf: dict[str, dict[str, int]] = {}
        for r in rs:
            for o in r["per_object"]:
                conf.setdefault(o["gt_motion"], {})
                k = o["pred_motion"] or "unmatched"
                conf[o["gt_motion"]][k] = conf[o["gt_motion"]].get(k, 0) + 1
        entry["motion_confusion"] = conf
        summary[f"{source} | {set_name}"] = entry
    return summary


def to_markdown(summary: dict) -> str:
    head = ("| 来源 | 子集 | n | 有效 | IoU t0 | IoU tT | 查全@.5 | 查准@.5 | 运动类型 | 动/不动 | 轴误差°(n) | 数目差 |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines = []
    for key, e in summary.items():
        source, set_name = key.split(" | ")
        ax = "—" if e["axis_err"] is None else f"{e['axis_err']:.1f} ({e['n_axis']})"
        lines.append(f"| {source} | {set_name} | {e['n']} | {e['valid']:.2f} | {e['iou_t0']:.3f} | {e['iou_tT']:.3f} | "
                     f"{e['recall50']:.2f} | {e['prec50']:.2f} | {e['motion_acc']:.2f} | {e['moving_acc']:.2f} | {ax} | {e['count_err']:.2f} |")
    return head + "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sets", default="holdout_in,holdout_out")
    ap.add_argument("--pred", action="append", default=[], help="name=predictions.jsonl，可重复")
    ap.add_argument("--oracles", default="gt,static")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-set-limit", type=int, default=0)
    a = ap.parse_args()

    samples = []
    for set_name in a.sets.split(","):
        rows = load_samples(a.samples, [set_name])
        samples += rows[: a.per_set_limit] if a.per_set_limit else rows
    jobs = []
    for name in [x for x in a.oracles.split(",") if x]:
        for s in samples:
            prog = read_json(s["program"])
            objs = target_objects(prog) if name == "gt" else static_oracle(prog)
            jobs.append((f"oracle_{name}", s, objs, a.work))
    for spec in a.pred:
        name, path = spec.split("=", 1)
        preds = load_predictions(path)
        for s in samples:
            jobs.append((name, s, preds.get(s["id"]), a.work))
    print(f"{len(jobs)} evaluations over {len(samples)} samples, {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(eval_one, jobs), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(summary)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    bad = [k for k, e in summary.items() if k.startswith("oracle_gt") and (e["iou_t0"] < 0.999 or e["iou_tT"] < 0.999)]
    if bad:
        print("量具自检没过：真值预言机不是满分 →", bad, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
