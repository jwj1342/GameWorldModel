#!/usr/bin/env python
"""铰接探针的评测：预测的 motion 换进真值程序 → 校验 → 编译 → 渲染两端 id 通道 → 逐部件比。

部件的 id 是给定的，不用配对。每条样本：
  valid          过校验且编译渲染成功
  type_acc       部件的关节类型（static / revolute / prismatic）对不对，部件平均
  moving_ok      动了的部件集合和真值完全一致
  hinge_ok       真值是铰链的部件：预测也是铰链，轴夹角 < 20°，且预测的铰链线离真值铰链中点 < 0.15 倍部件尺寸
  slide_ok       真值是滑轨的部件：预测也是滑轨，方向夹角 < 20°，末端位移误差 < 0.3 倍真值行程
  iou_move_tT    真值会动的部件在末帧的掩码 IoU 均值（执行出来的结果，方向开反、开错部件都会在这里掉）
  iou_still_tT   真值不动的部件在末帧的 IoU（动了不该动的会在这里掉）
  success        末帧每个看得见的部件 IoU ≥ 0.7
连续量另记：axis_err（铰链不分正负，滑轨分）、pivot_err、rot_err（末态转动差的角度，和轴的符号写法无关）、travel_err。
两个预言机当标尺：gt 原样送进去必须全对；static 全部不动，给出 iou_move_tT 的下限。

用法:
  python scripts/statediff/evaluate_art.py --samples SET/samples.jsonl --sets holdout_in,holdout_out \
      --pred zs8b=PRED.jsonl --oracles gt,static --work WORK --out OUT --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.compiler.validate import validate                                              # noqa: E402
from gwm.config import load_config                                                     # noqa: E402
from scripts.statediff.art import extract_motions, merge_motions                       # noqa: E402
from scripts.statediff.common import iou, load_samples, object_masks, read_json, render_endpoints  # noqa: E402

RATES = ("type_acc", "moving_ok", "hinge_ok", "slide_ok", "iou_move_tT", "iou_still_tT", "success")
CONT = ("axis_err", "pivot_err", "rot_err", "travel_err")


def _schedule_value(schedule, start, t, key):
    value = prev = start
    for s in schedule:
        d = s.get("duration", 1.0)
        if t < s["t"]:
            return value
        if t < s["t"] + d:
            return prev + (s[key] - prev) * ((t - s["t"]) / d)
        value = prev = s[key]
    return value


def angle_at(m: dict, t: float) -> float:
    start = m.get("from_deg", (m.get("range_deg") or [0])[0])
    if m.get("schedule"):
        return _schedule_value(m["schedule"], start, t, "to_deg")
    return start + m.get("rate_dps", 0.0) * t


def dist_at(m: dict, t: float) -> float:
    start = m.get("from", 0.0)
    if m.get("schedule"):
        return _schedule_value(m["schedule"], start, t, "to")
    return start + m.get("rate", 0.0) * t


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _rot(axis, deg) -> np.ndarray:
    k = _unit(axis)
    a = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _ang(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(_unit(a), _unit(b))), -1.0, 1.0))))


def joint_errors(g: dict, p: dict, extent, T: float) -> dict:
    """同类型时的连续误差。g/p 是真值与预测的 motion。"""
    gt_type, pr_type = g.get("type", "static"), p.get("type", "static")
    out = {"hinge_ok": None, "slide_ok": None}
    if gt_type == "revolute":
        out["hinge_ok"] = 0.0
        if pr_type == "revolute":
            ax = _ang(g["axis"], p.get("axis", [0, 1, 0]))
            ax = min(ax, 180 - ax)
            d = _unit(p.get("axis", [0, 1, 0]))
            rel = np.asarray(g["pivot"], float) - np.asarray(p.get("pivot", [0, 0, 0]), float)
            piv = float(np.linalg.norm(np.cross(rel, d))) / max(extent)
            Rg, Rp = _rot(g["axis"], angle_at(g, T)), _rot(p.get("axis", [0, 1, 0]), angle_at(p, T))
            rot = float(np.degrees(np.arccos(np.clip((np.trace(Rp.T @ Rg) - 1) / 2, -1.0, 1.0))))
            out.update(axis_err=ax, pivot_err=piv, rot_err=rot, hinge_ok=float(ax < 20 and piv < 0.15))
    if gt_type == "prismatic":
        out["slide_ok"] = 0.0
        if pr_type == "prismatic":
            ax = _ang(g["axis"], p.get("axis", [1, 0, 0]))
            dg = _unit(g["axis"]) * dist_at(g, T)
            dp = _unit(p.get("axis", [1, 0, 0])) * dist_at(p, T)
            tr = float(np.linalg.norm(dp - dg) / np.linalg.norm(dg))
            out.update(axis_err=ax, travel_err=tr, slide_ok=float(ax < 20 and tr < 0.3))
    return out


def _empty(row: dict, reason: str, gt_prog: dict) -> dict:
    moving = [o for o in gt_prog["objects"] if (o.get("motion") or {}).get("type", "static") != "static"]
    row.update(valid=0, reason=reason, type_acc=0.0, moving_ok=0.0, success=0.0,
               hinge_ok=0.0 if any(o["motion"]["type"] == "revolute" for o in moving) else None,
               slide_ok=0.0 if any(o["motion"]["type"] == "prismatic" for o in moving) else None,
               iou_move_tT=0.0, iou_still_tT=0.0 if len(moving) < len(gt_prog["objects"]) else None,
               per_part=[], **{c: [] for c in CONT})
    return row


def eval_one(job: tuple) -> dict:
    source, sample, motions, work_root = job
    gt_prog = read_json(sample["program"])
    T = float(sample["duration"])
    row = {"id": sample["id"], "set": sample["set"], "family": sample["family"], "source": source,
           "tags": sample.get("tags", [])}
    if motions is None:
        return _empty(row, "no_json", gt_prog)
    program, missing = merge_motions(gt_prog, motions)
    row["missing"] = missing
    report = validate(program)
    if not report["ok"]:
        return _empty(row, "invalid:" + ",".join(e.get("code", "?") for e in report["errors"][:3]), gt_prog)
    work = Path(work_root) / source / sample["id"].replace("/", "__") / ("attempt-" + uuid.uuid4().hex)
    res = render_endpoints(program, work, [0.0, T], load_config())
    if not res["ok"]:
        return _empty(row, res["stage"], gt_prog)
    gt0, gt1 = object_masks(gt_prog, sample["id0"]), object_masks(gt_prog, sample["id1"])
    p0, p1 = object_masks(program, res["files"][0.0]["id"]), object_masks(program, res["files"][round(T, 3)]["id"])
    per, cont = [], {c: [] for c in CONT}
    for g_obj, p_obj in zip(gt_prog["objects"], program["objects"]):
        pid = g_obj["id"]
        g, p = g_obj.get("motion") or {"type": "static"}, p_obj.get("motion") or {"type": "static"}
        errs = joint_errors(g, p, g_obj["geom"]["extent"], T)
        for c in CONT:
            if c in errs:
                cont[c].append(errs[c])
        per.append({"id": pid, "gt": g.get("type", "static"), "pred": p.get("type", "static"),
                    "iou_t0": iou(gt0[pid], p0[pid]) if gt0[pid].any() else None,
                    "iou_tT": iou(gt1[pid], p1[pid]) if gt1[pid].any() else None,
                    "hinge_ok": errs["hinge_ok"], "slide_ok": errs["slide_ok"],
                    **{c: errs[c] for c in CONT if c in errs}})
    mv = [r for r in per if r["gt"] != "static"]
    st = [r for r in per if r["gt"] == "static"]
    mean = lambda xs: float(np.mean(xs)) if xs else None
    row.update(valid=1, reason="",
               type_acc=mean([r["gt"] == r["pred"] for r in per]),
               moving_ok=float({r["id"] for r in mv} == {r["id"] for r in per if r["pred"] != "static"}),
               hinge_ok=mean([r["hinge_ok"] for r in per if r["hinge_ok"] is not None]),
               slide_ok=mean([r["slide_ok"] for r in per if r["slide_ok"] is not None]),
               iou_move_tT=mean([r["iou_tT"] for r in mv if r["iou_tT"] is not None]),
               iou_still_tT=mean([r["iou_tT"] for r in st if r["iou_tT"] is not None]),
               success=float(all(r["iou_tT"] >= 0.7 for r in per if r["iou_tT"] is not None)),
               per_part=per, evaluation_dir=str(work), **cont)
    return row


def load_predictions(path: str) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = extract_motions(rec.get("json"))
    return out


def summarise(rows: list[dict], by: str) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["source"], r[by]), []).append(r)
    out = {}
    for (source, key), rs in sorted(groups.items()):
        e = {"n": len(rs), "valid": float(np.mean([r["valid"] for r in rs]))}
        for m in RATES:
            vals = [r[m] for r in rs if r.get(m) is not None]
            e[m] = float(np.mean(vals)) if vals else None
        for c in CONT:
            vals = [v for r in rs for v in r.get(c, [])]
            e[c] = float(np.median(vals)) if vals else None
            e["n_" + c] = len(vals)
        conf: dict[str, dict[str, int]] = {}
        for r in rs:
            for p in r["per_part"]:
                conf.setdefault(p["gt"], {})
                conf[p["gt"]][p["pred"]] = conf[p["gt"]].get(p["pred"], 0) + 1
        e["type_confusion"] = conf
        e["reasons"] = {}
        for r in rs:
            if r["reason"]:
                k = r["reason"].split(":")[0]
                e["reasons"][k] = e["reasons"].get(k, 0) + 1
        out[f"{source} | {key}"] = e
    return out


def to_markdown(summary: dict) -> str:
    head = ("| 来源 | 分组 | n | 有效 | 类型 | 动的集合 | 铰链对 | 滑轨对 | 末帧IoU·动 | 末帧IoU·静 | 成功 | 轴误差° | 铰链偏/尺寸 | 末态转差° | 行程误差 |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    f = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    lines = []
    for key, e in summary.items():
        source, group = key.split(" | ")
        lines.append(f"| {source} | {group} | {e['n']} | {f(e['valid'])} | {f(e['type_acc'])} | {f(e['moving_ok'])} | "
                     f"{f(e['hinge_ok'])} | {f(e['slide_ok'])} | {f(e['iou_move_tT'], 3)} | {f(e['iou_still_tT'], 3)} | "
                     f"{f(e['success'])} | {f(e['axis_err'], 1)} | {f(e['pivot_err'])} | {f(e['rot_err'], 1)} | {f(e['travel_err'])} |")
    return head + "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sets", default="holdout_in,holdout_out")
    ap.add_argument("--pred", action="append", default=[], help="name=predictions.jsonl，可重复")
    ap.add_argument("--oracles", default="gt,static")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
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
            motions = {o["id"]: (o["motion"] if name == "gt" else {"type": "static"}) for o in prog["objects"]}
            jobs.append((f"oracle_{name}", s, motions, a.work))
    for spec in a.pred:
        name, path = spec.split("=", 1)
        preds = load_predictions(path)
        jobs += [(name, s, preds.get(s["id"]), a.work) for s in samples]
    print(f"{len(jobs)} evaluations over {len(samples)} samples, {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(eval_one, jobs), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by_set, by_family = summarise(rows, "set"), summarise(rows, "family")
    (out / "summary.json").write_text(json.dumps({"by_set": by_set, "by_family": by_family}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    md = "## 按子集\n\n" + to_markdown(by_set) + "\n## 按家族\n\n" + to_markdown(by_family)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    bad = [k for k, e in by_set.items() if k.startswith("oracle_gt") and (e["success"] < 0.999 or e["hinge_ok"] not in (None, 1.0)
                                                                      or e["slide_ok"] not in (None, 1.0))]
    if bad:
        print("量具自检没过：真值预言机不是全对 →", bad, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
