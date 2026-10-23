"""漏掉的真值物体，到底卡在哪一环——而「3DGS 当老师」只修得了其中一类。

为什么要有这个数。物体召回现在是 0.356，而 `probe_shape_headroom.py` 量到的上界是
0.949（把形状种类全毁掉也只掉五个点）。中间差着将近六十个点，**但没有人知道这六十
个点分布在哪些失败模式上**。这件事直接决定研究叙事：

  * 如果主要是「检出来了，但位置差太远」——那正是稠密三维重建（3DGS/深度）能给的，
    教师那条路有用；
  * 如果主要是「那个位置根本什么都没检出来」——教师给不了语义，这条路白走。

做法。不改程序、不喂预言机，只对**已有的评测结果**做一次归因：对每个没被配上的真值
物体，量它到**最近的那个预测节点**（object 和 static 一起看）有多远，按这个距离分桶。

  占用   最近的候选在容差以内，却还是没配上  → 一对一指派把它让给了别人，或者碎片
  偏位   最近的候选在 1~4 倍容差            → **教师修的就是这一类**
  缺席   最近的候选在 4 倍容差以外，或压根没有 → 检测/语义问题，教师修不了

口径要说在前头：这三个桶是用**和召回同一个距离**定义的，所以它们和被测量当然相关
——这里做的是把召回拆开，不是拿一个独立的量去验证它。

自查（两条，任何一条不过，下面的数一个都不能信）：
  1. 本脚本复现出来的配对数，必须等于 `evaluate()` 自己报的 `by_mean_distance`；
  2. 本脚本算的召回，必须等于 `evaluate()` 报的 `object_recall`。
复现不上就说明规范对齐没套对（这个坑踩过一次：探针漏了 align_world 的平移，
把「119 个簇对 128」读成了 0.93）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from gwm.feedback.gt_metrics import (                                  # noqa: E402
    align_gauge, camera_scale_hint, coarse_gauge, evaluate, match_objects, tracks,
)
from eval import find_ground_truth, states_of, work_dir                 # noqa: E402


def _sample(track: dict, times: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(times, track["t"], track["pos"][:, k]) for k in range(3)])


def _supported(ts: np.ndarray, *trs: dict) -> np.ndarray:
    lo = max([ts[0]] + [t["t"][0] for t in trs])
    hi = min([ts[-1]] + [t["t"][-1] for t in trs])
    return ts[(ts >= lo) & (ts <= hi)]


def decompose(gt_program: dict, pred_program: dict, gt_states: list[dict], pred_states: list[dict],
              holdout_frac: float, max_dist_m: float, n_samples: int,
              match_on: str, gauge: str) -> dict:
    """复刻 evaluate() 的规范对齐与配对，然后对没配上的真值物体做归因。"""
    official = evaluate(gt_program, pred_program, gt_states, pred_states,
                        holdout_frac=holdout_frac, max_dist_m=max_dist_m,
                        n_samples=n_samples, match_on=match_on, gauge=gauge)
    if "error" in official:
        return {"ok": False, "why": official["error"]}

    gt_tr, pred_tr = tracks(gt_states), tracks(pred_states)
    pred_static = tracks(pred_states, kinds=("static",))
    t_start, t_end = float(gt_states[0]["t"]), float(gt_states[-1]["t"])
    cut = t_start + (t_end - t_start) * (1 - holdout_frac)

    def prefix(source):
        return {k: {**v, "t": v["t"][v["t"] <= cut], "pos": v["pos"][v["t"] <= cut]}
                for k, v in source.items() if np.any(v["t"] <= cut)}

    observed_gt, observed_pred = prefix(gt_tr), prefix(pred_tr)
    obs_times = np.linspace(t_start, cut, n_samples)

    # —— 规范对齐，逐行照抄 evaluate()；差一步就是另一个坐标系 ——
    R0, t0v, s0 = np.eye(3), np.zeros(3), 1.0
    with_scale = gauge == "similarity"
    if observed_pred:
        def centroid(track):
            ts = _supported(obs_times, track)
            return _sample(track, ts).mean(0) if len(ts) else track["pos"].mean(0)
        gt_c = np.vstack([centroid(d) for d in observed_gt.values()])
        pred_c = np.vstack([centroid(d) for d in observed_pred.values()])
        hint = None
        if with_scale:
            hint, _ = camera_scale_hint([s for s in gt_states if t_start <= s["t"] <= cut],
                                        [s for s in pred_states if t_start <= s["t"] <= cut],
                                        gt_c, pred_c)
        R0, t0v, s0, _coarse = coarse_gauge(gt_c, pred_c, max_dist_m, with_scale,
                                            hint if hint and hint != 1.0 else None)
        rough = {k: {**d, "pos": s0 * (d["pos"] @ R0.T) + t0v} for k, d in observed_pred.items()}
        gp, pp = [], []
        for g, q, _ in match_objects(observed_gt, rough, obs_times, max_dist_m):
            if q is None:
                continue
            ts = _supported(obs_times, observed_gt[g], observed_pred[q])
            if len(ts):
                gp.append(_sample(observed_gt[g], ts)); pp.append(_sample(observed_pred[q], ts))
        if gp:
            R0, t0v, _s = align_gauge(np.vstack(gp), s0 * np.vstack(pp), with_scale=False)
    aligned = {k: {**d, "pos": s0 * (d["pos"] @ R0.T) + t0v} for k, d in pred_tr.items()}
    aligned_static = {k: {**d, "pos": s0 * (d["pos"] @ R0.T) + t0v} for k, d in pred_static.items()}

    pair_times = obs_times
    if match_on == "start" and len(obs_times):
        pair_times = np.asarray([obs_times[0]], float)
    try:
        pairs = match_objects(observed_gt, prefix(aligned), pair_times, max_dist_m)
    except ValueError:
        pairs = match_objects(observed_gt, prefix(aligned), obs_times, max_dist_m)
    pairs += [(g, None, float("inf")) for g in sorted(set(gt_tr) - set(observed_gt))]
    matched = {g for g, q, _ in pairs if q is not None}

    # —— 自查 ——
    want_pairs = official["matching"]["match_criterion"]["by_mean_distance"]
    want_recall = official["full"]["objects"]["object_recall"]
    got_recall = len(matched) / len(gt_tr) if gt_tr else 0.0
    checks = {"pairs_mine": len(matched), "pairs_official": want_pairs,
              "recall_mine": round(got_recall, 6), "recall_official": round(float(want_recall), 6)}
    checks["ok"] = (len(matched) == want_pairs
                    and abs(got_recall - float(want_recall)) < 1e-6)
    if not checks["ok"]:
        return {"ok": False, "why": "自查不通过，规范对齐没复现上", "checks": checks}

    # —— 归因：每个没配上的真值物体，到最近的预测节点有多远 ——
    candidates = ([(k, v, "object") for k, v in prefix(aligned).items()]
                  + [(k, v, "static") for k, v in prefix(aligned_static).items()])
    buckets: dict[str, int] = {"占用": 0, "偏位": 0, "缺席": 0}
    nearest_is_static = {"占用": 0, "偏位": 0, "缺席": 0}
    distances: list[float] = []
    taken = {q for _g, q, _d in pairs if q is not None}
    for g in sorted(set(gt_tr) - matched):
        gt_track = observed_gt.get(g) or gt_tr[g]
        best, best_kind, best_free = float("inf"), None, None
        for key, cand, kind in candidates:
            ts = _supported(pair_times if len(pair_times) else obs_times, gt_track, cand)
            if not len(ts):
                continue
            d = float(np.linalg.norm(_sample(gt_track, ts) - _sample(cand, ts), axis=1).mean())
            if d < best:
                best, best_kind, best_free = d, kind, key not in taken
        distances.append(best)
        if best <= max_dist_m:
            tag = "占用"
        elif best <= 4 * max_dist_m:
            tag = "偏位"
        else:
            tag = "缺席"
        buckets[tag] += 1
        if best_kind == "static":
            nearest_is_static[tag] += 1
        _ = best_free
    return {"ok": True, "checks": checks, "n_gt": len(gt_tr), "n_matched": len(matched),
            "n_missed": len(gt_tr) - len(matched), "buckets": buckets,
            "nearest_is_static": nearest_is_static,
            "missed_distances": [round(d, 3) for d in distances],
            "n_pred_objects": len(pred_tr), "n_pred_static": len(pred_static)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-from", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--harness", default=str(REPO / "scripts" / "node_harness.sh"))
    ap.add_argument("--max-dist", type=float, default=3.0)
    ap.add_argument("--gauge", default="rigid", choices=["rigid", "similarity"])
    ap.add_argument("--match-on", default="mean", choices=["mean", "start"])
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-samples", type=int, default=120)
    a = ap.parse_args(argv)

    runs = [Path(l.strip()) for l in Path(a.runs_from).read_text(encoding="utf-8").splitlines() if l.strip()]
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    print(f"容差 {a.max_dist} m，规范 {a.gauge}，配对判据 {a.match_on}，{len(runs)} 个运行目录\n")

    total = {"占用": 0, "偏位": 0, "缺席": 0}
    total_static = {"占用": 0, "偏位": 0, "缺席": 0}
    n_gt = n_matched = 0
    all_d: list[float] = []
    failed: list[str] = []
    for run in runs:
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
        clip = manifest.get("clip", run.parent.name)
        gt_path, _link, _cands = find_ground_truth(clip, [])
        if gt_path is None:
            failed.append(f"{clip}：没有真值程序"); continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        duration = float(gt.get("meta", {}).get("duration") or 8.0)
        import hashlib
        sha = hashlib.sha256(gt_path.read_bytes()).hexdigest()
        try:
            gt_states = states_of(gt_path, work_dir(work, f"{clip}_gt", sha), duration, a.harness)
            pred_states = states_of(run / "program.json",
                                    work_dir(work, f"{run.name}_pred", str(run.resolve())),
                                    duration, a.harness)
        except Exception as exc:                                   # noqa: BLE001
            failed.append(f"{clip}：录制失败 {exc}"); continue
        res = decompose(gt, json.loads((run / "program.json").read_text(encoding="utf-8")),
                        gt_states, pred_states, a.holdout_frac, a.max_dist,
                        a.n_samples, a.match_on, a.gauge)
        if not res.get("ok"):
            failed.append(f"{clip}：{res.get('why')} {res.get('checks', '')}"); continue
        b = res["buckets"]
        print(f"{clip:14s} 真值 {res['n_gt']:3d} 配上 {res['n_matched']:3d} "
              f"漏 {res['n_missed']:3d}  占用 {b['占用']:3d} 偏位 {b['偏位']:3d} 缺席 {b['缺席']:3d}"
              f"   （预测 object {res['n_pred_objects']}，static {res['n_pred_static']}）")
        for k in total:
            total[k] += b[k]; total_static[k] += res["nearest_is_static"][k]
        n_gt += res["n_gt"]; n_matched += res["n_matched"]
        all_d += res["missed_distances"]

    print("\n" + "=" * 72)
    if failed:
        print("没算成的：")
        for f in failed:
            print(f"  {f}")
        print()
    if not n_gt:
        print("一条都没算成，下面没有数。"); return 1
    missed = n_gt - n_matched
    print(f"真值物体 {n_gt} 个，配上 {n_matched}，召回 {n_matched / n_gt:.3f}，漏掉 {missed}\n")
    print("漏掉的那些，最近的预测节点在哪：")
    for k, label in (("占用", f"最近的候选在容差（{a.max_dist} m）以内，却没配上"),
                     ("偏位", f"最近的候选在 1~4 倍容差（{a.max_dist}~{4 * a.max_dist} m）"),
                     ("缺席", f"最近的候选在 4 倍容差（{4 * a.max_dist} m）以外")):
        share = total[k] / missed if missed else 0.0
        print(f"  {k}  {total[k]:3d} / {missed}  = {share:.3f}   {label}")
        print(f"        其中最近的那个是 static 节点：{total_static[k]}")
    if all_d:
        finite = [d for d in all_d if np.isfinite(d)]
        if finite:
            print(f"\n  漏掉物体到最近预测节点的距离：中位 {statistics.median(finite):.2f} m，"
                  f"四分位 {np.percentile(finite, 25):.2f} / {np.percentile(finite, 75):.2f} m")
    print("""
判读：
  「偏位」占比高  → 东西检出来了只是放错地方，**稠密三维重建（教师）修的正是这一类**；
  「缺席」占比高  → 那个位置根本没有候选，教师给不了语义，这条路修不到；
  「占用」占比高  → 问题在一对一指派和碎片，和教师、和形状都无关。
提醒：这三个桶用的是和召回同一个距离，所以它们和被测量天然相关。
      这里做的是把召回拆开，不是拿独立的量去验证它。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
