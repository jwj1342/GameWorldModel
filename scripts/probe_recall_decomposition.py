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

还有零假设必须一起报，否则「偏位」这一桶会骗人：**场景里到处都是节点的话，
随便哪个位置附近都有一个。** 报两个：

甲（场景间距）每个漏掉的物体到最近的**另一个真值节点**有多远。洞在于这个比较把
「我们预测了几个节点」也混了进来——预测得少，「最近的预测更远」就天然成立，
跟放没放对无关，而这个偏差恰好朝着否定教师的方向。

乙（密度匹配，**这个才是判据**）在真值场景的包围盒里均匀撒随机点，用**同一批**
预测节点量它们的最近距离。预测的个数和位置一点没动，密度被自动匹配掉。漏掉的
物体如果并不比随机点更靠近预测，那「附近有个预测节点」就说明不了任何事。

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
    objects = (official.get("full") or {}).get("objects") or {}
    criterion = objects.get("match_criterion") or {}
    want_pairs = criterion.get("by_mean_distance")
    want_recall = objects.get("recall")
    if want_pairs is None or want_recall is None:
        return {"ok": False, "why": "evaluate() 没报 match_criterion/recall，键路径对不上，"
                                    f"full.objects 的键是 {sorted(objects)[:12]}"}
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
    null_distances: list[float] = []
    ratios: list[float] = []
    taken = {q for _g, q, _d in pairs if q is not None}
    # 零假设，两个，因为第一个自己带着一个洞。
    #
    # 零假设甲（场景间距）：对每个漏掉的物体，量它到**最近的另一个真值节点**有多远。
    # 那是这个场景的固有间距。
    #   洞在这里：这个比较把「预测节点有多少个」也混了进去。我们预测的节点比真值
    #   少的话，「最近的预测更远」纯粹是密度造成的，跟放没放对无关——而这个偏差
    #   恰好是朝着「否定教师」的方向，所以不能只靠它下结论。
    #
    # 零假设乙（密度匹配，这个才是判据）：在真值场景的包围盒里**均匀撒随机点**，
    # 用同一批预测节点量它们的最近距离。预测节点的个数和位置完全不变，所以密度
    # 被自动匹配掉了。漏掉的物体如果并不比随机点更靠近预测，那就说明「附近有个
    # 预测节点」只是场景里到处都有节点，不是「这个物体被检出来了只是放错地方」。
    gt_static_tr = tracks(gt_states, kinds=("static",))
    gt_all = ([(k, v) for k, v in prefix(gt_tr).items()]
              + [(f"static::{k}", v) for k, v in prefix(gt_static_tr).items()])
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
        null = float("inf")
        for key, other in gt_all:
            if key == g:
                continue
            ts = _supported(pair_times if len(pair_times) else obs_times, gt_track, other)
            if not len(ts):
                continue
            d = float(np.linalg.norm(_sample(gt_track, ts) - _sample(other, ts), axis=1).mean())
            null = min(null, d)
        null_distances.append(null)
        if np.isfinite(best) and np.isfinite(null) and null > 1e-9:
            ratios.append(best / null)
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

    # —— 零假设乙：密度匹配的随机点 ——
    # 在真值场景的包围盒里均匀撒点，用**同一批**预测节点量最近距离。预测的个数和
    # 位置一点没动，所以密度被匹配掉了。种子写死，换次运行不该换结论。
    rand_near: list[float] = []
    ref_times = pair_times if len(pair_times) else obs_times
    cand_pts = []
    for _key, cand, _kind in candidates:
        ts = _supported(ref_times, cand)
        if len(ts):
            cand_pts.append(_sample(cand, ts).mean(0))
    gt_pts_all = []
    for _key, other in gt_all:
        ts = _supported(ref_times, other)
        if len(ts):
            gt_pts_all.append(_sample(other, ts).mean(0))
    if cand_pts and len(gt_pts_all) >= 2:
        C = np.vstack(cand_pts)
        G = np.vstack(gt_pts_all)
        lo, hi = G.min(0), G.max(0)
        rng = np.random.default_rng(20261023)
        draws = rng.uniform(lo, hi, size=(512, 3))
        rand_near = [float(np.linalg.norm(C - p, axis=1).min()) for p in draws]

    return {"ok": True, "checks": checks, "n_gt": len(gt_tr), "n_matched": len(matched),
            "n_missed": len(gt_tr) - len(matched), "buckets": buckets,
            "nearest_is_static": nearest_is_static,
            "missed_distances": [round(d, 3) for d in distances],
            "null_distances": [round(d, 3) for d in null_distances],
            "pred_over_null": [round(r, 3) for r in ratios],
            "random_point_nearest": [round(d, 3) for d in rand_near],
            "n_pred_objects": len(pred_tr), "n_pred_static": len(pred_static)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-from", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--harness", default=str(REPO / "scripts" / "node_harness.sh"))
    ap.add_argument("--gt-root", action="append", default=[],
                    help="额外的真值根目录。合成数据集产在 $SCRATCH 上，不在仓库里，必须指进来。")
    ap.add_argument("--max-dist", type=float, default=3.0)
    ap.add_argument("--gauge", default="rigid", choices=["rigid", "similarity"])
    ap.add_argument("--match-on", default="mean", choices=["mean", "start"])
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-samples", type=int, default=120)
    a = ap.parse_args(argv)

    runs = [Path(l.strip()) for l in Path(a.runs_from).read_text(encoding="utf-8").splitlines() if l.strip()]
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    print(f"容差 {a.max_dist} m，规范 {a.gauge}，配对判据 {a.match_on}，{len(runs)} 个运行目录")
    print(f"真值根目录：{a.gt_root or '（只有仓库内置的）'}\n")

    total = {"占用": 0, "偏位": 0, "缺席": 0}
    total_static = {"占用": 0, "偏位": 0, "缺席": 0}
    n_gt = n_matched = 0
    all_d: list[float] = []
    all_null: list[float] = []
    all_ratio: list[float] = []
    all_rand: list[float] = []
    failed: list[str] = []
    for run in runs:
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
        clip = manifest.get("clip", run.parent.name)
        gt_path, _link, _cands = find_ground_truth(clip, a.gt_root)
        if gt_path is None:
            failed.append(f"{clip}：没有真值程序（找过 {a.gt_root or '仓库内置的 examples/ 和 data/ground_truth/'}）"); continue
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
        try:
            res = decompose(gt, json.loads((run / "program.json").read_text(encoding="utf-8")),
                            gt_states, pred_states, a.holdout_frac, a.max_dist,
                            a.n_samples, a.match_on, a.gauge)
        except Exception as exc:                                   # noqa: BLE001
            # 一条片段算不出来不该把整个配置带走，但也不能悄悄吞掉——记下类型和位置。
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            failed.append(f"{clip}：归因崩了 {type(exc).__name__}: {exc} @ {tb[-2].strip() if len(tb) > 1 else '?'}")
            continue
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
        all_null += res["null_distances"]; all_ratio += res["pred_over_null"]
        all_rand += res["random_point_nearest"]

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
    fn = [d for d in all_null if np.isfinite(d)]
    fr = [r for r in all_ratio if np.isfinite(r)]
    if fn:
        print(f"  同一批物体到**最近的另一个真值节点**：中位 {statistics.median(fn):.2f} m"
              f"（场景的固有间距，这是零假设）")
    if fr:
        closer = sum(1 for r in fr if r < 1.0) / len(fr)
        print(f"\n  零假设甲（场景间距）比值（到最近预测 ÷ 到最近的另一个真值）："
              f"中位 {statistics.median(fr):.2f}，{closer:.1%} 的物体比值 < 1")
        print("    注意这个比较把「我们预测了几个节点」也混了进来，预测得少就天然偏大。")
    fd = [d for d in all_rand if np.isfinite(d)]
    if fd and all_d:
        finite = [d for d in all_d if np.isfinite(d)]
        if finite:
            m_obj, m_rand = statistics.median(finite), statistics.median(fd)
            print(f"\n  零假设乙（密度匹配，**这个才是判据**）：")
            print(f"    漏掉的物体  到最近预测 中位 {m_obj:.2f} m（{len(finite)} 个）")
            print(f"    场景里随机点 到最近预测 中位 {m_rand:.2f} m（{len(fd)} 个，同一批预测节点）")
            print(f"    比值 {m_obj / m_rand:.2f}" if m_rand > 1e-9 else "    随机点距离退化")
            print("    明显小于 1 → 漏掉的物体附近确实有东西，是检出来了放错地方，教师能修；")
            print("    接近或大于 1 → 「附近有预测节点」只是场景里到处都是节点，说明不了任何事。")
    print("""
判读：
  「偏位」占比高  → 东西检出来了只是放错地方，**稠密三维重建（教师）修的正是这一类**；
  「缺席」占比高  → 那个位置根本没有候选，教师给不了语义，这条路修不到；
  「占用」占比高  → 问题在一对一指派和碎片，和教师、和形状都无关。
提醒一：这三个桶用的是和召回同一个距离，所以它们和被测量天然相关。
        这里做的是把召回拆开，不是拿独立的量去验证它。
提醒二：**先看零假设乙再读「偏位」。** 场景里到处都是节点的话，随便哪个位置
        附近都有一个，「偏位」会被巧合撑起来。零假设乙不明显小于 1 的话，
        这一桶说明不了任何事。
提醒三：「缺席」里混着两类性质相反的东西——真的没检出来（教师修不了），
        和检出来了算到地下于是被管线扔掉（教师正好能修）。
        后者数在 `probe_ground_penetration.py` 的 dropped_sunk 里。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
