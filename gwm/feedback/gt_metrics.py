"""把预测出来的场景程序和真值程序作比较，给出可跨视频汇总的指标。

和 metrics.py 的区别是比较对象。metrics.py 拿渲染结果和感知结果比，衡量的是
「程序和感知吻合得怎么样」；感知本身有误差，而且有一条确定性基线直接把感知结果
抄进程序，在那套分数上天然接近满分。这里拿程序和真值比，衡量的是
「程序和视频里真实发生的事情吻合得怎么样」，这才是能写进论文的数字。

轨迹不在 Python 里重算。运动语义的唯一实现在 kernel/motions.js，重写一份会漂。
做法是两份程序都经 harness 跑一遍、导出逐帧状态，这里只消费状态。

指标范围刻意收窄到能撑论文主表的四项：运动类型准确率、轨迹误差、
物体召回与误检、以及末段诊断。没有输入隔离记录时不能称为外推。尺度误差、相机位姿、事件时序这些
等主表立住了再加。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from gwm.config import load_config


def load_states(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def tracks(states: list[dict], kinds=("object",)) -> dict[str, dict]:
    """Group instances by runtime name, retaining the parent id for motion labels."""
    out: dict[str, dict] = {}
    previous = None
    if not states:
        raise ValueError("missing state frames")
    for s in states:
        if not isinstance(s, dict):
            raise ValueError("state frame must be an object")
        t = s.get("t")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t):
            raise ValueError("state time must be finite")
        if previous is not None and t <= previous:
            raise ValueError("state times must be strictly increasing")
        previous = t
        if not isinstance(s.get("objects"), list):
            raise ValueError("state objects must be an array (empty is allowed)")
        seen = set()
        for o in s["objects"]:
            if not isinstance(o, dict):
                raise ValueError("state entry must be an object")
            if kinds and o.get("kind") not in kinds: continue
            identity = o.get("name") or o["id"]
            pos = np.asarray(o.get("pos"), float)
            if identity in seen or pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError("duplicate instance or invalid position")
            seen.add(identity)
            d = out.setdefault(identity, {"t": [], "pos": [], "class": o.get("class"), "object_id": o["id"]})
            if d["object_id"] != o["id"]:
                raise ValueError("instance changes parent object id")
            d["t"].append(t); d["pos"].append(pos)
    for d in out.values():
        d["t"] = np.asarray(d["t"], float); d["pos"] = np.asarray(d["pos"], float)
    return out


def motion_types(program: dict) -> dict[str, str]:
    return {o["id"]: (o.get("motion") or {}).get("type", "static") for o in program.get("objects", [])}


def _sample(track: dict, times: np.ndarray) -> np.ndarray:
    """Interpolate within measured support only; never clamp outside endpoints."""
    if not len(times) or times.min() < track["t"][0] or times.max() > track["t"][-1]:
        raise ValueError("requested times are outside track support")
    return np.column_stack([np.interp(times, track["t"], track["pos"][:, k]) for k in range(3)])


def align_gauge(gt_pts: np.ndarray, pred_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """估计把预测对齐到真值的水平平移加绕 y 轴旋转，返回 (R, t)。

    单目视频确定不了世界原点的水平位置，也确定不了整体朝向——那是规范自由度，
    任何一个选择都同样合法。感知那边 align_world 只把地面压到 y=0，
    x 和 z 留在重建自己的原点上（通常是首帧相机位置），所以重建结果相对真值
    会整体平移若干米。实测过一次是 8.3 米。

    拿这个去扣分是在罚一个视频里观测不到的量。SLAM 评测的 ATE 也是先对齐
    再算残差，同样的道理。这里只做刚性对齐（水平平移加偏航），
    不做逐物体对齐，所以物体之间的相对结构错了仍然会暴露出来。

    高度不对齐：地面已经被压到 y=0，y 方向是可观测的。
    尺度也不对齐：那由假设的相机高度定死，算可观测。
    """
    if len(gt_pts) < 2:
        return np.eye(3), (gt_pts.mean(0) - pred_pts.mean(0)) * [1, 0, 1] if len(gt_pts) else np.zeros(3)
    gc, pc = gt_pts.mean(0), pred_pts.mean(0)
    g, q = gt_pts - gc, pred_pts - pc
    # 只在水平面上求最佳偏航，用 Kabsch 的二维特例
    H = q[:, [0, 2]].T @ g[:, [0, 2]]
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R2 = Vt.T @ np.diag([1.0, d]) @ U.T
    R = np.eye(3); R[0, 0], R[0, 2], R[2, 0], R[2, 2] = R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]
    t = gc - R @ pc
    t[1] = 0.0                      # 高度不动，它是可观测的
    return R, t


_AXIS_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def coarse_gauge(gt_pts: np.ndarray, pred_pts: np.ndarray, max_dist_m: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """粗对齐：枚举四个轴向偏航，再拿单对物体的隐含平移去投票，取对上最多的那个。

    拿两边质心之差做粗对齐有两个毛病。一是多一个假物体、或者少检出一个真物体，
    质心就被拽走，原本对得上的也对不上了——完美感知下跑真实场景实测过一次，
    十二个本该进 static 的结构落进了 objects，质心被拽偏 1.4 米。二是整体转了
    九十度的时候，光靠平移永远凑不到阈值以内，一对都匹配不上，后面估偏航那步
    根本启动不了。

    单对物体的隐含平移不受别的物体影响，再按阈值内的个数投票，天然忽略掉少数
    离群的。只枚举四个轴向偏航是因为这一步只要够得着就行，精确的偏航在匹配
    建立之后由 align_gauge 解析求出。非轴向的大角度整体旋转仍可能启动失败。
    """
    best_R, best_t, best_in, best_cost = np.eye(3), np.zeros(3), -1, np.inf
    if not len(gt_pts) or not len(pred_pts):
        return best_R, best_t, {"yaw_deg": 0.0, "inliers": 0, "candidates": 0}
    for yaw in _AXIS_YAWS_DEG:
        R = _rot_y(yaw)
        rotated = pred_pts @ R.T
        for g in gt_pts:
            for q in rotated:
                t = g - q
                t[1] = 0.0                                  # 高度可观测，不参与对齐
                nearest = np.linalg.norm(gt_pts[:, None, :] - (rotated + t)[None, :, :], axis=2).min(1)
                inside = nearest <= max_dist_m
                inliers = int(inside.sum())
                cost = float(nearest[inside].sum()) if inliers else np.inf
                if (inliers, -cost) > (best_in, -best_cost):
                    best_R, best_t, best_in, best_cost = R, t.copy(), inliers, cost
    return best_R, best_t, {"yaw_deg": round(float(np.degrees(np.arctan2(best_R[0, 2], best_R[0, 0]))), 2),
                            "inliers": best_in, "candidates": len(_AXIS_YAWS_DEG) * len(gt_pts) * len(pred_pts)}


def match_objects(gt: dict[str, dict], pred: dict[str, dict], times: np.ndarray, max_dist_m: float) -> list[tuple]:
    """Match within available requested times, maximizing valid pairs before distance.

    超过 max_dist_m 的匹配不算数，两边都记成未匹配，避免把八竿子打不着的物体凑成一对。
    """
    gt_ids, pred_ids = sorted(gt), sorted(pred)
    if not gt_ids: return []
    if not pred_ids: return [(g, None, float("inf")) for g in gt_ids]

    if not np.isfinite(max_dist_m) or max_dist_m < 0:
        raise ValueError("max_dist_m must be finite and non-negative")
    cost = np.full((len(gt_ids), len(pred_ids)), np.inf)
    for i, g in enumerate(gt_ids):
        for j, p in enumerate(pred_ids):
            supported = times[(times >= max(gt[g]["t"][0], pred[p]["t"][0]))
                              & (times <= min(gt[g]["t"][-1], pred[p]["t"][-1]))]
            if len(supported):
                cost[i, j] = float(np.mean(np.linalg.norm(_sample(gt[g], supported) - _sample(pred[p], supported), axis=1)))

    # Dummy columns represent unmatched GTs. Normalized feasible distances are
    # <=1, so one extra valid pair dominates any possible distance reduction.
    penalty = min(len(gt_ids), len(pred_ids)) + 1
    feasible = cost <= max_dist_m
    assignment = np.full((len(gt_ids), len(pred_ids) + len(gt_ids)), float(penalty))
    assignment[:, :len(pred_ids)] = np.where(feasible, cost / max(max_dist_m, 1e-12), penalty * 2)
    rows, cols = linear_sum_assignment(assignment)
    paired = {int(r): int(c) for r, c in zip(rows, cols) if c < len(pred_ids) and feasible[r, c]}
    return [(g, pred_ids[paired[i]] if i in paired else None,
             cost[i, paired[i]] if i in paired else float("inf")) for i, g in enumerate(gt_ids)]


def evaluate(gt_program: dict, pred_program: dict, gt_states: list[dict], pred_states: list[dict],
             holdout_frac: float = 0.2, max_dist_m: float = 3.0, n_samples: int = 120,
             min_time_coverage: float | None = None) -> dict:
    """Keep existing metric keys, using a GT window and frozen prefix identities."""
    if min_time_coverage is None:
        min_time_coverage = load_config().get("gt_evaluation", {}).get("min_time_coverage", 0.9)
    if (isinstance(min_time_coverage, bool) or not isinstance(min_time_coverage, (int, float))
            or not np.isfinite(min_time_coverage) or not 0 <= min_time_coverage <= 1):
        raise ValueError("min_time_coverage must be finite and within [0, 1]")
    if not 0 < holdout_frac < 1 or n_samples < 2 or not np.isfinite(max_dist_m) or max_dist_m < 0:
        raise ValueError("invalid metric configuration")
    try:
        gt_tr, pred_tr = tracks(gt_states), tracks(pred_states)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc), "status": "invalid_states"}
    if not gt_tr:
        return {"error": "真值里没有可比的物体"}

    gt_motion, pred_motion = motion_types(gt_program), motion_types(pred_program)
    for side, source, labels in (("gt", gt_tr, gt_motion), ("pred", pred_tr, pred_motion)):
        unknown = sorted({track["object_id"] for track in source.values()} - labels.keys())
        if unknown:
            return {"error": f"{side} state objects are absent from its Program: {unknown}",
                    "status": "state_program_mismatch"}

    t_start, t_end = float(gt_states[0]["t"]), float(gt_states[-1]["t"])
    if t_end <= t_start:
        return {"error": "reference needs a positive time span", "status": "invalid_states"}
    cut = t_start + (t_end - t_start) * (1 - holdout_frac)

    # Truncate actual source samples, not just query times: interpolating at cut
    # through a later sample would leak the tail into the identity assignment.
    def prefix(source):
        return {key: {**track, "t": track["t"][track["t"] <= cut], "pos": track["pos"][track["t"] <= cut]}
                for key, track in source.items() if np.any(track["t"] <= cut)}

    observed_gt, observed_pred = prefix(gt_tr), prefix(pred_tr)
    obs_times = np.linspace(t_start, cut, n_samples)

    def _supported(ts, *tracks):
        """取这几条轨迹共同覆盖得到的查询时刻；_sample 不许外推，所以要先裁。"""
        lo = max([ts[0]] + [t["t"][0] for t in tracks])
        hi = min([ts[-1]] + [t["t"][-1] for t in tracks])
        return ts[(ts >= lo) & (ts <= hi)]

    # 规范对齐和身份对应必须定在同一段上。水平平移与偏航是单目视频观测不到的
    # 规范自由度，不先对齐掉，整体差着 8 米时一对都配不上；而配对已经冻在观察段，
    # 对齐就不能再留在 window() 里按窗口各算各的——那样留出段会重新拟合，
    # 把末段的共同偏移当成规范差异吸收掉。
    R0, t0v, coarse = np.eye(3), np.zeros(3), {"inliers": 0, "candidates": 0, "yaw_deg": 0.0}
    if observed_pred:
        def centroid(track):
            ts = _supported(obs_times, track)
            return _sample(track, ts).mean(0) if len(ts) else track["pos"].mean(0)
        gt_c = np.vstack([centroid(d) for d in observed_gt.values()])
        pred_c = np.vstack([centroid(d) for d in observed_pred.values()])
        R0, t0v, coarse = coarse_gauge(gt_c, pred_c, max_dist_m)
        rough = {k: {**d, "pos": d["pos"] @ R0.T + t0v} for k, d in observed_pred.items()}
        gp, pp = [], []
        for g, q, _ in match_objects(observed_gt, rough, obs_times, max_dist_m):
            if q is None: continue
            ts = _supported(obs_times, observed_gt[g], observed_pred[q])
            if len(ts):
                gp.append(_sample(observed_gt[g], ts)); pp.append(_sample(observed_pred[q], ts))
        if gp:
            R0, t0v = align_gauge(np.vstack(gp), np.vstack(pp))
    aligned = {k: {**d, "pos": d["pos"] @ R0.T + t0v} for k, d in pred_tr.items()}

    pairs = match_objects(observed_gt, prefix(aligned), obs_times, max_dist_m)
    pairs += [(g, None, float("inf")) for g in sorted(set(gt_tr) - set(observed_gt))]
    matched = [(g, p, d) for g, p, d in pairs if p is not None]

    def window(t0: float, t1: float) -> dict:
        distances, unavailable, coverage, evaluated_windows = {}, {}, {}, {}
        for g, p, _ in pairs:
            if p is None:
                coverage[g] = None  # No identity pair, not measured zero coverage.
                evaluated_windows[g] = None
                unavailable[g] = "not_observed_in_prefix" if g not in observed_gt else "no_prefix_match"
            else:
                start = max(t0, gt_tr[g]["t"][0], pred_tr[p]["t"][0])
                end = min(t1, gt_tr[g]["t"][-1], pred_tr[p]["t"][-1])
                coverage[g] = float(max(0.0, end - start) / (t1 - t0))
                evaluated_windows[g] = [float(start), float(end)] if end > start else None
                if end <= start or coverage[g] < min_time_coverage:
                    unavailable[g] = "insufficient_time_coverage"
                    continue
                times = np.linspace(start, end, n_samples)
                distances[g] = float(np.mean(np.linalg.norm(_sample(gt_tr[g], times) - _sample(aligned[p], times), axis=1)))

        # 轨迹误差：逐物体算整段平均距离。
        # 只报两个汇总数：中位数抗少数崩掉的物体，但场景里只有一个物体出问题时
        # 它会归零（敏感性检查里实测到了），所以最大值也要报。
        # 别的统计量从 per_object 里都能算出来，不在这里重复。
        errs = sorted(distances.values())
        traj = {"median_m": float(np.median(errs)) if errs else None,
                "max_m": float(max(errs)) if errs else None,
                "n": len(errs),
                "per_object": {g: round(d, 4) for g, d in distances.items()},
                "unavailable": unavailable,
                "coverage": coverage,
                "evaluated_window_s": evaluated_windows,
                "min_time_coverage": min_time_coverage,
                "status": "complete" if len(distances) == len(gt_tr) and all(c == 1 for c in coverage.values())
                          else "partial" if distances else "unavailable"}

        # 运动类型：只在匹配上的物体里算，没匹配上的属于召回问题，分开记
        hits = [(gt_motion[gt_tr[g]["object_id"]], pred_motion[pred_tr[p]["object_id"]]) for g, p, _ in matched]
        acc = float(np.mean([a == b for a, b in hits])) if hits else None
        confusion: dict[str, dict[str, int]] = {}
        for a, b in hits: confusion.setdefault(a, {}).setdefault(b, 0); confusion[a][b] += 1

        n_gt, n_pred = len(gt_tr), len(pred_tr)
        return {
            "gauge_alignment": {"yaw_deg": round(float(np.degrees(np.arctan2(R0[0, 2], R0[0, 0]))), 2),
                                "translation_xz_m": [round(float(t0v[0]), 3), round(float(t0v[2]), 3)],
                                "fitted_on_s": [round(t_start, 3), round(cut, 3)], "coarse": coarse,
                                "note": "水平平移和偏航是视频观测不到的规范自由度，算指标前先对齐掉，"
                                        "否则是在罚一个确定不了的量。变换与身份对应都在观察段定死后冻住，"
                                        "每个窗口共用同一个，留出段不重新拟合。"
                                        "只做刚性对齐，相对结构错了仍会暴露。"},
            "trajectory": traj,
            "motion_type": {"accuracy": acc, "n": len(hits), "confusion": confusion},
            "objects": {"gt": n_gt, "pred": n_pred, "matched": len(matched),
                        "recall": len(matched) / n_gt if n_gt else None,
                        "precision": len(matched) / n_pred if n_pred else None},
            "window_s": [round(t0, 3), round(t1, 3)],
        }

    return {"full": window(t_start, t_end),
            "holdout": window(cut, t_end),
            "holdout_frac": holdout_frac,
            "matching": {"window_s": [t_start, cut], "pairs": {g: p for g, p, _ in pairs}},
            "note": "holdout仅为参考末段诊断；对应关系只用前段样本并冻结。"
                    "未提供重建输入隔离证明，不宣称留出外推。位置误差不覆盖旋转、相机或物理发散协议。"}
