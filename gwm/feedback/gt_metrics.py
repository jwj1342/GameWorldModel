"""把预测出来的场景程序和真值程序作比较，给出可跨视频汇总的指标。

和 metrics.py 的区别是比较对象。metrics.py 拿渲染结果和感知结果比，衡量的是
「程序和感知吻合得怎么样」；感知本身有误差，而且有一条确定性基线直接把感知结果
抄进程序，在那套分数上天然接近满分。这里拿程序和真值比，衡量的是
「程序和视频里真实发生的事情吻合得怎么样」，这才是能写进论文的数字。

轨迹不在 Python 里重算。运动语义的唯一实现在 kernel/motions.js，重写一份会漂。
做法是两份程序都经 harness 跑一遍、导出逐帧状态，这里只消费状态。

指标范围刻意收窄到能撑论文主表的四项：运动类型准确率、轨迹误差、
物体召回与误检、以及留出时间段的外推。尺度误差、相机位姿、事件时序这些
等主表立住了再加。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_states(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def tracks(states: list[dict], kinds=("object",)) -> dict[str, dict]:
    """把逐帧状态整理成 {id: {t: [...], pos: Nx3, class: str}}。"""
    out: dict[str, dict] = {}
    for s in states:
        for o in s.get("objects", []):
            if kinds and o.get("kind") not in kinds: continue
            d = out.setdefault(o["id"], {"t": [], "pos": [], "class": o.get("class")})
            d["t"].append(s["t"]); d["pos"].append(o["pos"])
    for d in out.values():
        d["t"] = np.asarray(d["t"], float); d["pos"] = np.asarray(d["pos"], float)
    return out


def motion_types(program: dict) -> dict[str, str]:
    return {o["id"]: (o.get("motion") or {}).get("type", "static") for o in program.get("objects", [])}


def _sample(track: dict, times: np.ndarray) -> np.ndarray:
    """按时间线性插值取位置，两边程序的帧时刻不一定对齐。"""
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


def match_objects(gt: dict[str, dict], pred: dict[str, dict], times: np.ndarray, max_dist_m: float) -> list[tuple]:
    """按整段轨迹的平均距离做最优匹配，返回 [(gt_id, pred_id 或 None, 平均距离)]。

    超过 max_dist_m 的匹配不算数，两边都记成未匹配，避免把八竿子打不着的物体凑成一对。
    """
    gt_ids, pred_ids = sorted(gt), sorted(pred)
    if not gt_ids: return []
    if not pred_ids: return [(g, None, float("inf")) for g in gt_ids]

    cost = np.zeros((len(gt_ids), len(pred_ids)))
    for i, g in enumerate(gt_ids):
        a = _sample(gt[g], times)
        for j, p in enumerate(pred_ids):
            cost[i, j] = float(np.mean(np.linalg.norm(a - _sample(pred[p], times), axis=1)))

    rows, cols = linear_sum_assignment(cost)
    paired = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] <= max_dist_m}
    return [(g, pred_ids[paired[i]] if i in paired else None,
             cost[i, paired[i]] if i in paired else float("inf")) for i, g in enumerate(gt_ids)]


def evaluate(gt_program: dict, pred_program: dict, gt_states: list[dict], pred_states: list[dict],
             holdout_frac: float = 0.2, max_dist_m: float = 3.0, n_samples: int = 120) -> dict:
    """返回主表需要的四项指标，整段和留出段各算一遍。"""
    gt_tr, pred_tr = tracks(gt_states), tracks(pred_states)
    if not gt_tr:
        return {"error": "真值里没有可比的物体"}

    t_end = min(max(d["t"].max() for d in gt_tr.values()),
                max(d["t"].max() for d in pred_tr.values()) if pred_tr else 0.0)
    if t_end <= 0:
        return {"error": "两份状态没有重叠的时间段"}

    gt_motion, pred_motion = motion_types(gt_program), motion_types(pred_program)

    def window(t0: float, t1: float) -> dict:
        times = np.linspace(t0, t1, n_samples)

        # 先用整体质心做一次粗对齐，破开「匹配要先对齐、对齐要先匹配」这个循环。
        # 两边物体数一般不同，所以粗对齐只能比质心，不能逐点配对——
        # Kabsch 要求一一对应，直接 vstack 会维度不等。
        gt_c = np.vstack([_sample(d, times).mean(0) for d in gt_tr.values()])
        pred_c = (np.vstack([_sample(d, times).mean(0) for d in pred_tr.values()])
                  if pred_tr else gt_c)
        t0v = np.array([gt_c[:, 0].mean() - pred_c[:, 0].mean(), 0.0,
                        gt_c[:, 2].mean() - pred_c[:, 2].mean()])
        R0 = np.eye(3)      # 粗对齐只平移，偏航等匹配上之后再估
        pred_rough = {k: {**d, "pos": d["pos"] @ R0.T + t0v} for k, d in pred_tr.items()}

        pairs = match_objects(gt_tr, pred_rough, times, max_dist_m)
        matched_rough = [(g, p) for g, p, _ in pairs if p is not None]
        if matched_rough:
            gp = np.vstack([_sample(gt_tr[g], times) for g, _ in matched_rough])
            pp = np.vstack([_sample(pred_tr[p], times) for _, p in matched_rough])
            R0, t0v = align_gauge(gp, pp)
        pred_aligned = {k: {**d, "pos": d["pos"] @ R0.T + t0v} for k, d in pred_tr.items()}
        pairs = match_objects(gt_tr, pred_aligned, times, max_dist_m)
        matched = [(g, p, d) for g, p, d in pairs if p is not None]

        # 轨迹误差：逐物体算整段平均距离。
        # 只报两个汇总数：中位数抗少数崩掉的物体，但场景里只有一个物体出问题时
        # 它会归零（敏感性检查里实测到了），所以最大值也要报。
        # 别的统计量从 per_object 里都能算出来，不在这里重复。
        errs = sorted(d for _, _, d in matched)
        traj = {"median_m": float(np.median(errs)) if errs else None,
                "max_m": float(max(errs)) if errs else None,
                "n": len(errs),
                "per_object": {g: round(d, 4) for g, _, d in matched}}

        # 运动类型：只在匹配上的物体里算，没匹配上的属于召回问题，分开记
        hits = [(gt_motion.get(g, "static"), pred_motion.get(p, "static")) for g, p, _ in matched]
        acc = float(np.mean([a == b for a, b in hits])) if hits else None
        confusion: dict[str, dict[str, int]] = {}
        for a, b in hits: confusion.setdefault(a, {}).setdefault(b, 0); confusion[a][b] += 1

        n_gt, n_pred = len(gt_tr), len(pred_tr)
        return {
            "gauge_alignment": {"yaw_deg": round(float(np.degrees(np.arctan2(R0[0, 2], R0[0, 0]))), 2),
                                "translation_xz_m": [round(float(t0v[0]), 3), round(float(t0v[2]), 3)],
                                "note": "水平平移和偏航是视频观测不到的规范自由度，算指标前先对齐掉，"
                                        "否则是在罚一个确定不了的量。只做刚性对齐，相对结构错了仍会暴露。"},
            "trajectory": traj,
            "motion_type": {"accuracy": acc, "n": len(hits), "confusion": confusion},
            "objects": {"gt": n_gt, "pred": n_pred, "matched": len(matched),
                        "recall": len(matched) / n_gt if n_gt else None,
                        "precision": len(matched) / n_pred if n_pred else None},
            "window_s": [round(t0, 3), round(t1, 3)],
        }

    cut = t_end * (1.0 - holdout_frac)
    return {"full": window(0.0, t_end),
            "holdout": window(cut, t_end),
            "holdout_frac": holdout_frac,
            "note": "holdout 是视频末尾的一段。模型在生成时看不到这段，"
                    "所以它能区分「真的建模了动力学」和「只是对看过的画面做了拟合」。"}
