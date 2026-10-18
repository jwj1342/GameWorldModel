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


def class_names(program: dict) -> dict[str, str]:
    return {o["id"]: str(o.get("class", "")).strip().lower() for o in program.get("objects", [])}


def node_extent(node: dict) -> np.ndarray | None:
    """节点的长宽高，不管它是基本体还是资产。

    真值程序用的是基本体（sphere/box/cylinder/cone），我们产出的是带 extent 的
    资产节点，两边的形状描述方式不一样，所以统一折成三个边长再比。
    """
    g = node.get("geom") or {}
    shape = g.get("shape")
    if shape == "sphere":
        d = 2.0 * float(g.get("radius", 0.0))
        return np.array([d, d, d]) if d > 0 else None
    if shape in ("cylinder", "cone"):
        d = 2.0 * float(g.get("radius", 0.0))
        h = float(g.get("height", d))
        return np.array([d, h, d]) if d > 0 and h > 0 else None
    ext = g.get("extent")
    if isinstance(ext, list) and len(ext) == 3:
        e = np.asarray(ext, float)
        return e if np.all(e > 0) else None
    return None


def shape_scale_agreement(gt_node: dict, pred_node: dict) -> tuple[float, float] | None:
    """(形状差, 尺度比)。形状差 0 是完美，尺度比 1 是完美。

    形状差 = 把两边的边长各自归一化成「占最长边的比例」再算平均绝对差。这样它
    **对整体缩放不敏感**，只问长宽高的比例对不对——一个细长的东西被重建成细长的，
    哪怕大小差一倍，形状这一项也是对的。尺度单独报。

    为什么不比类名：这个合成数据集的类名和形状是**脱钩的**（生成器从词表挑名字、
    再独立配一个基本体，`girder` 是个球、`coin` 是个盒子、`crate` 用过三种形状）。
    所以类名准确率在这里量的是一个原理上测不出来的量，而形状和尺寸是有视觉意义的。
    """
    a, b = node_extent(gt_node), node_extent(pred_node)
    if a is None or b is None:
        return None
    ga = np.sort(a)[::-1] / a.max()
    pa = np.sort(b)[::-1] / b.max()
    return float(np.abs(ga - pa).mean()), float(b.max() / a.max())


def _sample(track: dict, times: np.ndarray) -> np.ndarray:
    """Interpolate within measured support only; never clamp outside endpoints."""
    if not len(times) or times.min() < track["t"][0] or times.max() > track["t"][-1]:
        raise ValueError("requested times are outside track support")
    return np.column_stack([np.interp(times, track["t"], track["pos"][:, k]) for k in range(3)])


def _rms_radius(pts: np.ndarray) -> float:
    return float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean())) if len(pts) >= 2 else 0.0


def gauge_scale_hint(gt_pts: np.ndarray, pred_pts: np.ndarray) -> float:
    """不靠任何对应关系估一个整体尺度：两边点集相对各自质心的均方半径之比。

    粗对齐那一步还没有任何配对，而尺度差上三倍的时候，按绝对米制阈值投票一对都投
    不出来——阈值不是尺度不变的。均方半径只用到点集自身的铺开程度，不需要知道谁对
    应谁。

    **但拿物体质心来算是不行的，实测过。** 这个链路的过检很重（112 个预测物体里
    只有 21.4% 配上真值），预测那一边的点集大部分是噪声，它的铺开程度跟真实场景
    没什么关系。用它去估尺度，解出来的倍数是 1.00、2.03、2.27、8.26、20.15 这种
    数，比不估还差：配对 24 掉到 19，召回 0.464 掉到 0.226。
    所以这个函数只是兜底，优先用 `camera_scale_hint`。
    """
    g, q = _rms_radius(gt_pts), _rms_radius(pred_pts)
    return g / q if q > 1e-9 and g > 0 else 1.0


def camera_scale_hint(gt_states: list[dict], pred_states: list[dict],
                      gt_pts: np.ndarray | None = None,
                      pred_pts: np.ndarray | None = None) -> tuple[float, dict]:
    """用**相机轨迹**估尺度，返回 (倍数, 诊断)。

    相机是这条链路上唯一天然一一对应的东西：真值和预测各只有一条相机轨迹，不存在
    检错、漏检、配错的问题。而物体那一边七成八是噪声，用它估尺度只会把噪声放大
    （见 `gauge_scale_hint` 的注释）。

    用均方半径之比而不是路径长度：路径长度对逐帧抖动敏感，重建出来的相机轨迹抖动
    比真值大得多，会把尺度系统性地估大。

    **相机不怎么动的时候这个估计不成立**，必须退回去。判据是无量纲的：相机轨迹的
    均方半径要和物体点云的铺开程度可比，低于十分之一就认为相机基本是钉住的，这时
    返回 1.0 并在诊断里说明——宁可不估，也不要用一个噪声主导的倍数去缩放整个场景。
    """
    def cam(states):
        out = []
        for s in states:
            c = (s.get("camera") or {}).get("pos") if isinstance(s, dict) else None
            if isinstance(c, (list, tuple)) and len(c) == 3 and np.isfinite(np.asarray(c, float)).all():
                out.append([float(v) for v in c])
        return np.asarray(out, float).reshape(-1, 3)

    g_cam, p_cam = cam(gt_states), cam(pred_states)
    info = {"gt_frames": len(g_cam), "pred_frames": len(p_cam)}
    if len(g_cam) < 2 or len(p_cam) < 2:
        return 1.0, {**info, "used": "none", "why": "相机轨迹点不够"}
    g_r, p_r = _rms_radius(g_cam), _rms_radius(p_cam)
    info.update({"gt_radius": round(g_r, 4), "pred_radius": round(p_r, 4)})
    for pts, radius, side in ((gt_pts, g_r, "gt"), (pred_pts, p_r, "pred")):
        spread = _rms_radius(pts) if pts is not None and len(pts) else 0.0
        if spread > 0 and radius < 0.1 * spread:
            return 1.0, {**info, "used": "none",
                         "why": f"{side} 相机基本没动（轨迹半径 {radius:.3f} 不到场景铺开 {spread:.3f} 的十分之一）"}
    if p_r <= 1e-9 or g_r <= 0:
        return 1.0, {**info, "used": "none", "why": "轨迹半径退化"}
    return g_r / p_r, {**info, "used": "camera", "scale": round(g_r / p_r, 4)}


def align_gauge(gt_pts: np.ndarray, pred_pts: np.ndarray,
                with_scale: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """估计把预测对齐到真值的水平平移加绕 y 轴旋转，返回 (R, t, s)。

    `with_scale=False`（默认）时 s 恒为 1.0，行为和以前完全一样。开了之后多估一个
    各向同性的整体尺度，也就是从刚性变换换成相似变换。

    **为什么要有这个开关。** 尺度本来不算规范自由度：它由 `assumed_camera_height_m`
    定死，所以是可观测的、该被罚的。但实测发现这个假设是个常数而取景不是——ab015
    这一批真值相机高度从 0.62 米到 3.89 米，41.7% 的片段在两米以上，而假设恒为
    1.2 米，于是俯拍那批被压小三到八倍（#68）。配对容差是 0.45 米的绝对值，在这个
    尺度下它们根本配不上，于是「尺度错了」被记成了「检测不出来」，其它所有指标都
    跟着不可解释。
    
    开了尺度之后报出来的数**不是**更好的成绩，是另一个问题的答案：「把尺度这一条
    刨掉之后，还剩多少误差」。两者要并排看，刚性那一列仍然是主口径。

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
        t = (gt_pts.mean(0) - pred_pts.mean(0)) * [1, 0, 1] if len(gt_pts) else np.zeros(3)
        return np.eye(3), t, 1.0
    gc, pc = gt_pts.mean(0), pred_pts.mean(0)
    g, q = gt_pts - gc, pred_pts - pc
    # 只在水平面上求最佳偏航，用 Kabsch 的二维特例
    H = q[:, [0, 2]].T @ g[:, [0, 2]]
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R2 = Vt.T @ np.diag([1.0, d]) @ U.T
    R = np.eye(3); R[0, 0], R[0, 2], R[2, 0], R[2, 2] = R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]
    # 尺度和旋转一起解（Umeyama）：先把旋转定下来，再取「对齐后的内积」比「预测的
    # 平方和」。注意高度也参与——尺度是各向同性的，只在水平面上估会和 y 方向打架。
    s = 1.0
    if with_scale:
        denom = float((q ** 2).sum())
        if denom > 1e-12:
            s = float((g * (q @ R.T)).sum() / denom)
        if not np.isfinite(s) or s <= 1e-6:
            s = 1.0
    t = gc - s * (R @ pc)
    # 高度不参与对齐，开不开尺度都一样：两边的地面都已经在 y=0，预测整体乘上 s
    # 之后地面还在 y=0，不需要再平移。放开 t[1] 会把真实的高度误差吸收掉。
    t[1] = 0.0
    return R, t, s


_AXIS_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def coarse_gauge(gt_pts: np.ndarray, pred_pts: np.ndarray, max_dist_m: float,
                 with_scale: bool = False,
                 scale_hint: float | None = None) -> tuple[np.ndarray, np.ndarray, float, dict]:
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
        return best_R, best_t, 1.0, {"yaw_deg": 0.0, "inliers": 0, "candidates": 0}
    # 阈值不是尺度不变的：尺度差上三倍，按绝对米制投票一对都投不出来，后面估偏航
    # 那步根本启动不了。所以开了尺度之后先用不需要对应关系的均方半径之比垫一脚。
    if not with_scale:
        s0 = 1.0
    elif scale_hint is not None and np.isfinite(scale_hint) and scale_hint > 0:
        s0 = float(scale_hint)
    else:
        s0 = gauge_scale_hint(gt_pts, pred_pts)
    pred_pts = pred_pts * s0
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
    return best_R, best_t, s0, {"yaw_deg": round(float(np.degrees(np.arctan2(best_R[0, 2], best_R[0, 0]))), 2),
                                "inliers": best_in, "scale_hint": round(float(s0), 4),
                                "candidates": len(_AXIS_YAWS_DEG) * len(gt_pts) * len(pred_pts)}


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
             match_on: str = "mean", gauge: str = "rigid",
             min_time_coverage: float | None = None) -> dict:
    """Keep existing metric keys, using a GT window and frozen prefix identities."""
    if min_time_coverage is None:
        min_time_coverage = load_config().get("gt_evaluation", {}).get("min_time_coverage", 0.9)
    if (isinstance(min_time_coverage, bool) or not isinstance(min_time_coverage, (int, float))
            or not np.isfinite(min_time_coverage) or not 0 <= min_time_coverage <= 1):
        raise ValueError("min_time_coverage must be finite and within [0, 1]")
    if gauge not in ("rigid", "similarity"):
        raise ValueError('gauge must be "rigid" or "similarity"')
    if match_on not in ("mean", "start"):
        raise ValueError("match_on must be 'mean' or 'start'")
    if not 0 < holdout_frac < 1 or n_samples < 2 or not np.isfinite(max_dist_m) or max_dist_m < 0:
        raise ValueError("invalid metric configuration")
    try:
        gt_tr, pred_tr = tracks(gt_states), tracks(pred_states)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc), "status": "invalid_states"}
    if not gt_tr:
        return {"error": "真值里没有可比的物体"}

    gt_motion, pred_motion = motion_types(gt_program), motion_types(pred_program)
    gt_class, pred_class = class_names(gt_program), class_names(pred_program)
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
    R0, t0v, s0 = np.eye(3), np.zeros(3), 1.0
    coarse = {"inliers": 0, "candidates": 0, "yaw_deg": 0.0}
    with_scale = gauge == "similarity"
    if observed_pred:
        def centroid(track):
            ts = _supported(obs_times, track)
            return _sample(track, ts).mean(0) if len(ts) else track["pos"].mean(0)
        gt_c = np.vstack([centroid(d) for d in observed_gt.values()])
        pred_c = np.vstack([centroid(d) for d in observed_pred.values()])
        hint, hint_info = ((None, {"used": "off"}) if not with_scale else
                           camera_scale_hint(gt_states, pred_states, gt_c, pred_c))
        R0, t0v, s0, coarse = coarse_gauge(gt_c, pred_c, max_dist_m, with_scale,
                                           hint if hint and hint != 1.0 else None)
        coarse["scale_source"] = hint_info
        rough = {k: {**d, "pos": s0 * (d["pos"] @ R0.T) + t0v} for k, d in observed_pred.items()}
        gp, pp = [], []
        for g, q, _ in match_objects(observed_gt, rough, obs_times, max_dist_m):
            if q is None: continue
            ts = _supported(obs_times, observed_gt[g], observed_pred[q])
            if len(ts):
                gp.append(_sample(observed_gt[g], ts)); pp.append(_sample(observed_pred[q], ts))
        if gp:
            # 精对齐**不再自由解尺度**：配对集是在 s0 这个尺度下筛出来的（容差 0.45 米
            # 是绝对量），在它上面再解一次尺度就是用结论挑样本再在样本上算结论。
            # s0 由相机轨迹定下来之后就冻住，这和平移、偏航冻在观察段是同一个道理。
            # 注意要喂**已经乘过 s0 的**预测点：最终应用的是 s0*(p@R.T)+t，
            # 在未缩放的点上解 R、t 会和它差一个 s0。
            R0, t0v, _s = align_gauge(np.vstack(gp), s0 * np.vstack(pp), with_scale=False)
    aligned = {k: {**d, "pos": s0 * (d["pos"] @ R0.T) + t0v} for k, d in pred_tr.items()}

    # 配对判据。默认 "mean"（观察窗内逐时刻距离的平均）是历史口径，但它**把被评测的量
    # 写进了配对条件**：一个真值里走了几米的物体，如果我们把它的运动写成 static，
    # 平均距离就是行程的一半，远超容差，于是配不上——既算成漏检，又被踢出运动准确率
    # 的分母。实测这是个闭环：分母里剩下的全是不怎么动的物体，于是「恒答 static」
    # 成为最优解，而七个配置的准确率**精确等于**那个常数基线（见 #7）。
    # "start" 用观察窗起点的位置配，和运动无关：物体是不是同一个，和我们有没有把它
    # 的运动猜对，本来是两件事。
    pair_times = obs_times
    if match_on == "start" and len(obs_times):
        pair_times = np.asarray([obs_times[0]], float)
    try:
        pairs = match_objects(observed_gt, prefix(aligned), pair_times, max_dist_m)
    except ValueError:                           # 某条轨迹不覆盖那一瞬，退回整段口径
        pairs = match_objects(observed_gt, prefix(aligned), obs_times, max_dist_m)
    pairs += [(g, None, float("inf")) for g in sorted(set(gt_tr) - set(observed_gt))]
    matched = [(g, p, d) for g, p, d in pairs if p is not None]

    # 诊断：换一个**和运动无关**的配对判据，看能多配上几个。
    #
    # 上面的代价是观察窗内逐时刻距离的平均。一个真值里走了几米的物体，如果我们把它
    # 写成 static（全真那一档里 76% 都是），平均距离就是行程的一半左右，远超容差，
    # 于是配不上。后果有两层：召回量到的其实是「检出**且运动也对**」，而运动类型
    # 准确率只在「位置本来就差不多」的物体上算，系统性偏向不怎么动的那些。
    #
    # 这里只报数、不改默认配对，先看这个偏差有多大。判据是观察窗**起点**的位置：
    # 物体是不是同一个，和我们有没有把它的运动猜对，本来就是两件事。
    start_pairs: list = []
    if len(obs_times):
        t_first = np.asarray([obs_times[0]], float)
        try:
            start_pairs = match_objects(observed_gt, prefix(aligned), t_first, max_dist_m)
        except ValueError:                       # 某条轨迹不覆盖这一瞬，退回不报
            start_pairs = []
    start_matched = {g for g, p, _ in start_pairs if p is not None}
    mean_matched = {g for g, p, _ in pairs if p is not None}
    match_criterion = {
        "by_mean_distance": len(mean_matched),
        "by_start_position": len(start_matched) if start_pairs else "unknown",
        "only_by_start_position": sorted(start_matched - mean_matched) if start_pairs else [],
        "used": match_on,
        "note": "按整段平均距离配对，会把运动判错的物体一并算成没检出。"
                "only_by_start_position 里的是「位置对上了但运动没对上」的那些。",
    }

    def same_class(gt_id: str, pred_id: str) -> bool:
        a, b = gt_class.get(gt_id, ""), pred_class.get(pred_id, "")
        return bool(a) and bool(b) and (a == b or a in b or b in a)

    class_correct = sum(1 for g, q, _ in matched if same_class(g, q))

    gt_nodes = {o["id"]: o for o in gt_program.get("objects", [])}
    pred_nodes = {o["id"]: o for o in pred_program.get("objects", [])}
    shape_errs, scale_ratios = [], []
    for g, q, _ in matched:
        pair = (shape_scale_agreement(gt_nodes[g], pred_nodes[q])
                if g in gt_nodes and q in pred_nodes else None)
        if pair:
            shape_errs.append(pair[0]); scale_ratios.append(pair[1])

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
                                "mode": gauge, "scale": round(float(s0), 4),
                                "scale_source": coarse.get("scale_source", {"used": "off"}),
                                "fitted_on_s": [round(t_start, 3), round(cut, 3)], "coarse": coarse,
                                "note": "水平平移和偏航是视频观测不到的规范自由度，算指标前先对齐掉，"
                                        "否则是在罚一个确定不了的量。变换与身份对应都在观察段定死后冻住，"
                                        "每个窗口共用同一个，留出段不重新拟合。"
                                        "只做刚性对齐，相对结构错了仍会暴露。"},
            "trajectory": traj,
            "motion_type": {"accuracy": acc, "n": len(hits), "confusion": confusion},
            "objects": {"gt": n_gt, "pred": n_pred, "matched": len(matched),
                        "recall": len(matched) / n_gt if n_gt else None,
                        "precision": len(matched) / n_pred if n_pred else None,
                        # 配对只看位置，**不看类名**——一个叫 "cardboard box" 的东西
                        # 摆在真值 marble 的位置上，在召回和精确率里算命中。所以
                        # 「召回 0.500」的意思是「一半的真值物体位置上有东西」，
                        # 不是「一半的物体被正确重建了」。
                        #
                        # 这不是小事：类名决定检索哪个资产，资产决定场景长什么样。
                        # 实测即使把场景里确切存在的类名直接喂给检测器，配对上的 24 个
                        # 里也只有 3 个（12.5%）贴对了标签；换成一组完全不相干的通用
                        # 词表，配对数和精确率几乎不变（30 个 / 0.218 对 24 个 / 0.226），
                        # 而类名准确率是 0（见 #56）。
                        "class_correct": class_correct,
                        "class_accuracy": (class_correct / len(matched)) if matched else None,
                        # 形状和尺度才是「复原得像不像」该报的数。类名在当前这个
                        # 合成数据集上和形状脱钩（生成器从词表挑名字、再独立配一个
                        # 基本体），所以 class_accuracy 在这里恒等于噪声，别拿它
                        # 当成绩也别拿它调参——换成类名有语义的数据集时它才有意义。
                        "shape_error": (float(np.median(shape_errs)) if shape_errs else None),
                        "shape_ok_frac": (float(np.mean(np.asarray(shape_errs) <= 0.2))
                                          if shape_errs else None),
                        "scale_ratio": (float(np.median(scale_ratios)) if scale_ratios else None),
                        "scale_ok_frac": (float(np.mean((np.asarray(scale_ratios) >= 0.8)
                                                        & (np.asarray(scale_ratios) <= 1.25)))
                                          if scale_ratios else None),
                        "match_criterion": match_criterion},
            "window_s": [round(t0, 3), round(t1, 3)],
        }

    return {"full": window(t_start, t_end),
            "holdout": window(cut, t_end),
            "holdout_frac": holdout_frac,
            "matching": {"window_s": [t_start, cut], "pairs": {g: p for g, p, _ in pairs}},
            "note": "holdout仅为参考末段诊断；对应关系只用前段样本并冻结。"
                    "未提供重建输入隔离证明，不宣称留出外推。位置误差不覆盖旋转、相机或物理发散协议。"}
