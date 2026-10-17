"""跨帧刚体配准：从「每帧各算一个中心」换成「帧与帧之间解一个变换」。

为什么要换：包围盒中心是可见点在各轴上 3/97 百分位的中点，可见面一变两端就变，中心
跟着动。实测掩码完全准确时位置误差中位数仍有 0.244 m，而且换成点云中位数或均值更差
（`scripts/probe_motion_signal.py` 量过三条廉价修法，全部失败）。可见点集的任何逐帧
汇总量都给不出干净轨迹——因为每一帧看到的是物体的不同部分，汇总量跟着可见部分走。

跨帧配准绕开了这件事：它不问「这一帧的中心在哪」，只问「这一帧的点云相对上一帧转了
多少、挪了多少」。**相邻两帧的可见面重叠最多**，所以这个问题是良态的，而累积起来就是
轨迹。代价是误差会沿帧累积，所以要报每一步的配准残差，让下游知道哪一段不能信。

这里只做点到点的 ICP，不做特征匹配：物体点云只有几百到几千个点，相邻帧的位姿变化很小，
质心对齐之后迭代几轮就够了。
"""
from __future__ import annotations

import numpy as np


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """已知一一对应时的最优刚体变换，返回 (旋转, 平移)，使 R @ source + t ≈ target。"""
    src_c, dst_c = source.mean(0), target.mean(0)
    H = (source - src_c).T @ (target - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T          # 禁止反射，否则会解出镜像
    return R, dst_c - R @ src_c


def _subsample(points: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[rng.choice(len(points), limit, replace=False)]


def icp(source: np.ndarray, target: np.ndarray, *, max_iter: int = 25,
        tolerance: float = 1e-4, reject_factor: float = 2.5,
        max_points: int = 2000, seed: int = 0) -> dict:
    """把 source 配准到 target。返回旋转、平移、残差和参与配准的点数。

    `reject_factor` 控制剔除：距离大于「中位距离乘这个系数」的对应点不参与求解。
    相邻帧之间总有一部分点是刚露出来或刚被遮住的，它们在另一帧里没有对应，必须剔掉，
    否则会把整个变换拽偏。
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    src = _subsample(np.asarray(source, float), max_points, rng)
    dst = _subsample(np.asarray(target, float), max_points, rng)
    if len(src) < 10 or len(dst) < 10:
        return {"ok": False, "reason": "too few points",
                "rotation": np.eye(3), "translation": np.zeros(3), "rmse": None, "pairs": 0}
    tree = cKDTree(dst)
    R_total, t_total = np.eye(3), np.zeros(3)
    current = src.copy()
    # 先对齐质心：相邻帧位姿差很小，这个初值足够把 ICP 带进收敛域
    shift = dst.mean(0) - current.mean(0)
    current = current + shift
    t_total = t_total + shift
    rmse = None
    for _ in range(max_iter):
        distances, indices = tree.query(current, k=1)
        median = float(np.median(distances))
        keep = distances <= max(median * reject_factor, 1e-6)
        if keep.sum() < 10:
            break
        R_step, t_step = kabsch(current[keep], dst[indices[keep]])
        current = current @ R_step.T + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step
        new_rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        if rmse is not None and abs(rmse - new_rmse) < tolerance:
            rmse = new_rmse
            break
        rmse = new_rmse
    return {"ok": True, "rotation": R_total, "translation": t_total,
            "rmse": rmse, "pairs": int(len(current))}


def track_rigid(clouds: list[np.ndarray], *, anchor: np.ndarray | None = None,
                model_points: int = 6000, model_frames: int = 6, seed: int = 0,
                **icp_kwargs) -> dict:
    """把一串逐帧点云配成一条轨迹。

    点云本身已经在世界坐标里，所以帧与帧之间的变换就是物体这一步的运动。

    **不逐帧链式配准**。实测链式做法虽然每一步的残差只有 0.026 m，轨迹误差却比包围盒
    中心还大（0.298 对 0.220）——每步的小误差沿三十帧累积起来就盖过了信号。这里改成
    维护一个「物体自身坐标系下的累积模型」：每来一帧，先用上一帧解出的位姿把它摆过去
    当初值，再和累积模型配准，然后把对齐后的点并进模型。这样既没有链式累积，重叠面积
    又比只对第一帧配准大得多。

    返回每帧位置、每步残差和锚点。锚点默认取第一帧质心，它落在可见表面上所以绝对位置
    本来就偏；轨迹的**形状**不依赖锚点，而运动估计用的正是形状。
    """
    if not clouds:
        return {"positions": [], "residuals": [], "ok": False, "reason": "no clouds"}
    rng = np.random.default_rng(seed)
    model = np.asarray(clouds[0], float)
    point = model.mean(0) if anchor is None else np.asarray(anchor, float)
    positions = [point.copy()]
    residuals: list[float | None] = []
    steps_t: list[float] = []                  # 每一步解出来的平移量
    steps_r: list[float] = []                  # 每一步解出来的转角（度）
    step_transforms: list[tuple[np.ndarray, np.ndarray]] = []
    R_acc, t_acc = np.eye(3), np.zeros(3)      # 第 k 帧 -> 第 0 帧
    for index, current in enumerate(clouds[1:], start=1):
        cloud = np.asarray(current, float)
        # 用上一帧的位姿做初值：相邻帧位姿接近，这样 ICP 从一开始就在收敛域里
        guess = cloud @ R_acc.T + t_acc
        step = icp(guess, model, **icp_kwargs)
        if step["ok"]:
            # 每步的修正量是判静止最稳的信号：累积位置会被缓慢滑移带偏，
            # 而一个真正没动的物体，每一步解出来的修正都接近单位阵。
            steps_t.append(float(np.linalg.norm(step["translation"])))
            step_transforms.append((step["rotation"].copy(), step["translation"].copy()))
            angle = (np.trace(step["rotation"]) - 1.0) / 2.0
            steps_r.append(float(np.degrees(np.arccos(np.clip(angle, -1.0, 1.0)))))
            R_acc = step["rotation"] @ R_acc
            t_acc = step["rotation"] @ t_acc + step["translation"]
            # 只让前几帧进模型。让它一直长下去的话，静止物体会跟着可见面缓慢滑移——
            # 实测静止漂移从 0.18 m 涨到 0.30 m，而轨迹误差并没有因此变好。
            if index < model_frames:
                aligned = cloud @ R_acc.T + t_acc
                model = np.concatenate([model, aligned], axis=0)
                if len(model) > model_points:
                    model = model[rng.choice(len(model), model_points, replace=False)]
        residuals.append(step["rmse"] if step["ok"] else None)
        positions.append(R_acc.T @ (point - t_acc))
    return {"positions": [np.asarray(p, float).tolist() for p in positions],
            "residuals": residuals, "anchor": point.tolist(),
            "step_translation": steps_t, "step_rotation_deg": steps_r,
            "total_rotation": R_acc.tolist(), "total_translation": t_acc.tolist(),
            "step_transforms": step_transforms,
            "ok": True, "frames": len(positions)}


def looks_static(tracked: dict, scale_m: float, *, translation_frac: float = 0.02,
                 rotation_deg: float = 1.5) -> bool:
    """每一步的修正都小到可以忽略，就认为这个物体没动。

    判静止用每步修正量，不用累积位置：累积位置会被配准的缓慢滑移带偏（实测静止物体
    的累积漂移 0.286 m，比包围盒中心的 0.182 m 还大），而逐步修正量不受滑移影响——
    滑移是每步一点点攒出来的，每一步本身仍然很小。

    阈值按物体自己的尺度给：`scale_m` 传物体包围盒的对角线长度。
    """
    translations = tracked.get("step_translation") or []
    rotations = tracked.get("step_rotation_deg") or []
    if not translations:
        return False
    limit = max(translation_frac * float(scale_m), 1e-4)
    moved = float(np.median(translations))
    turned = float(np.median(rotations)) if rotations else 0.0
    return moved <= limit and turned <= rotation_deg


def _axis_of(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """旋转矩阵的转轴和转角（弧度）。转角接近零时轴没有意义，返回零向量。"""
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-6:
        return np.zeros(3), angle
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]], float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.zeros(3), angle
    return axis / norm, angle


def infer_articulation(tracked: dict, *, min_rotation_deg: float = 8.0,
                       min_slide_m: float = 0.05, axis_consistency: float = 0.85,
                       slide_straightness: float = 0.9) -> dict | None:
    """从配准解出来的逐步变换里推断这是不是一个关节件。

    这是把「关节参数靠手写」换成「从视频里看出来」的那一步。配准每一步给的就是刚体
    变换，所以关节的两种形态可以直接读出来：

    * **铰链**：每一步都绕大致同一根轴转。轴取各步转轴的主方向，支点由整段的累积
      变换解出来——绕轴 a 过支点 c 转 R 的刚体变换，平移部分满足 t = (I - R) c。
    * **滑轨**：每一步几乎不转，平移方向又都差不多。轴取平移的主方向，行程取投影到
      轴上的最小和最大值。

    判不出来就返回 None。宁可不写，也不要往程序里塞一个猜的关节——用户拖一下就会发现。
    """
    transforms = tracked.get("step_transforms") or []
    if len(transforms) < 4:
        return None
    axes, angles, translations = [], [], []
    for rotation, translation in transforms:
        axis, angle = _axis_of(np.asarray(rotation, float))
        if angle > 1e-4 and np.linalg.norm(axis) > 0:
            # 轴的正负号没有意义，统一到同一个半空间再求平均
            axes.append(axis if axis[np.argmax(np.abs(axis))] >= 0 else -axis)
            angles.append(angle)
        translations.append(np.asarray(translation, float))

    total_rotation = np.asarray(tracked.get("total_rotation") or np.eye(3), float)
    total_axis, total_angle = _axis_of(total_rotation)
    if axes and np.degrees(total_angle) >= min_rotation_deg:
        stacked = np.stack(axes)
        mean_axis = stacked.mean(0)
        norm = float(np.linalg.norm(mean_axis))
        if norm > 1e-9:
            mean_axis = mean_axis / norm
            consistency = float(np.abs(stacked @ mean_axis).mean())
            if consistency >= axis_consistency:
                t_total = np.asarray(tracked.get("total_translation") or np.zeros(3), float)
                # (I - R) 沿转轴是奇异的，所以必须截断小奇异值：rcond=None 会让零空间
                # 那个方向主导出一个天文数字。截断之后得到的是垂直于轴的那个分量，
                # 支点沿轴可以任意平移——铰链本来就是一条线，不是一个点。
                pivot, *_ = np.linalg.lstsq(np.eye(3) - total_rotation, t_total, rcond=1e-6)
                # 沿轴挪到离物体最近的位置，写进程序之后人看得懂
                anchor = np.asarray(tracked.get("anchor") or np.zeros(3), float)
                pivot = pivot + mean_axis * float((anchor - pivot) @ mean_axis)
                return {"grab": "hinge", "axis": [float(v) for v in mean_axis],
                        "pivot": [float(v) for v in pivot],
                        "limit": [0.0, round(float(total_angle), 4)],
                        "evidence": {"axis_consistency": round(consistency, 3),
                                     "total_rotation_deg": round(float(np.degrees(total_angle)), 2)}}

    if translations:
        stacked = np.stack(translations)
        total = stacked.sum(0)
        travel = float(np.linalg.norm(total))
        if travel >= min_slide_m and np.degrees(total_angle) < min_rotation_deg:
            direction = total / travel
            # 每一步都沿着同一个方向，才算滑轨；来回乱走的不是
            projected = stacked @ direction
            straightness = float(np.abs(projected).sum() / max(np.linalg.norm(stacked, axis=1).sum(), 1e-9))
            if straightness >= slide_straightness:
                cumulative = np.cumsum(projected)
                return {"grab": "slide", "axis": [float(v) for v in direction],
                        "limit": [round(float(min(0.0, cumulative.min())), 4),
                                  round(float(max(0.0, cumulative.max())), 4)],
                        "evidence": {"straightness": round(straightness, 3),
                                     "travel_m": round(travel, 4)}}
    return None
