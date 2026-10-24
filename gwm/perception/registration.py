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

import math

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


def estimate_normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    """每个点的法向，取邻域主成分里最小的那个方向。符号不统一，点到面用的是平方距离，
    所以不影响结果。"""
    from scipy.spatial import cKDTree

    points = np.asarray(points, float)
    if len(points) < k + 1:
        return np.tile(np.array([0.0, 1.0, 0.0]), (len(points), 1))
    _, idx = cKDTree(points).query(points, k=k)
    neighbourhoods = points[idx]                      # (N, k, 3)
    centred = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
    # 每个邻域的协方差矩阵，批量做特征分解
    cov = np.einsum("nki,nkj->nij", centred, centred) / k
    _, vectors = np.linalg.eigh(cov)
    return vectors[:, :, 0]                           # 最小特征值对应的方向


def visible_subset(model: np.ndarray, cam: np.ndarray, *, k: int = 12,
                   normals: np.ndarray | None = None) -> np.ndarray:
    """模型里**从这个相机位置看得见**的那一部分（的下标）。

    为什么需要它。`refine_poses` 那条「全部帧建完整模型、每帧独立对它配准」实测更差
    （`172d5ae`），原因写得很清楚——

    > 一帧只看得见一个局部表面，拿它去配准更完整的模型时，这块局部可以滑到模型上
    > 好几个位置都说得通；而相邻两帧的可见面大小相当、重叠又多，反倒良态。

    也就是说「部分对完整」欠定，**是因为没告诉它该对哪一部分**。而那个信息我们是有的：
    可见面为什么变？因为相机动了。相机位姿是几何后端一起给出来的，现成的，
    但 `registration.py` 里一行相机都没有——`track_rigid` 只拿到点云。

    所以这里按法向朝不朝着相机来筛模型：留下正对相机的那一半，再去配准。
    这样源和目标就是**以同样的方式**局部的，欠定的根源就去掉了。

    **法向必须先统一朝外。** `estimate_normals` 取的是邻域最小主成分方向，符号是任意
    的，所以不能直接看点积正负。第一版用了 `|n·view| > 阈值`，结果留下的是正对**和**
    背对两个面、只丢掉掠射的一圈——球面上留下来的点质心仍在中间，**根本没选出可见那
    一侧**。单元测试当场抓住了这件事。

    这里先按「从点云质心指向该点」的方向给法向定向（这些基本体都接近凸的，这个定向
    是对的），再留下 `n·view > 0` 的那些，也就是法向朝着相机的面。掠射的也一并去掉
    ——掠射面在深度图里本来就采不到点，留着只会制造假对应。
    """
    model = np.asarray(model, float)
    if len(model) < k + 1:
        return np.arange(len(model))
    n = estimate_normals(model, k=k) if normals is None else np.asarray(normals, float)
    outward = model - model.mean(0)                   # 近似凸：质心指向表面就是朝外
    flip = np.sign(np.einsum("ij,ij->i", n, outward))
    flip[flip == 0] = 1.0
    n = n * flip[:, None]
    view = np.asarray(cam, float) - model
    view = view / np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
    cos = np.einsum("ij,ij->i", n, view)
    return np.flatnonzero(cos > 0.34)                 # 朝向相机且非掠射（约 70 度以内）

def _point_to_plane_step(source: np.ndarray, target: np.ndarray,
                         normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """点到面的一步：小角度线性化之后解一个 6x6 方程。

    点到点在平坦表面上会沿着面滑动——我们的物体大多是盒子和平板，正好是最坏情况。
    点到面只惩罚沿法向的偏差，允许沿面滑动，所以在这种几何上收敛得稳得多。
    """
    cross = np.cross(source, normals)
    A = np.concatenate([cross, normals], axis=1)                   # (N, 6)
    b = -np.einsum("ij,ij->i", source - target, normals)           # (N,)
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    r, t = solution[:3], solution[3:]
    # 小角度旋转向量转回旋转矩阵
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3), t
    axis = r / theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R, t


def icp(source: np.ndarray, target: np.ndarray, *, max_iter: int = 25,
        tolerance: float = 1e-4, reject_factor: float = 2.5,
        max_points: int = 2000, seed: int = 0, point_to_plane: bool = False,
        translation_only: bool = False) -> dict:
    """把 source 配准到 target。返回旋转、平移、残差和参与配准的点数。

    `reject_factor` 控制剔除：距离大于「中位距离乘这个系数」的对应点不参与求解。
    相邻帧之间总有一部分点是刚露出来或刚被遮住的，它们在另一帧里没有对应，必须剔掉，
    否则会把整个变换拽偏。

    `translation_only=True` 时干脆不给转动这个自由度，每步只解对应点的质心差。
    它是用来当**零假设**的：ICP 在部分曲面上总能找到一个让残差更小的转动，
    所以静止物体也会被解出 13.5 度的假转动（#7 在另一批数据上量到 26.6 度），
    而铰链门槛是 8 度——第三档全是假关节就是这么来的。
    有了这一档，「这个物体到底转没转」就能变成一个**带零假设的检验**：
    只有当允许转动显著降低残差时，才认它真的转了。
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    src = _subsample(np.asarray(source, float), max_points, rng)
    dst = _subsample(np.asarray(target, float), max_points, rng)
    if len(src) < 10 or len(dst) < 10:
        return {"ok": False, "reason": "too few points",
                "rotation": np.eye(3), "translation": np.zeros(3), "rmse": None, "pairs": 0}
    tree = cKDTree(dst)
    normals = estimate_normals(dst) if point_to_plane else None
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
        if normals is not None:
            R_step, t_step = _point_to_plane_step(current[keep], dst[indices[keep]],
                                                  normals[indices[keep]])
        elif translation_only:
            # 只解平移：对应点的质心差。转动这个自由度干脆不给。
            R_step = np.eye(3)
            t_step = dst[indices[keep]].mean(0) - current[keep].mean(0)
        else:
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




def rotation_earns_its_keep(source: np.ndarray, target: np.ndarray, *,
                            min_gain: float = 0.05, **icp_kwargs) -> dict:
    """转动要「挣」出来：只有显著降低残差才认它真的转了。

    为什么需要。ICP 无条件享有旋转自由度，而在一块**部分曲面**上它总能找到某个转动
    让残差更小——于是静止物体也被解出转动。实测 13.5 度（#7 在另一批数据上是
    26.6 度），而铰链门槛是 8 度，所以第三档产出的关节全是假的。

    位置那一侧已经被时间平滑治住了（位移误差 0.045 → 0.028，静止漂移 0.077 → 0.039），
    **但平滑不影响转动**，所以这个数一直挂在那儿。它是现在挡住第三档的那一个数。

    做法是把「转没转」变成一个**带零假设的检验**，而不是无条件相信 ICP：
    零假设是「只有平移」，备择是「平移 + 转动」。只有当后者把残差压低超过
    `min_gain`（相对比例）时，才采用带转动的解；否则返回转动为单位阵的那个。

    这正是今天反复踩的那个坑的反面——报任何「更好」之前，先摆出一个什么都不做的基线。
    """
    free = icp(source, target, **icp_kwargs)
    fixed = icp(source, target, translation_only=True, **icp_kwargs)
    if not free.get("ok") or not fixed.get("ok"):
        return free
    r_free, r_fixed = free.get("rmse"), fixed.get("rmse")
    if r_free is None or r_fixed is None or r_fixed <= 1e-9:
        return free
    gain = (r_fixed - r_free) / r_fixed
    out = dict(free if gain >= min_gain else fixed)
    out["rotation_gain"] = float(gain)
    out["rotation_accepted"] = bool(gain >= min_gain)
    return out

def track_rigid(clouds: list[np.ndarray], *, anchor: np.ndarray | None = None,
                model_points: int = 6000, model_frames: int = 6, seed: int = 0,
                refine: int = 0, cam_positions: list | None = None,
                rotation_min_gain: float | None = None,
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

    `cam_positions`（可选，世界坐标下每帧的相机位置）给了之后，每一帧只拿**它看得见
    的那部分模型**去配，而不是整个模型。

    为什么这条值得试。`refine_poses` 那种「全部帧建完整模型、每帧独立配」实测更差
    （`172d5ae`）：一帧只看得见一个局部表面，拿它去配更完整的模型时，这块局部能滑到
    好几个位置都说得通。**欠定的根源是没告诉它该对哪一部分**——而这个信息现成就有：
    可见面为什么变？因为相机动了。相机位姿是几何后端一起给出来的，但
    `registration.py` 里原本一行相机都没有，`track_rigid` 只拿到点云。

    不给就是原来的行为，一点不变。
    """
    if not clouds:
        return {"positions": [], "residuals": [], "ok": False, "reason": "no clouds"}
    rng = np.random.default_rng(seed)
    model = np.asarray(clouds[0], float)
    point = model.mean(0) if anchor is None else np.asarray(anchor, float)
    positions = [point.copy()]
    # 朝向也要一起给出来。旋转类的运动（spin、revolute、周期转动）全靠朝向拟合，
    # 而包围盒的朝向是可见点在 xz 平面上的主轴，物体一转可见面就变，主轴跟着跳。
    orientations = [np.eye(3)]
    residuals: list[float | None] = []
    steps_t: list[float] = []                  # 每一步解出来的平移量
    steps_r: list[float] = []                  # 每一步解出来的转角（度）
    step_transforms: list[tuple[np.ndarray, np.ndarray]] = []
    R_acc, t_acc = np.eye(3), np.zeros(3)      # 第 k 帧 -> 第 0 帧
    transforms = [(R_acc.copy(), t_acc.copy())]
    for index, current in enumerate(clouds[1:], start=1):
        cloud = np.asarray(current, float)
        # 用上一帧的位姿做初值：相邻帧位姿接近，这样 ICP 从一开始就在收敛域里
        guess = cloud @ R_acc.T + t_acc
        # 给了相机轨迹就只拿**这一帧看得见的那部分模型**去配。
        # 「部分对完整」欠定（见 visible_subset 的注释和 172d5ae），而欠定的根源是
        # 没告诉它该对哪一部分——相机位置正好给出这个信息，而且是现成的。
        target = model
        if cam_positions is not None and index < len(cam_positions):
            cam_obj = R_acc @ np.asarray(cam_positions[index], float) + t_acc
            keep = visible_subset(model, cam_obj)
            if len(keep) >= 50:
                target = model[keep]
        step = (rotation_earns_its_keep(guess, target, min_gain=rotation_min_gain, **icp_kwargs)
                if rotation_min_gain is not None else icp(guess, target, **icp_kwargs))
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
        transforms.append((R_acc.copy(), t_acc.copy()))
        positions.append(R_acc.T @ (point - t_acc))
        # R_acc 把第 k 帧映回第 0 帧，所以物体在第 k 帧相对第 0 帧的朝向是它的转置
        orientations.append(R_acc.T.copy())
    if refine > 0 and len(clouds) >= 3 and len(transforms) == len(clouds):
        # 第二遍：全部帧建模型，每帧独立对它配准，链式累积就断开了
        refined = refine_poses(clouds, transforms, iterations=refine, seed=seed, **icp_kwargs)
        positions = [R.T @ (point - t) for R, t in refined]
        orientations = [R.T.copy() for R, t in refined]
    return {"positions": [np.asarray(p, float).tolist() for p in positions],
            "orientations": [np.asarray(r, float).tolist() for r in orientations],
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


# **这个函数在当前链路上不成立，默认关着（`infer_articulation: false`）。原因是
# #7 那个 ICP 伪造转动，不是这里的阈值没调好。**
#
# #7 做过一个合成对照：一个完全静止的物体被 ICP 解出 **26.6°** 的转动，根因是
# 「图像上固定的采样栅格 × 相机运动」让两帧的点落在物体表面的不同位置上。而下面
# 铰链的门槛是 **8°**。实测触发时的转角中位数是 22.4° / 37.8° / 76.5°（三批数据）
# ——和伪造转动同一个量级，全在门槛之上。
#
# 这一个机制解释了四个现象：
#   写出来的关节 12 个 hinge 只有 1 个 slide —— 伪造的是转动，所以几乎全判成铰链；
#   全真值几何 + 全真值分割也救不了（12 个真关节只找回 1 个）—— 完美的掩码和深度
#     不会让 ICP 不伪造转动，伪造来自采样方式随相机动；
#   铰链重建残差那个判据方向是反的（真关节 0.606 > 非关节 0.530）—— 假关节的
#     「转动」本来就是拟合出来的，当然拟合得好；
#   抬门槛没用 —— 真关节的转角和伪造转动完全重叠（#7 里那张表：抬门槛会挡掉
#     十四个真转动里的九个）。
#
# 所以这一档挂在 #7 后面，不是 #56 后面。在 ICP 那个问题解决之前，调这里的任何
# 阈值都没有意义。
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


def refine_poses(clouds: list[np.ndarray], poses: list[tuple[np.ndarray, np.ndarray]],
                 *, iterations: int = 2, model_points: int = 8000, seed: int = 0,
                 **icp_kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
    """拿全部帧建一个完整模型，然后每一帧**独立**对它配准。

    这一步要解决的是链式累积：逐帧两两配准时，每步的小误差会沿三十帧攒起来，实测每步
    残差只有 0.026 m 却得到 0.298 m 的轨迹误差。这里的两遍法把链断开——第二遍里每一帧
    都直接对完整模型配准，相互之间没有依赖，所以误差不会传递。

    模型也比第一遍好：第一遍只能用已经走过的帧（而且实测只吸收前六帧最好，再多反而把
    噪声攒进去），第二遍可以用全部帧，包括物体转过去之后才露出来的那些面。

    `poses` 是第一遍给出的 (R, t)，把第 k 帧的点映到第 0 帧。返回同样形状的精化结果。
    """
    rng = np.random.default_rng(seed)
    current = [(np.asarray(R, float).copy(), np.asarray(t, float).copy()) for R, t in poses]
    for _ in range(max(1, iterations)):
        aligned = [np.asarray(c, float) @ R.T + t for c, (R, t) in zip(clouds, current)]
        model = np.concatenate(aligned, axis=0) if aligned else np.zeros((0, 3))
        if len(model) > model_points:
            model = model[rng.choice(len(model), model_points, replace=False)]
        if len(model) < 20:
            break
        nxt = []
        for cloud, (R, t) in zip(clouds, current):
            guess = np.asarray(cloud, float) @ R.T + t
            step = icp(guess, model, **icp_kwargs)
            if step["ok"]:
                nxt.append((step["rotation"] @ R, step["rotation"] @ t + step["translation"]))
            else:
                nxt.append((R, t))
        current = nxt
    return current
