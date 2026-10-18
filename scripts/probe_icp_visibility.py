#!/usr/bin/env python
"""ICP 把「可见面换了」当成「物体转了」吗——纯合成的受控实验。

起因（#7）：把 `track_rigid` 单独拎出来，拿真值深度 + 真值掩码渲出的点云喂进去，
两个方向都错：

  真值**根本不转**的物体，解出 19.85° / 73.05° / 39.09° 的转动（静止 / 纯平移 / 滑轨）
  真的在转的 `spin`，解出的序列和真值的相关系数只有 0.095

怀疑是**自遮挡**：单目永远只看得见朝向相机的一半，两帧之间可见的那半换了，ICP 会把
「可见面换了」解成「物体转了」。这个脚本不渲染、不跑感知，直接在合成点云上做一个
三乘三的受控对照，几秒钟跑完：

  运动：静止 / 纯平移 / 绕 y 轴转
  可见性：完整点云 / 固定相机只看得见的一面 / 相机绕着走只看得见的一面

如果「完整」那一列全对、「只看得见一面」那两列出错，自遮挡这个假设就成立，
而修法在「用什么表示物体」，不在 ICP 的参数——那套参数已经扫完了。

用法:
  python -m scripts.probe_icp_visibility
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.perception.registration import track_rigid  # noqa: E402


def box_surface(extent=(0.6, 0.4, 0.3), per_face=900, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """长方体表面上的点，连同每个点的外法线。法线是用来做可见性剔除的。"""
    rng = np.random.default_rng(seed)
    half = np.asarray(extent, float) / 2
    pts, normals = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            uv = rng.uniform(-1.0, 1.0, size=(per_face, 2))
            p = np.zeros((per_face, 3))
            others = [i for i in range(3) if i != axis]
            p[:, others[0]] = uv[:, 0] * half[others[0]]
            p[:, others[1]] = uv[:, 1] * half[others[1]]
            p[:, axis] = sign * half[axis]
            n = np.zeros((per_face, 3)); n[:, axis] = sign
            pts.append(p); normals.append(n)
    return np.vstack(pts), np.vstack(normals)


def camera_at(mode: str, k: int, n: int) -> np.ndarray | None:
    """相机位置。None 表示不做可见性剔除（完整点云）。"""
    if mode == "full":
        return None
    if mode.startswith("fixed"):      # orbit_grid_voxel 等带后缀的都走下面的绕行相机
        return np.array([0.0, 0.8, 3.0])
    # 整段绕物体走三分之一圈，但**角速度不均匀**——均匀的话每步角步长是常数，
    # 和任何东西的相关系数都算不出来（分母是零方差）。真实手持拍摄也不是匀速的。
    frac = k / max(n - 1, 1)
    angle = 2 * np.pi * 0.35 * (frac + 0.25 * np.sin(2 * np.pi * frac))
    return np.array([3.0 * np.sin(angle), 0.8, 3.0 * np.cos(angle)])


def pose_at(motion: str, k: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """第 k 帧的真值旋转和平移。"""
    frac = k / max(n - 1, 1)
    if motion == "static":
        return np.eye(3), np.zeros(3)
    if motion == "translate":
        return np.eye(3), np.array([1.2 * frac, 0.0, 0.0])
    if motion == "rotate":
        return R.from_euler("y", 90.0 * frac, degrees=True).as_matrix(), np.zeros(3)
    raise ValueError(motion)


def image_grid_resample(pts: np.ndarray, eye: np.ndarray, bins: int = 48) -> np.ndarray:
    """按相机的像素格重采样一遍，模拟深度图。

    这是深度图点云和「在物体表面上均匀采样」最本质的区别：深度图的采样格子**钉在
    图像上**，不钉在物体上。相机一动，格子扫过的物体表面位置就全换了——两帧之间
    看到的根本不是同一批物理点。ICP 找最近邻对应的时候，这种重采样本身就会被解成
    一点运动。
    """
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = pts - eye
    depth = rel @ forward
    u = (rel @ right) / np.maximum(depth, 1e-6)
    v = (rel @ up) / np.maximum(depth, 1e-6)
    lo, hi = np.array([u.min(), v.min()]), np.array([u.max(), v.max()])
    span = np.maximum(hi - lo, 1e-9)
    cell = np.floor((np.column_stack([u, v]) - lo) / span * (bins - 1)).astype(int)
    key = cell[:, 0] * bins + cell[:, 1]
    order = np.lexsort((depth, key))                 # 每个格子留最近的那个点
    keyed = key[order]
    first = np.ones(len(order), bool)
    first[1:] = keyed[1:] != keyed[:-1]
    return pts[order[first]]


def voxel_downsample(pts: np.ndarray, size: float) -> np.ndarray:
    """按固定边长的体素格取每格质心。

    这是对「采样格子钉在图像上」最直接的对冲：体素格钉在世界里，不随相机动。
    两帧之间同一块表面落进同一批体素，每格的质心就不再跟着相机扫来扫去。
    代价是分辨率被体素边长限死，而且物体自己动的时候格子对不齐——所以这只是
    一个**候选修法的受控验证**，不是直接能抄进 `track_rigid` 的实现。
    """
    if not len(pts):
        return pts
    cell = np.floor(pts / size).astype(np.int64)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    out = np.zeros((inverse.max() + 1, 3))
    counts = np.bincount(inverse, minlength=len(out)).astype(float)
    for axis in range(3):
        out[:, axis] = np.bincount(inverse, weights=pts[:, axis], minlength=len(out))
    return out / counts[:, None]


def make_clouds(motion: str, visibility: str, n_frames: int = 24) -> list[np.ndarray]:
    base, base_normals = box_surface()
    clouds = []
    for k in range(n_frames):
        rot, trans = pose_at(motion, k, n_frames)
        pts = base @ rot.T + trans
        if visibility == "full":
            clouds.append(pts); continue
        normals = base_normals @ rot.T
        eye = camera_at(visibility, k, n_frames)
        view = eye - pts
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        keep = np.einsum("ij,ij->i", normals, view) > 0.05      # 背向相机的剔掉
        visible = pts[keep]
        if "_grid" in visibility:
            visible = image_grid_resample(visible, eye)
        if visibility.endswith("_voxel"):
            visible = voxel_downsample(visible, 0.02)
        clouds.append(visible)
    return clouds


def solved_rotation_against_template(motion: str, visibility: str, n_frames: int = 24) -> float:
    """把**完整模板**配到每一帧的局部观测上，而不是让局部观测互相配。

    受控实验里「完整点云」那一行是全对的（静止 0.3°、转 90° 解出 89.9°），而所有出错
    的行都有一个共同点：被配准的两边**都**是随视角扫动的局部表面。只要有一边是钉死的
    完整形状，采样就不再跟着相机跑。

    我们手上已经有这个完整形状——#60 合进来的资产检索给每个物体配了一个网格资产。
    这里用生成点云的那个长方体当模板，量一下这条路的上限。

    解出的是「把这一帧的观测摆回模板坐标系」的旋转，物体自身的转动是它的逆；
    这里比的是整段的转角**范围**，取逆不改变大小，所以可以直接比。
    """
    from gwm.perception.registration import icp
    template, _ = box_surface()
    clouds = make_clouds(motion, visibility, n_frames)
    angles = []
    for cloud in clouds:
        if len(cloud) < 30:
            continue
        # **方向很要紧**：source 必须是局部观测、target 必须是完整模板。
        # 反过来配（模板当 source）的话，模板上一多半的点在局部观测里根本没有对应，
        # 而 `icp` 的剔除阈值是按对应距离的中位数定的，被这批没有对应的点带偏，
        # 整个模板会被拽进一个错的位姿。第一版就是反的，静止物体解出 78.3°。
        step = icp(cloud, template)
        angles.append(float(np.degrees(R.from_matrix(
            np.asarray(step["rotation"], float)).magnitude())))
    return float(np.ptp(angles)) if len(angles) > 1 else float("nan")


def solved_rotation_deg(clouds: list[np.ndarray]) -> np.ndarray:
    tracked = track_rigid(clouds)
    orientations = tracked.get("orientations") or []
    if len(orientations) != len(clouds):
        return np.array([])
    return np.degrees([R.from_matrix(np.asarray(m, float)).magnitude() for m in orientations])


def forward_backward_gap(clouds: list[np.ndarray]) -> float:
    """相邻帧正反各配一次，看两个解对不对得上（度，中位数）。

    一个候选的**自检信号**。真实的刚体运动下，把 A 配到 B 和把 B 配到 A 解出来的
    应该互为逆变换，两者复合起来是单位阵。而「可见面随视角扫动」造出来的假转动不是
    真实运动，它取决于谁当模板——正反两个方向会各自偏向自己的模板，复合起来就不是
    单位阵了。

    这个量逐物体逐帧都能算，不需要真值，所以如果它真能把假转动和真转动分开，
    就能直接当成「这一帧的转动可信吗」的门——而现在那道门用的是足迹长短边比，
    已经量过分不开（#7 排除清单第十二条）。
    """
    from gwm.perception.registration import icp
    gaps = []
    for a, b in zip(clouds[:-1], clouds[1:]):
        if len(a) < 30 or len(b) < 30:
            continue
        fwd, bwd = icp(a, b), icp(b, a)
        composed = np.asarray(fwd["rotation"], float) @ np.asarray(bwd["rotation"], float)
        gaps.append(float(np.degrees(R.from_matrix(composed).magnitude())))
    return float(np.median(gaps)) if gaps else float("nan")


def camera_step_correlation(clouds: list[np.ndarray], visibility: str) -> float:
    """逐步解出的转角，和相机这一步转了多少，相关到什么程度。

    正反一致性那个自检信号失败之后（出错的那几行全是 0.00°，ICP 自洽但自洽地错），
    说明假转动不是模板偏置——相邻两帧采到的**确实**是差一个小旋转的两块表面，
    ICP 如实报告了它。所以自检要从别处找，而最现成的外部量就是相机自己的运动：
    假转动由相机扫过物体表面造成，它应该跟着相机的角步长走；物体自身的转动不该。

    返回逐步解出转角与逐步相机转角的相关系数。接近 1 就说明可以据此识别甚至扣除。
    """
    from gwm.perception.registration import icp
    n = len(clouds)
    cam = [camera_at(visibility, k, n) for k in range(n)]
    if any(c is None for c in cam):
        return float("nan")
    cam_step, obj_step = [], []
    for k in range(n - 1):
        if len(clouds[k]) < 30 or len(clouds[k + 1]) < 30:
            continue
        a, b = cam[k] / np.linalg.norm(cam[k]), cam[k + 1] / np.linalg.norm(cam[k + 1])
        cam_step.append(float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))))
        step = icp(clouds[k], clouds[k + 1])
        obj_step.append(float(np.degrees(R.from_matrix(
            np.asarray(step["rotation"], float)).magnitude())))
    if len(cam_step) < 3 or np.ptp(cam_step) < 1e-9 or np.ptp(obj_step) < 1e-12:
        return float("nan")
    return float(np.corrcoef(cam_step, obj_step)[0, 1])


def main() -> int:
    n = 24
    motions = [("static", "静止"), ("translate", "纯平移"), ("rotate", "绕 y 轴转 90°")]
    views = [("full", "完整点云"), ("fixed", "固定相机、只留可见面"),
             ("orbit", "相机绕着走、只留可见面"),
             ("fixed_grid", "固定相机 + 像素格重采样"),
             ("orbit_grid", "相机绕着走 + 像素格重采样"),
             ("orbit_grid_voxel", "再加体素下采样（候选修法）")]
    truth_range = {"static": 0.0, "translate": 0.0, "rotate": 90.0}

    print(f"{'运动':<16}{'可见性':<22}{'真值转角':>9}{'解出转角':>10}"
          f"{'正反不一致':>11}{'与相机相关':>10}{'点数':>8}")
    for motion, mlabel in motions:
        for view, vlabel in views:
            clouds = make_clouds(motion, view, n)
            sizes = [len(c) for c in clouds]
            solved = solved_rotation_deg(clouds)
            got = float(np.ptp(solved)) if len(solved) else float("nan")
            gap = forward_backward_gap(clouds)
            corr = camera_step_correlation(clouds, view)
            corr_txt = "  —  " if np.isnan(corr) else f"{corr:+.3f}"
            print(f"{mlabel:<16}{vlabel:<22}{truth_range[motion]:>9.1f}°"
                  f"{got:>9.1f}°{gap:>10.2f}°{corr_txt:>9}{int(np.median(sizes)):>8}")
        print()
    print("读法：「完整点云」是 ICP 的下限——它要是也错，问题不在点云怎么来的。")
    print("     「只留可见面」比它差多少，是自遮挡贡献的。")
    print("     「像素格重采样」再差多少，是**采样格子钉在图像上而不是物体上**贡献的")
    print("     ——深度图点云和表面均匀采样最本质的区别就在这里：相机一动，两帧之间")
    print("     看到的根本不是同一批物理点。")
    print("     最后一行是候选修法：体素格钉在世界里不随相机动，看能不能把假转动压回去。")
    print()
    print("—— 换一种配法：完整模板 → 每一帧的局部观测 ——")
    print(f"{'运动':<16}{'真值转角':>9}{'模板配局部解出':>15}")
    for motion, mlabel in motions:
        got = solved_rotation_against_template(motion, "orbit_grid", n)
        print(f"{mlabel:<16}{truth_range[motion]:>9.1f}°{got:>14.1f}°")
    print("  （被配准的两边只要有一边是钉死的完整形状，采样就不再跟着相机跑。")
    print("   我们已经有这个形状——#60 的资产检索给每个物体都配了网格。）")
    print()
    print("「正反不一致」这个自检信号**失败了**：出错的那几行也是 0.00°。ICP 自洽地错——")
    print("相邻两帧采到的确实是差一个小旋转的两块表面，它如实报告了，问题在问题本身。")
    print("「与相机相关」是换的第二个自检信号：假转动由相机扫过物体表面造成，应该跟着")
    print("相机的角步长走，物体自身的转动不该。接近 1 就说明能据此识别甚至扣除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
