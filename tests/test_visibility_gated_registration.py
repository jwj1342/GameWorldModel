"""配准时只拿「这一帧看得见的那部分模型」去配。

`refine_poses` 那种「全部帧建完整模型、每帧独立配」实测更差（172d5ae）：一帧只看得见
一个局部表面，拿它去配更完整的模型时，这块局部能滑到好几个位置都说得通。欠定的根源
是**没告诉它该对哪一部分**，而这个信息现成就有——可见面变是因为相机动了。

这里钉三件事：不给相机时行为一点不变；给了相机之后模型确实被筛小了；
以及筛出来的确实是朝向相机那一侧。
"""
import numpy as np

from gwm.perception.registration import track_rigid, visible_subset


def sphere_cloud(n=600, radius=1.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * radius


def test_不给相机时行为完全不变():
    clouds = [sphere_cloud(seed=i) + np.array([0.1 * i, 0, 0]) for i in range(6)]
    a = track_rigid(clouds, seed=0)
    b = track_rigid(clouds, seed=0, cam_positions=None)
    assert a["positions"] == b["positions"]
    assert a["residuals"] == b["residuals"]


def test_可见子集只留朝相机那一侧():
    """球面上，离相机近的那一半该被留下，远的那一半该被丢掉得更多。"""
    pts = sphere_cloud(n=2000, seed=1)
    cam = np.array([10.0, 0.0, 0.0])
    keep = visible_subset(pts, cam)
    assert 0 < len(keep) < len(pts), "既不该全留也不该全丢"
    near = pts[keep][:, 0]
    assert near.mean() > pts[:, 0].mean(), "留下来的该整体更靠近相机那一侧"


def test_点太少时原样返回不炸():
    pts = sphere_cloud(n=5, seed=2)
    keep = visible_subset(pts, np.array([5.0, 0.0, 0.0]))
    assert len(keep) == len(pts)


def test_给了相机也能跑完并且给出同样长度的轨迹():
    clouds = [sphere_cloud(seed=i) + np.array([0.1 * i, 0, 0]) for i in range(6)]
    cams = [np.array([5.0 + 0.1 * i, 1.0, 0.0]) for i in range(6)]
    out = track_rigid(clouds, seed=0, cam_positions=cams)
    assert out["ok"]
    assert len(out["positions"]) == len(clouds)
    assert len(out["residuals"]) == len(clouds) - 1


# ---------- 转动要「挣」出来 ----------
#
# ICP 无条件享有旋转自由度，而在一块部分曲面上它总能找到某个转动让残差更小，
# 于是静止物体也被解出转动：实测 13.5 度（#7 在另一批数据上 26.6 度），
# 而铰链门槛是 8 度。位置那一侧已经被时间平滑治住（0.045→0.028、0.077→0.039），
# **但平滑不影响转动**，所以这就是现在挡住第三档的那一个数。

from gwm.perception.registration import icp, rotation_earns_its_keep


def box_surface(n=800, seed=0):
    """非对称长方体的表面点。

    第一版用的是 x-y 平面上一块**均匀随机**的点片，那是个退化夹具：均匀随机点云没有
    「纹理」，绕法向转过之后每个点仍然能找到近邻，点到点 ICP 根本分不出转没转。
    两条用例因此都失败了——不是实现错，是题目出错了。

    换成三边都不等的长方体表面（1.0 x 0.6 x 0.3），朝向是良定义的。
    """
    rng = np.random.default_rng(seed)
    half = np.array([0.5, 0.3, 0.15])
    pts = rng.uniform(-1, 1, (n, 3)) * half
    axis = rng.integers(0, 3, n)                      # 把每个点推到某一个面上
    sign = rng.choice([-1.0, 1.0], n)
    pts[np.arange(n), axis] = sign * half[axis]
    return pts + rng.normal(0, 0.002, pts.shape)


def test_只平移这一档真的不转():
    src = box_surface(seed=3)
    dst = src + np.array([0.05, -0.02, 0.01])
    out = icp(src, dst, translation_only=True)
    assert out["ok"]
    assert np.allclose(out["rotation"], np.eye(3), atol=1e-9), "这一档不该解出任何转动"


def test_纯平移时转动挣不到_返回零转动():
    src = box_surface(seed=4)
    dst = src + np.array([0.04, 0.01, 0.0])
    out = rotation_earns_its_keep(src, dst, min_gain=0.05)
    ang = np.degrees(np.arccos(np.clip((np.trace(out["rotation"]) - 1) / 2, -1, 1)))
    why = (f"skipped={out.get('rotation_gate_skipped')} "
           f"gain={out.get('rotation_gain')} "
           f"rmse_free={out.get('rmse_free')} "
           f"rmse_trans={out.get('rmse_translation_only')} "
           f"scale={out.get('cloud_scale')} "
           f"already_good={out.get('rotation_already_good')}")
    # 阈值不能卡在 1e-6 度：自由 ICP 在纯平移上会解出 1e-5 度量级的残余转动，
    # 那是数值噪声不是转动。铰链门槛是 8 度，所以 0.5 度足够说明「没转」。
    assert ang < 0.5, f"纯平移却接受了 {ang:.4f} 度的转动（{why}）"
    assert out["rotation_accepted"] is False, why


def test_真的转了的时候转动要被接受():
    src = box_surface(n=1500, seed=5)
    ang = np.radians(25.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0],
                  [np.sin(ang), np.cos(ang), 0],
                  [0, 0, 1.0]])
    dst = src @ R.T
    out = rotation_earns_its_keep(src, dst, min_gain=0.05)
    assert out["rotation_accepted"] is True, "真有转动时不该被零假设挡掉"


def test_不给门槛时行为完全不变():
    clouds = [sphere_cloud(seed=i) + np.array([0.1 * i, 0, 0]) for i in range(6)]
    a = track_rigid(clouds, seed=0)
    b = track_rigid(clouds, seed=0, rotation_min_gain=None)
    assert a["positions"] == b["positions"]
