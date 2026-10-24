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
