"""跨帧刚体配准。用合成点云造已知的运动，看能不能解回来。

这是 #7 换信号的那一步：从「每帧各算一个中心」换成「帧与帧之间解一个变换」。
测试里刻意模拟了真实情况下最要紧的一条——每一帧只看得见物体的一部分。
"""
import numpy as np
import pytest

from gwm.perception.registration import icp, kabsch, track_rigid


def box_cloud(n=600, size=(1.0, 0.6, 0.4), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) - 0.5) * np.array(size)


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def test_已知对应时的最优变换():
    cloud = box_cloud()
    R, t = rot_y(37), np.array([1.0, -2.0, 0.5])
    moved = cloud @ R.T + t
    R_hat, t_hat = kabsch(cloud, moved)
    assert np.allclose(R_hat, R, atol=1e-8) and np.allclose(t_hat, t, atol=1e-8)


def test_不会解出镜像():
    """点云近似对称时，没有行列式约束的解法会给出反射，那在物理上不存在。"""
    cloud = box_cloud(seed=3)
    R_hat, _ = kabsch(cloud, cloud * np.array([1.0, 1.0, -1.0]))
    assert np.linalg.det(R_hat) > 0


def test_icp_在没有对应关系时也能解出平移和旋转():
    cloud = box_cloud(seed=1)
    R, t = rot_y(8), np.array([0.3, 0.0, -0.2])
    result = icp(cloud @ R.T + t, cloud)
    assert result["ok"]
    recovered = cloud @ R.T + t
    aligned = recovered @ result["rotation"].T + result["translation"]
    assert np.linalg.norm(aligned.mean(0) - cloud.mean(0)) < 0.05


def test_点太少时如实报告而不是瞎解一个():
    result = icp(np.zeros((3, 3)), box_cloud())
    assert result["ok"] is False and result["reason"] == "too few points"


def test_只看得见一部分时仍然跟得住轨迹():
    """真实情况下每帧只看得见朝向相机的那一面，而且看得见的那一面在变。
    这正是包围盒中心失效的原因，配准必须扛得住。"""
    base = box_cloud(n=1500, seed=5)
    clouds, truth = [], []
    for k in range(12):
        shift = np.array([0.08 * k, 0.0, 0.0])           # 匀速平移
        moved = base + shift
        # 每一帧只保留 x 大于某个随视角移动的阈值的那一半点
        cut = np.quantile(moved[:, 0], 0.35 + 0.02 * k)
        clouds.append(moved[moved[:, 0] > cut])
        truth.append(shift)
    out = track_rigid(clouds)
    assert out["ok"] and out["frames"] == 12
    positions = np.array(out["positions"])
    moved_total = positions[-1] - positions[0]
    # 真值是沿 x 走了 0.88 m；只看得见一半时，包围盒中心那套会被裁切位置带着走
    assert abs(moved_total[0] - 0.88) < 0.12, f"解出来走了 {moved_total}"
    assert abs(moved_total[1]) < 0.08 and abs(moved_total[2]) < 0.08


def test_静止物体在可见面变化时不该被解出运动():
    base = box_cloud(n=1500, seed=7)
    clouds = []
    for k in range(12):
        cut = np.quantile(base[:, 2], 0.2 + 0.04 * k)    # 可见面在变，物体没动
        clouds.append(base[base[:, 2] > cut])
    out = track_rigid(clouds)
    positions = np.array(out["positions"])
    drift = float(np.linalg.norm(positions.max(0) - positions.min(0)))
    assert drift < 0.12, f"静止物体被解出了 {drift:.3f} m 的漂移"


def test_空输入不炸():
    assert track_rigid([])["ok"] is False


def test_静止判据看每步修正量而不是累积位置():
    """累积位置会被配准的缓慢滑移带偏，而真正没动的物体每一步修正都接近单位阵。"""
    from gwm.perception.registration import looks_static
    base = box_cloud(n=1200, seed=11)
    clouds = [base[base[:, 2] > np.quantile(base[:, 2], 0.2 + 0.04 * k)] for k in range(12)]
    tracked = track_rigid(clouds)
    assert looks_static(tracked, scale_m=1.2), tracked["step_translation"][:3]


def test_真的在动的物体不会被判成静止():
    from gwm.perception.registration import looks_static
    base = box_cloud(n=1200, seed=13)
    clouds = [base + np.array([0.12 * k, 0.0, 0.0]) for k in range(12)]
    tracked = track_rigid(clouds)
    assert not looks_static(tracked, scale_m=1.2)


def test_没有步数时不乱判():
    from gwm.perception.registration import looks_static
    assert looks_static({"step_translation": []}, scale_m=1.0) is False


def hinge_clouds(total_deg=50.0, steps=12, pivot=(-0.4, 0.0, 0.0), seed=21):
    """绕 y 轴、过 pivot 转动的门板，每帧只看得见一部分。"""
    base = box_cloud(n=1200, size=(0.7, 1.4, 0.08), seed=seed)
    c = np.array(pivot, float)
    clouds = []
    for k in range(steps):
        R = rot_y(total_deg * k / (steps - 1))
        moved = (base - c) @ R.T + c
        cut = np.quantile(moved[:, 2], 0.15 + 0.02 * k)
        clouds.append(moved[moved[:, 2] > cut])
    return clouds


def slide_clouds(total=0.35, steps=12, axis=(0.0, 0.0, 1.0), seed=23):
    base = box_cloud(n=1200, size=(0.6, 0.25, 0.5), seed=seed)
    a = np.array(axis, float)
    return [base + a * (total * k / (steps - 1)) for k in range(steps)]


def test_从配准里看出这是个铰链():
    from gwm.perception.registration import infer_articulation
    spec = infer_articulation(track_rigid(hinge_clouds()))
    assert spec is not None and spec["grab"] == "hinge", spec
    axis = np.array(spec["axis"])
    assert abs(abs(axis[1]) - 1.0) < 0.15, f"转轴该是 y，实际 {axis}"
    assert np.degrees(spec["limit"][1]) > 25, f"开合角度该接近 50 度，实际 {np.degrees(spec['limit'][1]):.1f}"
    assert abs(np.array(spec["pivot"])[0] + 0.4) < 0.25, f"支点该在 x=-0.4 附近，实际 {spec['pivot']}"


def test_从配准里看出这是个滑轨():
    from gwm.perception.registration import infer_articulation
    spec = infer_articulation(track_rigid(slide_clouds()))
    assert spec is not None and spec["grab"] == "slide", spec
    axis = np.array(spec["axis"])
    assert abs(abs(axis[2]) - 1.0) < 0.15, f"滑轨方向该是 z，实际 {axis}"
    assert spec["limit"][1] > 0.2, f"行程该接近 0.35 m，实际 {spec['limit']}"


def test_静止的物体不该被说成关节件():
    """宁可不写，也不要往程序里塞一个猜的关节——用户拖一下就会发现。"""
    from gwm.perception.registration import infer_articulation
    base = box_cloud(n=1200, seed=29)
    clouds = [base[base[:, 2] > np.quantile(base[:, 2], 0.2 + 0.03 * k)] for k in range(12)]
    assert infer_articulation(track_rigid(clouds)) is None


def test_自由平移不会被说成滑轨以外的东西():
    from gwm.perception.registration import infer_articulation
    base = box_cloud(n=1200, seed=31)
    rng = np.random.default_rng(5)
    clouds = [base + rng.normal(0, 0.12, 3) for _ in range(12)]     # 乱走
    spec = infer_articulation(track_rigid(clouds))
    assert spec is None or spec["grab"] != "hinge", spec


def test_法向估计在平面上指向面外():
    from gwm.perception.registration import estimate_normals
    rng = np.random.default_rng(3)
    plane = np.column_stack([rng.random(400), np.zeros(400), rng.random(400)])   # y=0 的平面
    n = estimate_normals(plane)
    assert np.abs(np.abs(n[:, 1]) - 1.0).mean() < 0.05, "平面的法向该是 y"


def test_点到面在平板上比点到点稳():
    """平板是点到点最坏的情况：它会沿着面滑动。我们的物体大多是盒子和平板。"""
    from gwm.perception.registration import icp
    rng = np.random.default_rng(9)
    slab = np.column_stack([rng.random(900) * 2 - 1, rng.normal(0, 0.01, 900),
                            rng.random(900) * 2 - 1])               # 很薄的一块板
    shift = np.array([0.0, 0.08, 0.0])                              # 只沿法向挪
    moved = slab + shift
    plane = icp(moved, slab, point_to_plane=True)
    assert plane["ok"]
    # 沿法向的分量要解出来
    assert abs(plane["translation"][1] + 0.08) < 0.03, plane["translation"]


def test_点到面不会解出镜像或发散():
    from gwm.perception.registration import icp
    cloud = box_cloud(n=800, seed=17)
    R, t = rot_y(10), np.array([0.2, 0.05, -0.1])
    out = icp(cloud @ R.T + t, cloud, point_to_plane=True)
    assert out["ok"] and np.linalg.det(out["rotation"]) > 0.9
    assert np.all(np.isfinite(out["translation"])) and np.linalg.norm(out["translation"]) < 2.0
