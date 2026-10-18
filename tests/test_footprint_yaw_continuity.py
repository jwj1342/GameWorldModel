"""足迹主轴的朝向序列要先接连续，再交给运动估计。

`obb_from_points` 的 yaw 来自 xz 点云 PCA 的主轴，带两重规范自由度：特征向量的
符号（180 度，任何形状上都任意）和长短轴互换（90 度，足迹接近正方形时任意）。
两者都会在帧之间跳，而下游把跳变读成转动。

实测（真值感知、真值静止的物体）：足迹比 1.2–1.5 那一档 yaw 的 5–95 分位跨度
中位 73.5 度，2.5 以上只有 5.1 度，log 相关系数 −0.906。#63 里一个真值**静止**
的物体被报出 76.5 度转角，就是这么来的。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from gwm.perception.evidence import continuous_footprint_yaw


def yaws(quats):
    return np.degrees(R.from_quat(np.asarray(quats, float)).as_euler("yxz")[:, 0])


def quats_from_deg(degs):
    return R.from_euler("y", np.radians(np.asarray(degs, float))[:, None]).as_quat()


def test_静止物体的九十度跳轴被消掉():
    """一个方底面的静止物体，主轴在两个方向之间来回切。接连续之后应该几乎不动。"""
    raw = quats_from_deg([0, 90, 0, 90, 90, 0, 90, 0])
    out = continuous_footprint_yaw(raw, aspect_ratio=1.2)
    span = float(np.ptp(yaws(out)))
    assert span < 1e-6, f"跳轴没消掉，跨度还有 {span}°"


def test_静止物体的一百八十度符号翻转被消掉():
    """符号翻转在任何形状上都是规范自由度，细长物体上也要消。"""
    raw = quats_from_deg([30, -150, 30, -150, 30])
    out = continuous_footprint_yaw(raw, aspect_ratio=4.0)
    assert float(np.ptp(yaws(out))) < 1e-6


def test_真实的连续转动每一步都还在():
    """每帧转 10 度转满一整圈，接连续之后逐帧步长仍然是 10 度，没有跳变。

    这里不能断言「总转角还是 350 度」——函数返回的是四元数，而四元数本身按 360 度
    回绕，读回欧拉角一定会折。总转角由下游 `_orientation_signal` 自己展开，这个函数
    负责的是**不出现跳变**，断言就该落在逐帧步长上。

    顺带记一条更硬的限制：足迹主轴的朝向**本质上只定义到 180 度**，长方形转过半圈
    和没转是同一个足迹。所以转超过半圈的 spin 类物体，靠足迹 yaw 原理上就读不出来，
    这不是实现问题。
    """
    raw = quats_from_deg(np.arange(0, 360, 10))
    out = continuous_footprint_yaw(raw, aspect_ratio=4.0)
    steps = np.abs(np.diff(yaws(out)))
    steps = np.minimum(steps, 360 - steps)      # 读回欧拉角会折，比较时折回来
    assert steps.max() < 11.0, f"出现了跳变，最大步长 {steps.max()}°"


def test_细长物体的真实直角转动不被当成跳轴抹掉():
    """足迹比 4.0 的物体真的转了 90 度。这不是规范自由度，不能消。

    这正是「把 90 度那一重也无条件消掉」的过度修正会犯的错，所以单独盯一条。
    """
    raw = quats_from_deg([0, 15, 30, 45, 60, 75, 90])
    out = continuous_footprint_yaw(raw, aspect_ratio=4.0)
    total = float(yaws(out)[-1] - yaws(out)[0])
    assert abs(abs(total) - 90) < 1e-6, f"真的直角转动被抹掉了，剩 {total}°"


def test_方底面上的真实转动仍能看见只要每步不到四十五度():
    """方底面只允许消 90 度的倍数，所以每步小于 45 度的真转动仍然留得住。"""
    raw = quats_from_deg([0, 20, 40, 60, 80, 100, 120])
    out = continuous_footprint_yaw(raw, aspect_ratio=1.1)
    total = float(yaws(out)[-1] - yaws(out)[0])
    assert abs(abs(total) - 120) < 1e-6


def test_少于两帧原样返回():
    one = quats_from_deg([0.0])
    assert np.allclose(continuous_footprint_yaw(one, 1.0), one)
