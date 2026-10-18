"""相似变换对齐：多估一个整体尺度。

为什么要有这个开关（#68）：尺度本来不算规范自由度，它由 `assumed_camera_height_m`
定死，所以可观测、该罚。但实测那个假设是个**常数**而取景不是——ab015 真值相机
高度 0.62 到 3.89 米、41.7% 的片段在两米以上，而假设恒为 1.2 米，俯拍那批被压小
三到八倍。配对容差 0.45 米是绝对量，在那个尺度下根本配不上，于是「尺度错了」被
记成了「检测不出来」，别的指标全都不可解释。

所以 `similarity` 不是一个更宽松的成绩，是另一个问题的答案：把尺度刨掉之后还剩
多少。`rigid` 仍然是主口径，两个要并排看。
"""
import numpy as np
import pytest

from gwm.feedback.gt_metrics import align_gauge, coarse_gauge, gauge_scale_hint, _rot_y


def ring(n=8, r=3.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), np.linspace(0.2, 1.4, n), r * np.sin(a)], 1)


def test_默认是刚性_尺度恒为一():
    gt = ring()
    R, t, s = align_gauge(gt, gt * 0.3)
    assert s == 1.0, "不开开关就不该动尺度——尺度是可观测量，默认必须照罚"
    assert np.linalg.norm((gt * 0.3) @ R.T + t - gt, axis=1).max() > 1.0, \
        "刚性对齐救不了三倍多的尺度差，这正是 rigid 口径要表达的意思"


def test_开了之后能把整体缩放解出来():
    gt = ring()
    for factor in (0.27, 0.34, 1.0, 3.7):
        R, t, s = align_gauge(gt, gt * factor, with_scale=True)
        assert s == pytest.approx(1 / factor, rel=1e-6)
        back = s * (gt * factor) @ R.T + t
        assert np.linalg.norm(back - gt, axis=1).max() < 1e-6


def test_缩放加偏航加平移一起解():
    gt = ring()
    Ry = _rot_y(37.0)
    pred = (gt @ Ry) * 0.3 + np.array([4.0, 0.0, -9.0])
    R, t, s = align_gauge(gt, pred, with_scale=True)
    assert s == pytest.approx(1 / 0.3, rel=1e-3)
    assert np.linalg.norm(s * (pred @ R.T) + t - gt, axis=1).max() < 1e-6


def test_高度不参与对齐_开尺度也一样():
    """两边地面都在 y=0，预测整体乘上 s 之后地面还在 y=0。放开 y 平移会把真实的
    高度误差吸收掉，那是在罚不该罚的、也在放过该罚的。"""
    gt = ring()
    pred = gt + np.array([0.0, 1.5, 0.0])
    for with_scale in (False, True):
        _R, t, _s = align_gauge(gt, pred, with_scale=with_scale)
        assert t[1] == 0.0


def test_粗对齐要先垫一脚尺度_否则投不出票():
    """阈值不是尺度不变的。尺度差三倍多的时候按绝对米制投票一对都投不出来，
    后面估偏航那步根本启动不了——这是 #68 那批俯拍片段的实际处境。"""
    gt = ring()
    pred = (gt @ _rot_y(90.0)) * 0.27
    _R, _t, s0, info = coarse_gauge(gt, pred, 0.45, with_scale=False)
    assert info["inliers"] <= 1, "不垫尺度就该投不出票，这是对照组"
    _R, _t, s0, info = coarse_gauge(gt, pred, 0.45, with_scale=True)
    assert s0 == pytest.approx(1 / 0.27, rel=0.05)
    assert info["inliers"] >= len(gt) - 1, "垫上之后该几乎全部投中"


def test_尺度提示不需要任何对应关系():
    """粗对齐那一步还没有配对，所以这个估计只能用点集自身的铺开程度。"""
    gt = ring()
    shuffled = (gt * 0.2)[::-1]                      # 顺序打乱，对应关系无从谈起
    assert gauge_scale_hint(gt, shuffled) == pytest.approx(5.0, rel=1e-6)
    assert gauge_scale_hint(gt, gt[:1]) == 1.0       # 点太少就别猜


def test_评测入口认这个参数并记下来():
    from gwm.feedback.gt_metrics import evaluate
    gt_prog = {"objects": [{"id": "a", "pose": {"pos": [0, 0.5, 0]}},
                           {"id": "b", "pose": {"pos": [4, 0.5, 0]}}]}
    pred_prog = {"objects": [{"id": "p", "pose": {"pos": [0, 0.5, 0]}},
                             {"id": "q", "pose": {"pos": [1.2, 0.5, 0]}}]}

    def states(prog, scale=1.0):
        return [{"t": t, "objects": [{"id": n["id"], "name": n["id"], "kind": "object",
                                      "pos": [v * scale for v in n["pose"]["pos"]]}
                                     for n in prog["objects"]]}
                for t in (0.0, 0.5, 1.0)]

    out = evaluate(gt_prog, pred_prog, states(gt_prog), states(pred_prog),
                   max_dist_m=0.45, gauge="similarity")
    assert out["full"]["gauge_alignment"]["mode"] == "similarity"
    assert out["full"]["gauge_alignment"]["scale"] > 1.0, \
        "预测整体小 3.3 倍，解出来的尺度该大于一"

    with pytest.raises(ValueError):
        evaluate(gt_prog, pred_prog, states(gt_prog), states(pred_prog), gauge="affine")
