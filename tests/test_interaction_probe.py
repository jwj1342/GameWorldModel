"""实测可交互性并进报告时的几条规矩。

探针本身的行为由 tests/test_interaction_probe.mjs 盯着（它要真的跑物理）。这里只盯
Python 这一侧：量出来的数怎么和声明的数并排放、跑不起来会不会把链路带崩。
"""
from __future__ import annotations

from gwm.feedback.completeness import interactivity
from gwm.feedback.interaction_probe import merge


PROBE = {
    "ok": True,
    "summary": {"objects": 16, "declared_grabbable": 16, "works": 7,
                "manipulation_success_fraction": 0.438, "interactive_fraction": 0.438,
                "median_clickable_fraction": 0.44, "median_disturbance": 0.004,
                "fell_through": 0},
    "objects": [],
}


def test_声明值和实测值并排放着不互相覆盖():
    program = {"objects": [{"id": f"o{i}", "motion": {"type": "static"}} for i in range(16)]}
    declared = interactivity(program)
    assert declared["manipulable_fraction"] == 1.0          # 程序说十六个都能拖
    merged = merge(declared, PROBE)
    assert merged["manipulable_fraction"] == 1.0            # 声明值原样保留
    assert merged["measured_success_fraction"] == 0.438     # 实测只有七个
    assert merged["measured_works"] == 7


def test_探针没跑成不会让报告缺键():
    merged = merge(interactivity({"objects": []}), {"ok": False, "reason": "TimeoutExpired"})
    assert merged["measured"] == "TimeoutExpired"
    assert "measured_success_fraction" not in merged


def test_样式替换那一项只在量过的时候才出现():
    without = merge({}, PROBE)
    assert "style_swap_max_deviation" not in without
    with_swap = merge({}, {**PROBE, "style_swap": {"max_position_deviation": 0.0,
                                                   "compared_steps": 2400}})
    assert with_swap["style_swap_max_deviation"] == 0.0
    assert with_swap["style_swap_steps"] == 2400
