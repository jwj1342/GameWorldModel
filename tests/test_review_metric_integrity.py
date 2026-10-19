"""Offline invariants for prefix isolation, legacy motion and measured coverage."""
from copy import deepcopy

import pytest

from gwm.feedback.gt_metrics import camera_scale_hint, evaluate
from gwm.feedback.report import aggregate
from gwm.synthesis.direct import evidence_to_program


def node(oid, x, extent=(1, 1, 1)):
    return {"id": oid, "class": "box", "pose": {"pos": [x, .5, 0]},
            "geom": {"kind": "primitive", "shape": "box", "extent": list(extent)}}


def states(entries, times=range(11)):
    return [{"t": float(t), "camera": {"pos": [float(t), 2, 0]},
             "objects": [{"id": oid, "name": name, "kind": "object", "pos": [x, .5, 0]}
                         for oid, name, x in entries]} for t in times]


def test_similarity_does_not_read_tail_camera():
    gt = {"objects": [node("g0", 0), node("g1", 4)]}
    pred = {"objects": [node("p0", 0), node("p1", 4)]}
    gs = states([("g0", "g0", 0), ("g1", "g1", 4)])
    ps = states([("p0", "p0", 0), ("p1", "p1", 4)])
    before = evaluate(gt, pred, gs, ps, gauge="similarity", max_dist_m=.45)
    for s in ps:
        if s["t"] > 8:
            s["camera"]["pos"][0] *= 100
    after = evaluate(gt, pred, gs, ps, gauge="similarity", max_dist_m=.45)
    assert before == after
    assert after["full"]["gauge_alignment"]["scale"] == 1


def test_camera_scale_uses_common_time_support_and_sampling():
    gt = states([], times=(0, 1, 2, 3, 4))
    pred = states([], times=(2, 2.2, 2.4, 2.6, 2.8, 3))
    for s in pred:
        s["camera"]["pos"][0] /= 4
    factor, info = camera_scale_hint(gt, pred)
    assert factor == pytest.approx(4)
    assert info["window_s"] == [2, 3]


def test_camera_scale_disjoint_support_is_unavailable():
    factor, info = camera_scale_hint(states([], times=(0, 1)), states([], times=(2, 3)))
    assert factor == 1 and info["used"] == "none"


def test_tail_camera_cannot_supply_missing_prefix_camera():
    prog = {"objects": [node("g", 0)]}
    gs = states([("g", "g", 0)])
    ps = deepcopy(gs)
    for s in ps:
        if s["t"] <= 8:
            s.pop("camera")
    out = evaluate(prog, prog, gs, ps, gauge="similarity")
    info = out["full"]["gauge_alignment"]["scale_source"]
    assert info["used"] == "none" and info["pred_frames"] == 0


def test_invalid_camera_samples_do_not_crash_or_extrapolate():
    gt = states([], times=(0, 1, 2))
    pred = deepcopy(gt)
    pred[1]["camera"]["pos"] = ["bad", 2, 0]
    pred[2]["camera"]["pos"] = [float("nan"), 2, 0]
    factor, info = camera_scale_hint(gt, pred)
    assert factor == 1 and info["used"] == "none"


@pytest.mark.parametrize("moves", [None, False, True])
def test_explicit_legacy_motion_is_not_erased(moves):
    ev = {"meta": {"clip": "synthetic", "duration": 2},
          "camera": {"intrinsics": {"fov_deg": 60, "width": 640, "height": 360},
                     "poses": [{"t": float(t), "pos": [0, 1.2, 5], "quat": [0, 0, 0, 1]}
                               for t in range(3)]},
          "static": {"planes": []},
          "objects": [{"id": "moving_structure", "class_guess": "rail", "confidence": .9,
                       "obb": [{"t": float(t), "center": [t, .5, 0], "size": [1, 1, 1],
                                "quat": [0, 0, 0, 1]} for t in range(3)],
                       "motion_guess": {"type": "prismatic", "axis": [1, 0, 0],
                                        "speed": 1, "conf": .9}, "contacts": []}]}
    if moves is not None:
        ev["objects"][0]["motion_guess"]["moves"] = moves
    original = deepcopy(ev)
    out = evidence_to_program(ev, cfg={})
    assert ev == original
    moving = next(o for o in out["objects"] if o["id"] == "moving_structure")
    assert moving["motion"]["type"] == "trajectory"
    assert all(o["id"] != "moving_structure" for o in out["static"])


def instance_result(missing_geometry=False):
    gt = {"objects": [node("parent", 0), node("single", 8)]}
    gt["objects"][0]["instances"] = [{"pos": [0, .5, 0]}, {"pos": [4, .5, 0]}]
    pred = {"objects": [node("p0", 0), node("p1", 4), node("p2", 8)]}
    if missing_geometry:
        gt["objects"][0]["geom"] = {}
    return evaluate(gt, pred,
                    states([("parent", "parent#1", 0), ("parent", "parent#2", 4),
                            ("single", "single", 8)]),
                    states([("p0", "p0", 0), ("p1", "p1", 4), ("p2", "p2", 8)]),
                    max_dist_m=.45)


def test_instances_use_parent_geometry_and_labels():
    result = instance_result()
    ob = result["full"]["objects"]
    assert ob["matched"] == ob["class_correct"] == ob["geometry_n"] == 3
    assert ob["shape_error"] == 0 and ob["scale_ratio"] == 1
    assert result["full"]["shape_kind"]["n"] == 3
    assert set(ob["geometry_per_object"]) == {"parent#1", "parent#2", "single"}


def test_predicted_instances_also_resolve_parent_node():
    gt = {"objects": [node("g0", 0), node("g1", 4)]}
    pred = {"objects": [node("pred_parent", 0)]}
    pred["objects"][0]["instances"] = [{"pos": [0, .5, 0]}, {"pos": [4, .5, 0]}]
    out = evaluate(gt, pred, states([("g0", "g0", 0), ("g1", "g1", 4)]),
                   states([("pred_parent", "pred_parent#1", 0), ("pred_parent", "pred_parent#2", 4)]))
    assert out["full"]["objects"]["geometry_n"] == 2
    assert out["full"]["objects"]["class_accuracy"] == 1
    assert out["full"]["shape_kind"]["accuracy"] == 1


def test_missing_geometry_is_not_a_success_or_a_failure():
    result = instance_result(missing_geometry=True)
    ob = result["full"]["objects"]
    assert ob["matched"] == 3 and ob["geometry_n"] == 1
    assert ob["geometry_per_object"]["parent#1"]["shape_error"] is None
    report = aggregate([{"metrics": result}])
    assert report["shape_ok"]["n"] == report["scale_ok"]["n"] == 1
    assert report["scale_ratio"]["n"] == 1


def test_scale_median_is_pooled_over_objects_not_run_medians():
    def run(values):
        return {"metrics": {"full": {"objects": {
            "geometry_per_object": {str(i): {"shape_error": 0, "scale_ratio": v}
                                    for i, v in enumerate(values)}}}}}
    report = aggregate([run([1.] * 9), run([3.])])
    assert report["scale_ratio"]["n"] == 10
    assert report["scale_ratio"]["median"] == 1


def test_old_report_without_measured_count_is_not_inflated():
    report = aggregate([{"clip": "old", "metrics": {"full": {"objects": {
        "matched": 3, "shape_ok_frac": 1., "scale_ok_frac": 1., "scale_ratio": 1.}}}}])
    assert report["shape_ok"] is None and report["scale_ok"] is None
    assert report["scale_ratio"] is None
    assert report["scale_ratio_run_medians"]["n"] == 1
    assert report["geometry_unavailable_runs"] == ["old"]


def test_summary_with_known_measured_count_recovers_only_binary_rates():
    report = aggregate([{"metrics": {"full": {"objects": {
        "matched": 5, "geometry_n": 2, "shape_ok_frac": .5,
        "scale_ok_frac": 1., "scale_ratio": 1.}}}}])
    assert report["shape_ok"]["n"] == 2 and report["shape_ok"]["mean"] == .5
    assert report["scale_ok"]["n"] == 2 and report["scale_ok"]["mean"] == 1
    assert report["scale_ratio"] is None
    assert report["scale_ratio_run_medians"]["n"] == 1
