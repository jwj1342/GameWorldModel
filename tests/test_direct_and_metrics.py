import numpy as np
from gwm.synthesis.direct import evidence_to_program, motion_from_guess
from gwm.compiler.validate import validate
from gwm.feedback.metrics import id_to_color, registry_order, iou
from gwm.perception.evidence import classify_motion, obb_from_points

CFG = {"perception": {"motion": {"static_pos_thresh_m": 0.05, "static_rot_thresh_deg": 3.0, "revolute_axis_stability": 0.9, "periodic_autocorr_peak": 0.5, "smooth_window": 3}}}

def _ev():
    ts = np.linspace(0, 8, 33); cs = [[0.0 + 0.0, 1.0 + 1.2 * np.sin(2 * np.pi * t / 4.0), 0.0] for t in ts]
    return {"meta": {"clip": "x", "duration": 8.0, "scale": "relative"},
            "camera": {"intrinsics": {"fov_deg": 60, "width": 640, "height": 360}, "poses": [{"t": float(t), "pos": [0, 1.2, 5], "quat": [0, 0, 0, 1]} for t in ts]},
            "static": {"planes": [{"kind": "ground", "conf": 0.8, "center_hint": [0, 0, 0], "extent_hint": [10, 0.2, 10]}]},
            "objects": [{"id": "platform_1", "class_guess": "platform", "confidence": 0.9, "is_dynamic": True, "obb": [{"t": float(t), "center": c, "quat": [0, 0, 0, 1], "size": [2, 0.3, 2]} for t, c in zip(ts, cs)],
                         "motion_guess": {"type": "periodic_translate", "axis": [0, 1, 0], "period": 4.0, "amp": 1.2, "conf": 0.8}, "contacts": []}], "keyframes": []}

def test_direct_translation_validates():
    p = evidence_to_program(_ev(), "x"); r = validate(p); assert r["ok"], r["errors"]; assert p["objects"][0]["motion"]["type"] == "periodic_translate"

def test_classify_periodic_and_static():
    ts = np.linspace(0, 8, 33); c = np.stack([np.zeros_like(ts), 1 + 1.2 * np.sin(2 * np.pi * ts / 4), np.zeros_like(ts)], 1)
    mg = classify_motion(ts, c, np.zeros_like(ts), CFG); assert mg["type"] == "periodic_translate" and abs(mg["period"] - 4.0) < 0.6
    mg2 = classify_motion(ts, np.tile([1, 2, 3], (33, 1)) + 0.001 * np.random.default_rng(0).standard_normal((33, 3)), np.zeros_like(ts), CFG); assert mg2["type"] == "static"

def test_classify_linear():
    ts = np.linspace(0, 4, 17); c = np.stack([ts * 0.5, np.ones_like(ts), np.zeros_like(ts)], 1)
    mg = classify_motion(ts, c, np.zeros_like(ts), CFG); assert mg["type"] in ("prismatic", "trajectory")

def test_obb_from_points():
    rng = np.random.default_rng(0); P = rng.uniform([-1, 0, -0.25], [1, 0.5, 0.25], size=(500, 3))
    ob = obb_from_points(P); assert ob is not None and abs(ob["size"][0] - 2) < 0.3 and abs(ob["size"][2] - 0.5) < 0.2

def test_id_colors_unique_and_registry():
    cols = {id_to_color(i) for i in range(1, 300)}; assert len(cols) == 299
    prog = {"static": [{"id": "g"}], "objects": [{"id": "a"}, {"id": "b", "instances": [1, 2, 3]}]}
    reg = registry_order(prog); assert [r["idx"] for r in reg] == [1, 2, 3, 4, 5] and reg[-1]["id"] == "b"
    a = np.zeros((4, 4), bool); a[:2] = True; b = np.zeros((4, 4), bool); b[1:3] = True; assert abs(iou(a, b) - 1 / 3) < 1e-6

def test_camera_convention_roundtrip():
    """A point 2 m in front of camera 0 (OpenCV +z) lands at -z in the y-up world; three.js camera quat looks down -Z."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from gwm.perception.evidence import _fix, to_world, CV2THREE, project
    c2w = np.eye(4); M = _fix(c2w)
    pw = to_world(np.array([[0.0, 0.0, 2.0]]), M); assert np.allclose(pw, [[0, 0, -2]])
    Rw = M[:3, :3] @ CV2THREE; fwd = Rw @ np.array([0, 0, -1.0]); assert np.allclose(fwd, [0, 0, -1])  # three.js camera forward
    uv, ok = project(np.array([[0.0, 0.0, -2.0]]), np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1]], float), M, 1.0)
    assert ok[0] and np.allclose(uv[0], [320, 180])

def test_scale_program_and_static_classes():
    import json, copy
    from pathlib import Path
    from gwm.binding.platformer import scale_program, bind
    from gwm.config import load_config
    from gwm.compiler.validate import validate
    cfg = load_config()
    ex = json.loads((Path(__file__).resolve().parents[1] / "examples/handwritten/program.json").read_text())
    p2 = scale_program(ex, 2.0)
    assert p2["static"][0]["geom"]["extent"][0] == ex["static"][0]["geom"]["extent"][0] * 2 and p2["objects"][0]["motion"]["amp"] == ex["objects"][0]["motion"]["amp"] * 2
    assert validate(p2)["ok"]
    ev = _ev(); ev["objects"].append({"id": "floor_2", "class_guess": "floor", "confidence": 0.9, "is_dynamic": False, "obb": [{"t": 0.0, "center": [0, -0.1, 0], "quat": [0, 0, 0, 1], "size": [6, 0.2, 6]}, {"t": 1.0, "center": [0, -0.1, 0], "quat": [0, 0, 0, 1], "size": [6, 0.2, 6]}], "motion_guess": {"type": "static", "conf": 0.8}, "contacts": []})
    prog = evidence_to_program(ev, "x", cfg)
    assert any(s["id"] == "floor_2" for s in prog["static"]) and all(o["id"] != "floor_2" for o in prog["objects"])
    bound = bind(copy.deepcopy(prog), ev, cfg)
    r = validate(bound); assert r["ok"], r["errors"]
    assert bound["binding"]["slots"]["player_spawn"] and bound["binding"]["slots"]["goal_volume"]
