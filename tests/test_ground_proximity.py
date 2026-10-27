"""Synthetic geometry only: no model, API, GPU or real-video observations."""
from copy import deepcopy

import pytest

from gwm.perception.ground_proximity import measure_ground_proximity
from gwm.perception.contract import adapt_legacy_evidence, validate_evidence
from gwm.perception.quality import assess_evidence_quality
from gwm.synthesis.direct import evidence_to_program
from gwm.synthesis.writer import evidence_summary
from gwm.compiler.repair import repair_candidate


GROUND = {"kind": "ground", "normal": [0, 1, 0], "offset": 0, "conf": .9,
          "center_hint": [0, 0, 0], "extent_hint": [10, .2, 10]}


def box(i, gap=0., height=1., t=None):
    return {"frame": i, "t": float(i if t is None else t),
            "center": [0, height / 2 + gap, 0], "size": [1, height, 1],
            "quat": [0, 0, 0, 1], "conf": .9}


def measure(obbs, **kw):
    return measure_ground_proximity(obbs, [GROUND], **kw)


def evidence(obbs=None):
    obbs = obbs if obbs is not None else [box(0), box(1), box(2)]
    return {"meta": {"clip": "synthetic", "duration": max(b["t"] for b in obbs), "scale": "relative",
                     "geometry_frames": [b["frame"] for b in obbs]},
            "camera": {"intrinsics": {"width": 100, "height": 80, "fov_deg": 60},
                       "poses": [{"t": b["t"], "pos": [0, 1, 5], "quat": [0, 0, 0, 1]} for b in obbs]},
            "static": {"planes": [deepcopy(GROUND)]},
            "objects": [{"id": "sample", "class_guess": "box", "confidence": .9,
                         "is_dynamic": False, "obb": deepcopy(obbs), "contacts": [],
                         "motion_guess": {"type": "static", "conf": .9}}], "keyframes": []}


def with_proximity(ev):
    ev["objects"][0]["ground_proximity"] = measure(
        ev["objects"][0]["obb"], scale=ev["meta"]["scale"])
    return ev


@pytest.mark.parametrize("height", [.2, 1., 10.])
def test_height_does_not_relax_tolerance(height):
    out = measure([box(0, gap=-.1, height=height)], config={"perception": {"contact_dist_m": .08}})
    assert out["tolerance_world"] == .05
    assert out["samples"][0]["state"] == "penetrating"
    assert out["confidence"] == out["physical_contact"] == "unknown"


@pytest.mark.parametrize("gap,state", [(-.1, "penetrating"), (-.02, "compatible"),
                                       (0, "compatible"), (.02, "compatible"), (.2, "separated")])
def test_sample_geometry_not_physical_contact(gap, state):
    out = measure([box(0, gap)])
    assert out["samples"][0]["state"] == state
    assert out["temporal_scope"] == "sample_points"
    assert "t_start" not in out["samples"][0]


def test_majority_does_not_expand_to_interval():
    obbs = [box(i, 0 if i < 19 else .2) for i in range(32)]
    out = measure(obbs)
    assert [s["t"] for s in out["samples"] if s["state"] == "compatible"] == list(range(19))
    assert all(s["state"] == "separated" for s in out["samples"][19:])
    assert all("t_end" not in s for s in out["samples"])


def test_missing_frames_and_large_intervals_remain_unobserved():
    obbs = [box(0), box(2, t=100)]
    frames = [{"frame_index": 0, "t": 0.}, {"frame_index": 1, "t": 50.},
              {"frame_index": 2, "t": 100.}]
    out = measure(obbs, frames=frames)
    assert [s["state"] for s in out["samples"]] == ["compatible", "unknown", "compatible"]
    assert out["samples"][1]["gap"] == "unknown" and out["status"] == "partial"
    # Without a registry, the two supported endpoints still create no interval.
    assert [s["t"] for s in measure(obbs)["samples"]] == [0, 100]


@pytest.mark.parametrize("case", ["missing", "ambiguous", "multiple_candidates", "low_conf", "unknown_conf", "invalid_conf", "bad_normal", "alignment_failed"])
def test_ground_uncertainty_does_not_create_labels(case):
    planes, alignment = [deepcopy(GROUND)], None
    if case == "missing": planes = []
    if case == "ambiguous": planes *= 2
    if case == "multiple_candidates": alignment = {"ground": "candidate_0", "ground_candidate_count": 2}
    if case == "low_conf": planes[0]["conf"] = .5
    if case == "unknown_conf": planes[0]["conf"] = "unknown"
    if case == "invalid_conf": planes[0]["conf"] = 2.
    if case == "bad_normal": planes[0]["normal"] = [0, 0, 0]
    if case == "alignment_failed": alignment = {"ground": "not_found"}
    out = measure_ground_proximity([box(0)], planes, alignment=alignment)
    assert out["status"] == "unknown" and out["issues"]
    assert out["samples"][0]["gap"] == out["samples"][0]["state"] == "unknown"


def test_threshold_changes_and_legacy_tighter_config():
    assert measure([box(0, .03)], config={"perception": {"contact_dist_m": .02}})["samples"][0]["state"] == "separated"
    cfg = {"perception": {"ground_proximity": {"max_gap_world": .04, "min_ground_confidence": .95}}}
    assert measure([box(0)], config=cfg)["status"] == "unknown"
    cfg["perception"]["ground_proximity"]["min_ground_confidence"] = .8
    assert measure([box(0, .03)], config=cfg)["samples"][0]["state"] == "compatible"


@pytest.mark.parametrize("value", [-1, float("nan"), True])
def test_bad_config_fails_explicitly(value):
    with pytest.raises(ValueError):
        measure([box(0)], config={"perception": {"ground_proximity": {"max_gap_world": value}}})


def test_inputs_unchanged_and_plane_offset_respected():
    obbs, planes = [box(0, -2)], [deepcopy(GROUND)]
    planes[0]["offset"] = 2
    before = deepcopy((obbs, planes))
    out = measure_ground_proximity(obbs, planes)
    assert (obbs, planes) == before and out["samples"][0]["gap"] == 0


def test_rotated_box_and_unusable_geometry():
    import math
    ob = box(0)
    ob["center"][1] = 1.
    ob["size"] = [1, 1, 2]
    ob["quat"] = [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]
    assert measure([ob])["samples"][0]["gap"] == pytest.approx(0)
    ob["quat"] = [0, 0, 0, 0]
    assert measure([ob])["samples"][0]["state"] == "unknown"


def test_legacy_read_and_translation_keep_existing_support_behavior():
    ev = evidence()
    ev["objects"][0]["contacts"] = [{"with_id": "ground", "t_start": 0, "t_end": 2, "conf": .9}]
    before = deepcopy(ev)
    assert evidence_to_program(ev, cfg={})["objects"][0]["support"] == "ground"
    assert ev == before


def test_recorded_alignment_ambiguity_cannot_be_hidden_in_known_labels():
    ev = with_proximity(evidence())
    ev["meta"]["alignment"] = {"ground": "candidate_0", "ground_candidate_count": 2}
    assert not validate_evidence(ev)["ok"]


def test_new_optional_contract_and_old_evidence_are_readable():
    old = evidence()
    old["objects"][0]["contacts"] = [{"with_id": "ground", "t_start": 0, "t_end": 2, "conf": .9}]
    before = deepcopy(old)
    assert validate_evidence(old)["ok"] and old == before
    new = with_proximity(evidence())
    before = deepcopy(new)
    assert validate_evidence(new)["ok"] and new == before
    assert new["objects"][0]["contacts"] == []


@pytest.mark.parametrize("change", ["confidence", "state", "frame", "time", "missing_obb", "gap", "ground"])
def test_contract_rejects_fabricated_or_misreferenced_samples(change):
    ev = with_proximity(evidence())
    proximity = ev["objects"][0]["ground_proximity"]
    if change == "confidence": proximity["confidence"] = 1.
    if change == "state": proximity["samples"][0]["state"] = "separated"
    if change == "frame": proximity["samples"][0]["frame_index"] = 99
    if change == "time": proximity["samples"][1]["t"] = 0
    if change == "missing_obb": ev["objects"][0]["obb"] = []
    if change == "gap": proximity["samples"][0]["gap"] = .01
    if change == "ground": ev["static"]["planes"] = []
    assert not validate_evidence(ev)["ok"]


def test_quality_records_proximity_counts_and_warns_not_fixes():
    ev = with_proximity(evidence([box(0, -.2), box(1), box(2, 2)]))
    before = deepcopy(ev)
    report = assess_evidence_quality(ev)
    assert report["decision"] == "warn" and ev == before
    counts = report["metrics"]["objects"][0]["ground_proximity"]["sample_counts"]
    assert counts == {"compatible": 1, "separated": 1, "penetrating": 1, "unknown": 0}
    assert "ground_geometry_conflict" in {d["code"] for d in report["diagnostics"]}


def test_direct_and_writer_do_not_promote_proximity_to_support():
    ev = with_proximity(evidence())
    before = deepcopy(ev)
    out = evidence_to_program(ev, cfg={})
    assert "support" not in out["objects"][0] and ev == before
    assert out["objects"][0]["pose"]["pos"] == ev["objects"][0]["obb"][0]["center"]
    assert out["objects"][0]["geom"]["extent"] == [1., 1., 1.]
    text = evidence_summary(adapt_legacy_evidence(ev)[0])
    assert "physical_contact=unknown" in text and "no continuous interval" in text
    # Explicit markers are also rejected if supplied by a separate consumer.
    ev["objects"][0]["contacts"] = [{"kind": "geometric_proximity", "with_id": "ground", "conf": .99}]
    assert "support" not in evidence_to_program(ev, cfg={})["objects"][0]


def test_repair_refuses_marked_proximity_even_with_numeric_confidence():
    from test_candidate_repair import _config, _evidence, _program
    ev, program = _evidence(), _program(.43)
    ev["objects"][0]["contacts"][0]["kind"] = "geometric_proximity"
    before = deepcopy((ev, program))
    out, report, unresolved = repair_candidate(program, ev, None, _config())
    assert out == program and (ev, program) == before
    assert not report["repairs"] and unresolved
    assert any("geometric proximity" in d["reason"] for d in unresolved)


def test_new_proximity_cannot_snap_with_no_physical_contacts():
    from test_candidate_repair import _config, _evidence, _program
    ev, program = _evidence(), _program(.43)
    ev["objects"][0]["contacts"] = []
    ev["static"]["planes"] = [deepcopy(GROUND)]
    for i, b in enumerate(ev["objects"][0]["obb"]):
        b["frame"] = i
    ev["objects"][0]["ground_proximity"] = measure(
        ev["objects"][0]["obb"], scale=ev["meta"]["scale"])
    out, report, unresolved = repair_candidate(program, ev, None, _config())
    assert out == program and not report["repairs"] and unresolved


def test_producer_uses_new_contract_without_changing_obb_or_motion(tmp_path, monkeypatch):
    from test_observed_color import built
    from gwm.perception import evidence as producer
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir(); second_dir.mkdir()
    first = built(first_dir, monkeypatch)
    original = producer.measure_ground_proximity
    called = []
    def spy(obbs, *args, **kwargs):
        before = deepcopy(obbs)
        result = original(obbs, *args, **kwargs)
        assert obbs == before
        called.append(before)
        return result
    monkeypatch.setattr(producer, "measure_ground_proximity", spy)
    second = built(second_dir, monkeypatch)
    assert called and validate_evidence(second)["ok"]
    a, b = first["objects"][0], second["objects"][0]
    assert b["contacts"] == [] and b["ground_proximity"]["physical_contact"] == "unknown"
    for key in ("id", "obb", "motion_guess", "hypotheses"):
        assert a[key] == b[key]
