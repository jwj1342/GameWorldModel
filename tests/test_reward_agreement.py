"""Offline statistical checks; never run models, browser harnesses or real experiments."""
import copy
import json
import math
import sys

import pytest

from gwm.config import REPO

sys.path.insert(0, str(REPO / "scripts"))
import check_reward_agreement as agreement


def rows(rewards=(1., .5, 0.), errors=(0., 1., 2.)):
    return [{"family": "synthetic", "label": str(i), "reward_no_gt": r, "gt_traj_median_m": e,
             "reward_status": "available", "gt_status": "available"}
            for i, (r, e) in enumerate(zip(rewards, errors))]


def test_valid_zero_is_kept_and_inputs_unchanged():
    data = rows(); before = copy.deepcopy(data)
    result = agreement.assess_agreement(data)
    assert result["status"] == "available" and result["n_comparable"] == 3
    assert result["spearman_rho"] == -1 and result["included_indices"] == [0, 1, 2]
    assert not result["excluded"] and data == before


@pytest.mark.parametrize("field,values,reason", [
    ("reward_no_gt", (0., 0., 0.), "constant_reward"),
    ("gt_traj_median_m", (1., 1., 1.), "constant_gt_error"),
])
def test_constant_columns_unavailable_not_nan(field, values, reason):
    data = rows()
    for row, value in zip(data, values): row[field] = value
    result = agreement.assess_agreement(data)
    assert result["reason"] == reason and result["spearman_rho"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field", ["reward_no_gt", "gt_traj_median_m"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def test_nonfinite_or_missing_excluded_explicitly(field, value):
    data = rows(); data[2][field] = value
    result = agreement.assess_agreement(data)
    assert result["n_comparable"] == 2 and result["reason"] == "insufficient_samples"
    assert result["excluded"][0]["reasons"] == [field + ":non_finite_or_missing"]
    json.dumps(result, allow_nan=False)


def test_unknown_legacy_status_is_not_inferred_from_numbers():
    data = rows()
    del data[0]["reward_status"]
    data[1]["gt_status"] = "execution_failed"
    result = agreement.assess_agreement(data)
    assert result["n_comparable"] == 1
    assert "reward_status:unknown" in result["excluded"][0]["reasons"]
    assert "gt_status:execution_failed" in result["excluded"][1]["reasons"]


def test_families_cannot_be_pooled():
    data = rows(); data[1]["family"] = "other"
    with pytest.raises(ValueError, match="pooled"):
        agreement.assess_agreement(data)


def test_ties_and_small_unrounded_values():
    result = agreement.assess_agreement(rows((.000001, .000002, .000002), (3., 1., 1.)))
    assert result["status"] == "available" and math.isclose(result["spearman_rho"], -1)


def test_minimum_samples_is_configurable():
    assert agreement.assess_agreement(rows(), min_samples=4)["reason"] == "insufficient_samples"
    with pytest.raises(ValueError): agreement.assess_agreement(rows(), min_samples=2)


@pytest.mark.parametrize("objects,score,expected", [
    ({"target": {"program_id": "target"}}, 0., (0., "available")),
    ({}, 0., (None, "no_observations")),
    ({"target": {"program_id": None}}, 0., (None, "identity_unresolved")),
    ({"target": {"program_id": "target"}}, float("nan"), (None, "non_finite_score")),
])
def test_reward_availability_is_not_a_positive_score_filter(objects, score, expected):
    assert agreement.reward_measurement({"summary": {"per_object": objects, "score": score}}) == expected


def test_perturbation_is_not_a_coordinate_origin_change():
    program = {"objects": [{"pose": {"pos": [0, 0, 0]}, "motion": {"type": "spin"}}],
               "static": [{"pose": {"pos": [0, 0, 0]}}], "camera": {"pos": [1, 2, 3]}}
    before = copy.deepcopy(program)
    candidates = agreement.graded_candidates(program)
    label, moved = candidates[1]
    assert label.startswith("对象整体平移") and moved["objects"][0]["pose"]["pos"][0] == .15
    assert moved["static"] == program["static"] and moved["camera"] == program["camera"]
    assert program == before


def cli_source(tmp_path, monkeypatch):
    import gwm.feedback.metrics as metrics
    import gwm.feedback.render as renderer
    import gwm.perception.run as perception
    program = {"meta": {"duration": 6}, "static": [], "objects": []}
    run = tmp_path / "run"; (run / "perception").mkdir(parents=True)
    (run / "program.json").write_text(json.dumps(program), encoding="utf-8")
    from gwm.perception.provenance import file_sha256
    video = tmp_path / "video.bin"
    video.write_bytes(b"SYNTHETIC TEST BYTES, NOT A REAL VIDEO")
    evidence = run / "perception/evidence.json"
    evidence.write_text(json.dumps({"meta": {"source_video_sha256": file_sha256(video)},
                                    "frames": [{"frame_index": 0, "t": 0}], "keyframes": [{"index": 0, "t": 0}]}), encoding="utf-8")
    reference = tmp_path / "examples/handwritten"; reference.mkdir(parents=True)
    (reference / "program.json").write_text(json.dumps(program), encoding="utf-8")
    record = tmp_path / "generation.json"
    record.write_text(json.dumps({"synthetic_test": True}), encoding="utf-8")
    manifest = tmp_path / "associations.json"
    data = {"schema_version": 1, "associations": [{"association_id": "synthetic_test",
        "video": {"path": "video.bin", "sha256": file_sha256(video)},
        "ground_truth": {"path": "examples/handwritten/program.json", "sha256": file_sha256(reference / "program.json")},
        "time_mapping": {"video_start_s": 0, "program_start_s": 0, "duration_s": 6, "rate": 1},
        "provenance": {"kind": "synthetic_generation", "record_path": "generation.json", "record_sha256": file_sha256(record)},
        "review": {"status": "confirmed", "note": "Synthetic unit-test attestation only"}}]}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(agreement, "REPO", tmp_path)
    monkeypatch.setattr(agreement, "load_config", lambda *a: {"feedback": {"render_width": 4, "render_height": 4, "passes": []}})
    monkeypatch.setattr(agreement.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(agreement, "load_states", lambda *a: [{"t": i / 10} for i in range(61)])
    monkeypatch.setattr(renderer, "render", lambda *a: {})
    monkeypatch.setattr(perception, "load_masks", lambda *a: {})
    monkeypatch.setattr(metrics, "compute_metrics", lambda *a: {"summary": {"score": 0., "per_object": {}}})
    monkeypatch.setattr(agreement, "evaluate", lambda *a: {"full": {"trajectory": {"median_m": None}, "motion_type": {"accuracy": None}}})
    out = tmp_path / "report.json"
    args = ["--run", str(run), "--work", str(tmp_path / "work"), "--out", str(out),
            "--video", str(video), "--associations", str(manifest), "--duration", "6"]
    return args, out, run, video, manifest, data


def test_cli_reports_only_exploratory_statistics_with_mocked_execution(tmp_path, monkeypatch):
    args, out, *_ = cli_source(tmp_path, monkeypatch)
    assert agreement.main(args) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["protocol_status"] == "not_verified" and report["spearman_rho"] is None
    assert report["source_verification"]["status"] == "verified"
    assert report["n_comparable"] == 0 and len(report["agreement_by_family"]) == 2
    assert all(r["reward_no_gt"] is None and r["reward_raw"] == 0 for r in report["rows"])
    assert all(m["单调下降"] is None for m in report["monotonicity"].values())
    assert report["conclusion"].startswith("目前无法判断")
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["missing_args", "wrong_video", "pending", "changed_program", "wrong_evidence", "frame_time", "short_prediction", "self_reference"])
def test_source_rejection_precedes_any_execution(tmp_path, monkeypatch, failure):
    args, out, run, video, manifest, data = cli_source(tmp_path, monkeypatch)
    if failure == "missing_args": args = args[:6]
    if failure == "wrong_video": video.write_bytes(b"DIFFERENT SYNTHETIC VIDEO")
    if failure == "pending":
        data["associations"][0]["review"]["status"] = "pending"
        manifest.write_text(json.dumps(data), encoding="utf-8")
    if failure == "changed_program":
        (tmp_path / "examples/handwritten/program.json").write_text("{}", encoding="utf-8")
    if failure in ("wrong_evidence", "frame_time"):
        path = run / "perception/evidence.json"
        ev = json.loads(path.read_text(encoding="utf-8"))
        if failure == "wrong_evidence": ev["meta"]["source_video_sha256"] = "a" * 64
        else: ev["keyframes"][0]["t"] = 7
        path.write_text(json.dumps(ev), encoding="utf-8")
    if failure == "short_prediction":
        (run / "program.json").write_text(json.dumps({"meta": {"duration": 1}}), encoding="utf-8")
    if failure == "self_reference":
        from gwm.perception.provenance import file_sha256
        data["associations"][0]["ground_truth"] = {"path": "run/program.json", "sha256": file_sha256(run / "program.json")}
        manifest.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(agreement.subprocess, "run", lambda *a, **k: pytest.fail("execution before source gate"))
    assert agreement.main(args) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "source_rejected" and report["rows"] == []
    assert report["diagnostics"][0]["severity"] == "error"


@pytest.mark.parametrize("stage", ["candidate_compile", "candidate_render", "reward_metrics", "candidate_record", "gt_metrics"])
def test_candidate_failure_retained_with_stage_and_nonzero_exit(tmp_path, monkeypatch, stage):
    import gwm.feedback.metrics as metrics
    import gwm.feedback.render as renderer
    args, out, *_ = cli_source(tmp_path, monkeypatch)
    def fail(*a, **k): raise RuntimeError("synthetic failure")
    if stage == "candidate_render": monkeypatch.setattr(renderer, "render", fail)
    if stage == "reward_metrics": monkeypatch.setattr(metrics, "compute_metrics", fail)
    if stage == "gt_metrics": monkeypatch.setattr(agreement, "evaluate", fail)
    if stage in ("candidate_compile", "candidate_record"):
        def execution(command, **kwargs):
            selected = "c" in next((p for p in command if "/cp" in p.replace("\\", "/") or "/cg" in p.replace("\\", "/")), "")
            if selected and ((stage == "candidate_compile") == ("-m" in command)):
                fail()
        monkeypatch.setattr(agreement.subprocess, "run", execution)
    assert agreement.main(args) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "partial_execution_failure" and len(report["rows"]) == 12
    assert all(r["diagnostics"][0]["stage"] == stage for r in report["rows"])
    assert all(r["gt_traj_median_m"] is None and r["reward_no_gt"] is None for r in report["rows"])
    assert report["n_comparable"] == 0


def test_empty_or_short_reference_record_is_execution_failure(tmp_path, monkeypatch):
    args, out, *_ = cli_source(tmp_path, monkeypatch)
    monkeypatch.setattr(agreement, "load_states", lambda *a: [{"t": 0}, {"t": 1}])
    assert agreement.main(args) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "execution_failed" and report["rows"] == []
    assert report["diagnostics"][0]["stage"] == "reference_record"


@pytest.mark.parametrize("changed", ["video", "evidence", "prediction"])
def test_input_changed_during_measurement_invalidates_statistics(tmp_path, monkeypatch, changed):
    import gwm.feedback.metrics as metrics
    args, out, run, video, *_ = cli_source(tmp_path, monkeypatch)
    def measured(*a):
        if changed == "video": video.write_bytes(b"CHANGED DURING SYNTHETIC ACQUISITION")
        else:
            path = run / ("perception/evidence.json" if changed == "evidence" else "program.json")
            content = json.loads(path.read_text(encoding="utf-8"))
            content["synthetic_revision"] = True
            path.write_text(json.dumps(content), encoding="utf-8")
        return {"summary": {"score": 1., "per_object": {"target": {"program_id": "target"}}}}
    monkeypatch.setattr(metrics, "compute_metrics", measured)
    assert agreement.main(args) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "source_rejected" and report["n_comparable"] == 0
    assert all(r["reward_status"] == "source_rejected" and r["reward_no_gt"] is None for r in report["rows"])


def test_existing_report_not_overwritten(tmp_path, monkeypatch):
    args, out, *_ = cli_source(tmp_path, monkeypatch)
    out.write_text("old report", encoding="utf-8")
    with pytest.raises(SystemExit): agreement.main(args)
    assert out.read_text(encoding="utf-8") == "old report"


def test_structured_frame_reference_reaches_real_reward_function(tmp_path):
    from PIL import Image
    import numpy as np
    from gwm.compiler.ids import id_to_color
    from gwm.feedback.metrics import compute_metrics
    ev = {"frames": [{"frame_index": 0, "t": 0}], "keyframes": [{"index": 0, "t": 0}],
          "objects": [{"id": "target", "obb": []}]}
    before = copy.deepcopy(ev)
    frames = agreement.metric_frames(ev)
    Image.new("RGB", (2, 2), id_to_color(1)).save(tmp_path / "id.png")
    index = {"width": 2, "height": 2, "frames": [{"t": 0, "files": {"id": "id.png"}, "state": {"objects": []}}]}
    cfg = {"feedback": {"thresholds": {"iou_fail": .2, "centroid_px_frac_fail": .2, "ate_m_fail": 1},
                        "weights": {"iou": 1, "centroid": 0, "ate": 0, "dino": 0}}}
    measured = compute_metrics({"objects": [{"id": "target"}], "static": []}, index, tmp_path,
                               ev, frames, {"target": {0: np.ones((2, 2), bool)}}, cfg)
    assert agreement.reward_measurement(measured) == (1., "available")
    assert ev == before  # Adapter does not rewrite source Evidence.


@pytest.mark.parametrize("failure", ["no_index", "conflict", "unknown_ref", "time_mismatch", "duplicate"])
def test_frame_adapter_refuses_fabricated_or_inconsistent_references(failure):
    ev = {"frames": [{"frame_index": 0, "t": 0}], "keyframes": [{"index": 0, "t": 0}]}
    if failure == "no_index": ev["frames"][0].pop("frame_index")
    if failure == "conflict": ev["frames"][0]["index"] = 1
    if failure == "unknown_ref": ev["keyframes"][0]["index"] = 9
    if failure == "time_mismatch": ev["keyframes"][0]["t"] = 1
    if failure == "duplicate": ev["frames"].append(copy.deepcopy(ev["frames"][0]))
    with pytest.raises(agreement.SourceRejected): agreement.metric_frames(ev)


@pytest.mark.parametrize("times", [[0, 6], [i / 10 for i in range(60)], [i / 10 for i in range(61) if i != 20]])
def test_recording_must_include_endpoint_and_interior_sampling(tmp_path, monkeypatch, times):
    args, out, *_ = cli_source(tmp_path, monkeypatch)
    monkeypatch.setattr(agreement, "load_states", lambda *a: [{"t": t} for t in times])
    assert agreement.main(args) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "execution_failed" and report["diagnostics"][0]["stage"] == "reference_record"


def test_record_invocation_explicitly_requests_endpoint(tmp_path, monkeypatch):
    args, _, *_ = cli_source(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(agreement.subprocess, "run", lambda command, **kwargs: calls.append(command))
    assert agreement.main(args) == 0
    records = [c for c in calls if "record.mjs" in c]
    assert len(records) == 13 and all("--include-endpoint" in c for c in records)
