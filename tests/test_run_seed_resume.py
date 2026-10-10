"""Synthetic orchestration records only; no model, video decoder or provider."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from gwm import run_clip
from gwm.config import redact
from gwm.perception.provenance import file_sha256


@pytest.fixture
def inputs(tmp_path):
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"synthetic source bytes, not a decoded video")
    args = SimpleNamespace(video=str(video), clip="synthetic", resume=None, seed=None,
                           no_vlm=True, phrases=None, prompt=None, out=None, config=[])
    cfg = {"vlm": {"seed": 123}, "paths": {"out": str(tmp_path / "runs")}}
    return tmp_path / "run", args, cfg


def existing(inputs, monkeypatch):
    directory, args, cfg = inputs
    monkeypatch.setattr(run_clip, "git_commit", lambda: "original-commit")
    original = run_clip.prepare_run_manifest(directory, args, cfg, cfg["vlm"]["seed"])
    original["started_at"] = "original-start"
    original["stages"] = {"perception": {"seed": 123, "marker": "original-stage"}}
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    (directory / "perception").mkdir()
    evidence = {"meta": {"source_video_sha256": file_sha256(args.video)}, "marker": "original-evidence"}
    (directory / "perception/evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    original["stages"]["perception"]["evidence_sha256"] = file_sha256(directory / "perception/evidence.json")
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    args.resume = str(directory)
    return original


@pytest.mark.parametrize("config_seed,cli_seed,expected", [(None, None, None), (123, None, 123),
                                                          (123, 99, 99), (123, 0, 0), (0, None, 0)])
def test_effective_seed_precedence(config_seed, cli_seed, expected):
    cfg = {"vlm": {"seed": config_seed}}
    assert run_clip.resolve_seed(cfg, cli_seed) == expected
    assert cfg["vlm"]["seed"] == expected


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "123"])
def test_invalid_config_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        run_clip.resolve_seed({"vlm": {"seed": seed}})


class StopAtQuality(Exception):
    pass


def invoke(monkeypatch, directory, args, cfg, cli_seed=None, resume=False):
    monkeypatch.setattr(run_clip, "load_config", lambda *a: copy.deepcopy(cfg))
    seen = {}
    def perception(*a, **kwargs):
        seen["draws"] = [random.random(), float(np.random.random())]
        return {"synthetic": True}
    def quality(ev, config):
        seen["evidence"] = ev
        seen["seed"] = config["vlm"]["seed"]
        raise StopAtQuality()
    monkeypatch.setattr(run_clip, "run_perception", perception)
    monkeypatch.setattr(run_clip, "assess_evidence_quality", quality)
    command = ["--video", args.video, "--clip", args.clip, "--no-vlm",
               "--resume" if resume else "--out", str(directory)]
    if cli_seed is not None:
        command += ["--seed", str(cli_seed)]
    with pytest.raises(StopAtQuality):
        run_clip.main(command)
    return seen, json.loads((directory / "run.json").read_text(encoding="utf-8"))


def test_configuration_and_cli_initialize_identical_rngs(inputs, monkeypatch):
    directory, args, cfg = inputs
    first, manifest = invoke(monkeypatch, directory, args, cfg)
    second, other = invoke(monkeypatch, directory.parent / "other", args, {**cfg, "vlm": {"seed": None}}, cli_seed=123)
    assert first["draws"] == second["draws"]
    assert manifest["seed"] == other["seed"] == 123


def test_cli_seed_zero_initializes_both_rngs(inputs, monkeypatch):
    directory, args, cfg = inputs
    seen, manifest = invoke(monkeypatch, directory, args, cfg, cli_seed=0)
    assert seen["seed"] == manifest["seed"] == 0
    assert seen["draws"] == [random.Random(0).random(), float(np.random.RandomState(0).random_sample())]


def test_no_seed_does_not_reset_rngs(inputs, monkeypatch):
    directory, args, cfg = inputs
    cfg["vlm"]["seed"] = None
    monkeypatch.setattr(random, "seed", lambda *a: pytest.fail("unexpected random reseed"))
    monkeypatch.setattr(np.random, "seed", lambda *a: pytest.fail("unexpected numpy reseed"))
    _, manifest = invoke(monkeypatch, directory, args, cfg)
    assert manifest["seed"] is None


def test_same_resume_keeps_origin_and_stage_history(inputs, monkeypatch):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    evidence_bytes = (directory / "perception/evidence.json").read_bytes()
    monkeypatch.setattr(run_clip, "git_commit", lambda: "fix-commit")
    _, manifest = invoke(monkeypatch, directory, args, cfg, resume=True)
    for field in ("run_id", "started_at", "git_commit", "config", "args", "seed", "video_sha256"):
        assert manifest[field] == original[field]
    attempt = manifest["resume_attempts"][0]
    assert attempt["git_commit"] == "fix-commit"
    assert attempt["previous_stages"] == original["stages"]
    assert manifest["stages"]["perception"]["status"] == "reused"
    assert attempt["status"] == manifest["status"] == "failed"
    assert attempt["reused_artifacts"]["perception/evidence.json"] == file_sha256(directory / "perception/evidence.json")
    assert (directory / "perception/evidence.json").read_bytes() == evidence_bytes


@pytest.mark.parametrize("change", ["seed", "clip", "video_path", "video_content", "config", "no_vlm", "phrases", "prompt"])
def test_resume_mismatch_does_not_write_manifest_or_evidence(inputs, monkeypatch, change):
    existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    before = (directory / "run.json").read_bytes()
    evidence = (directory / "perception/evidence.json").read_bytes()
    if change == "seed": cfg["vlm"]["seed"] = 99
    elif change == "clip": args.clip = "another"
    elif change == "video_path":
        video = directory.parent / "different.mp4"
        video.write_bytes(b"other source")
        args.video = str(video)
    elif change == "video_content":
        from pathlib import Path
        Path(args.video).write_bytes(b"changed source under same filename")
    elif change == "config": cfg["vlm"]["temperature"] = .9
    else: setattr(args, change, False if change == "no_vlm" else "changed")
    with pytest.raises(ValueError, match="resume"):
        run_clip.prepare_run_manifest(directory, args, cfg, cfg["vlm"]["seed"])
    assert (directory / "run.json").read_bytes() == before
    assert (directory / "perception/evidence.json").read_bytes() == evidence


def test_changed_seed_cli_is_rejected_before_reusing_evidence(inputs, monkeypatch):
    existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    before = (directory / "run.json").read_bytes()
    monkeypatch.setattr(run_clip, "load_config", lambda *a: copy.deepcopy(cfg))
    monkeypatch.setattr(run_clip, "run_perception", lambda *a, **k: pytest.fail("provider must not run"))
    with pytest.raises(SystemExit) as result:
        run_clip.main(["--video", args.video, "--clip", args.clip, "--no-vlm", "--resume", str(directory), "--seed", "99"])
    assert result.value.code == 3          # 3 是用法错误；2 留给证据质量 block 和编译失败
    assert (directory / "run.json").read_bytes() == before


def test_resume_survives_a_source_video_that_is_no_longer_on_disk(inputs, monkeypatch):
    """作业超时之后靠 --resume 续跑是文档承诺的恢复路径，而 scratch 是 60 天轮转的，
    clip 也可能被 stage 到 $SLURM_TMPDIR。缓存的 Evidence 自带 source_video_sha256，
    来源有据可查，不该因为此刻读不到视频文件就把一次完好的运行判死。"""
    existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    from pathlib import Path
    Path(args.video).unlink()
    manifest = run_clip.prepare_run_manifest(directory, args, cfg, cfg["vlm"]["seed"])
    assert manifest["stages"]["perception"]["status"] == "reused"
    assert manifest["resume_attempts"][-1]["reused_artifacts"]["perception/evidence.json"]


def test_source_hash_can_be_verified_from_evidence(inputs, monkeypatch):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    del original["video_sha256"]
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    manifest = run_clip.prepare_run_manifest(directory, args, cfg, 123)
    assert len(manifest["resume_attempts"]) == 1


@pytest.mark.parametrize("fault", ["missing_manifest", "missing_source_hash", "bad_stages", "bad_attempts"])
def test_incomplete_resume_is_refused(inputs, monkeypatch, fault):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    if fault == "missing_manifest":
        (directory / "run.json").unlink()
    else:
        if fault == "missing_source_hash":
            del original["video_sha256"]
            (directory / "perception/evidence.json").write_text('{"meta": {}}', encoding="utf-8")
        elif fault == "bad_stages": original["stages"] = []
        else: original["resume_attempts"] = {}
        (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError):
        run_clip.prepare_run_manifest(directory, args, cfg, 123)


def test_existing_output_requires_explicit_resume(inputs, monkeypatch):
    existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    args.resume = None
    with pytest.raises(ValueError, match="existing run"):
        run_clip.prepare_run_manifest(directory, args, cfg, 123)


def test_manifest_retains_redaction(inputs):
    directory, args, cfg = inputs
    cfg["vlm"]["api_key"] = "synthetic-sensitive-marker"
    manifest = run_clip.prepare_run_manifest(directory, args, cfg, 123)
    assert "synthetic-sensitive-marker" not in json.dumps(manifest)
    assert manifest["config"] == redact(cfg)


def test_config_identity_preserves_keyframe_settings_but_excludes_credentials(inputs):
    _, _, cfg = inputs
    cfg["frames"] = {"keyframes": 12}
    cfg["vlm"]["api_key"] = "synthetic-key-one"
    original = run_clip.configuration_identity(cfg)
    cfg["vlm"]["api_key"] = "synthetic-key-two"
    assert run_clip.configuration_identity(cfg) == original
    cfg["frames"]["keyframes"] = 6
    assert run_clip.configuration_identity(cfg) != original


def test_legacy_manifest_without_config_identity_is_refused(inputs, monkeypatch):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    del original["config_identity"]
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    before = (directory / "run.json").read_bytes()
    with pytest.raises(ValueError, match="config identity"):
        run_clip.prepare_run_manifest(directory, args, cfg, 123)
    assert (directory / "run.json").read_bytes() == before


@pytest.mark.parametrize("change", ["source", "content"])
def test_cached_evidence_source_and_recorded_content_are_checked(inputs, monkeypatch, change):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    evidence_path = directory / "perception/evidence.json"
    original["stages"]["perception"]["evidence_sha256"] = file_sha256(evidence_path)
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if change == "source": evidence["meta"]["source_video_sha256"] = "wrong-source"
    else: evidence["marker"] = "different-observations"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    before = (directory / "run.json").read_bytes()
    with pytest.raises(ValueError, match="Evidence"):
        run_clip.prepare_run_manifest(directory, args, cfg, 123)
    assert (directory / "run.json").read_bytes() == before


@pytest.mark.parametrize("malformed", [[], {"meta": []}])
def test_malformed_evidence_metadata_is_refused(inputs, monkeypatch, malformed):
    existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    (directory / "perception/evidence.json").write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence metadata"):
        run_clip.prepare_run_manifest(directory, args, cfg, 123)


def test_stray_artifacts_do_not_block_a_new_run(inputs):
    """只有 run.json 才代表一次有来源记录的运行。目录里剩下的零散产物不该让新运行直接失败，
    否则任何固定了 --out 的重投脚本第二次就跑不起来。"""
    directory, args, cfg = inputs
    directory.mkdir()
    orphan = directory / "program.json"
    orphan.write_text("synthetic old artifact", encoding="utf-8")
    manifest = run_clip.prepare_run_manifest(directory, args, cfg, 123)
    assert manifest["seed"] == 123 and manifest["stages"] == {}
    assert orphan.read_text(encoding="utf-8") == "synthetic old artifact"


@pytest.mark.parametrize("fault", ["missing_hash", "empty_hash", "missing_artifact"])
def test_unverifiable_cached_artifact_is_rejected_without_writes(inputs, monkeypatch, fault):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    if fault == "missing_artifact":
        (directory / "perception/evidence.json").unlink()
    else:
        original["stages"]["perception"]["evidence_sha256"] = None if fault == "missing_hash" else ""
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    before = (directory / "run.json").read_bytes()
    with pytest.raises(ValueError, match="Evidence"):
        run_clip.prepare_run_manifest(directory, args, cfg, 123)
    assert (directory / "run.json").read_bytes() == before


def test_credential_file_rotation_can_resume(inputs, monkeypatch):
    directory, args, cfg = inputs
    cfg["vlm"]["api_key_file"] = "synthetic-original-path"
    original = existing(inputs, monkeypatch)
    cfg["vlm"]["api_key_file"] = "synthetic-replacement-path"
    manifest = run_clip.prepare_run_manifest(directory, args, cfg, 123)
    assert manifest["config"] == original["config"]
    cfg["frames"] = {"keyframes": 99}
    with pytest.raises(ValueError, match="config identity"):
        run_clip.prepare_run_manifest(directory, args, cfg, 123)


@pytest.mark.parametrize("failure", [None, RuntimeError("synthetic failure"), SystemExit(2), KeyboardInterrupt()])
def test_attempt_outcome_does_not_relabel_old_success(inputs, monkeypatch, failure):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    original["stages"]["playtest"] = {"win": True}
    original.update(status="failed", error_type="SystemExit", exit_code=99)
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(run_clip, "load_config", lambda *a: copy.deepcopy(cfg))
    def execute(a, cfg, directory, manifest):
        (directory / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        manifest["stages"]["evidence_quality"] = {"decision": "block" if failure else "proceed"}
        if failure is not None:
            raise failure
    monkeypatch.setattr(run_clip, "_execute_run", execute)
    command = ["--video", args.video, "--clip", args.clip, "--no-vlm", "--resume", str(directory)]
    if failure is None:
        run_clip.main(command)
    else:
        with pytest.raises(type(failure)):
            run_clip.main(command)
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    attempt = manifest["resume_attempts"][-1]
    assert manifest["status"] == attempt["status"] == ("failed" if failure else "completed")
    assert "playtest" not in manifest["stages"]
    assert attempt["previous_stages"]["playtest"]["win"] is True
    assert attempt["stages"] == manifest["stages"]
    assert attempt["previous_result"]["status"] == "failed"
    assert attempt["previous_result"]["error_type"] == "SystemExit"
    assert attempt["previous_result"]["exit_code"] == 99
    for record in (manifest, attempt):
        if failure is None:
            assert "error_type" not in record and "exit_code" not in record
        else:
            assert record["error_type"] == type(failure).__name__
            if isinstance(failure, SystemExit):
                assert record["exit_code"] == 2
            else:
                assert "exit_code" not in record


def test_generated_hash_persisted_before_quality_failure(inputs, monkeypatch):
    directory, args, cfg = inputs
    monkeypatch.setattr(run_clip, "load_config", lambda *a: copy.deepcopy(cfg))
    def perception(video, clip, pdir, *a, **kw):
        pdir.mkdir()
        evidence = {"meta": {"source_video_sha256": file_sha256(video)}}
        (pdir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        return evidence
    def quality(*a):
        raise RuntimeError("synthetic quality exception")
    monkeypatch.setattr(run_clip, "run_perception", perception)
    monkeypatch.setattr(run_clip, "assess_evidence_quality", quality)
    with pytest.raises(RuntimeError):
        run_clip.main(["--video", args.video, "--clip", args.clip, "--no-vlm", "--out", str(directory)])
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["stages"]["perception"]["evidence_sha256"] == file_sha256(directory / "perception/evidence.json")
    args.resume = str(directory)
    assert run_clip.prepare_run_manifest(directory, args, cfg, 123)["resume_attempts"]


def test_quality_block_records_failure_without_old_playtest_success(inputs, monkeypatch):
    original = existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    original["stages"]["playtest"] = {"win": True}
    (directory / "run.json").write_text(json.dumps(original), encoding="utf-8")
    report = {"validation": {"ok": False, "adapted": False, "errors": [], "warnings": []},
              "decision": "block", "metrics": {}, "diagnostics": []}
    monkeypatch.setattr(run_clip, "load_config", lambda *a: copy.deepcopy(cfg))
    monkeypatch.setattr(run_clip, "assess_evidence_quality", lambda *a: report)
    monkeypatch.setattr(run_clip, "format_evidence_errors", lambda *a: "synthetic block")
    monkeypatch.setattr(run_clip, "run_perception", lambda *a, **k: pytest.fail("must reuse verified cache"))
    with pytest.raises(SystemExit) as result:
        run_clip.main(["--video", args.video, "--clip", args.clip, "--no-vlm", "--resume", str(directory)])
    assert result.value.code == 2
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["stages"]["evidence_quality"]["decision"] == "block"
    assert "playtest" not in manifest["stages"]
    assert manifest["resume_attempts"][-1]["previous_stages"]["playtest"]["win"] is True
    assert manifest["resume_attempts"][-1]["exit_code"] == 2


def test_reused_status_survives_successful_quality_and_stage_summary(inputs, monkeypatch):
    existing(inputs, monkeypatch)
    directory, args, cfg = inputs
    pdir = directory / "perception"
    evidence = {"meta": {"source_video_sha256": file_sha256(args.video),
                          "geometry_backend": "synthetic", "fallbacks": []},
                "objects": [], "keyframes": []}
    (pdir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    manifest["stages"]["perception"]["evidence_sha256"] = file_sha256(pdir / "evidence.json")
    (directory / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pdir / "frames").mkdir()
    (pdir / "frames/frames.json").write_text('{"frames": []}', encoding="utf-8")
    monkeypatch.setattr(run_clip, "load_config", lambda *a: copy.deepcopy(cfg))
    monkeypatch.setattr(run_clip, "load_masks", lambda *a: {})
    def quality(ev, cfg):
        return {"evidence": ev, "validation": {"ok": True, "adapted": False, "errors": [], "warnings": []},
                "decision": "proceed", "metrics": {}, "diagnostics": []}
    monkeypatch.setattr(run_clip, "assess_evidence_quality", quality)
    def stop(*a):
        raise RuntimeError("synthetic stop before generation")
    monkeypatch.setattr(run_clip, "evidence_to_program", stop)
    monkeypatch.setattr(run_clip, "run_perception", lambda *a, **k: pytest.fail("must not rerun provider"))
    with pytest.raises(RuntimeError):
        run_clip.main(["--video", args.video, "--clip", args.clip, "--no-vlm", "--resume", str(directory)])
    after = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert after["stages"]["evidence_quality"]["decision"] == "proceed"
    assert after["stages"]["perception"]["status"] == "reused"
    assert after["resume_attempts"][-1]["stages"]["perception"]["status"] == "reused"
    assert after["resume_attempts"][-1]["reused_artifacts"]["perception/evidence.json"] == file_sha256(pdir / "evidence.json")
