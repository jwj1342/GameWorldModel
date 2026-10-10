"""Pure-Python wiring tests with fake clients/packaging, not model experiments."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from gwm.config import REPO

sys.path.insert(0, str(REPO / "scripts"))
import run_direct_code as direct


def inputs(tmp_path, count=12):
    folder = tmp_path / "frames"
    folder.mkdir()
    frames, keyframes = [], []
    for i in range(count):
        filename = f"{'z' if i % 2 == 0 else 'a'}_{i:02d}.png"
        path = folder / filename
        Image.new("RGB", (4, 4), (i * 10, 20, 30)).save(path)
        frames.append({"frame_index": i, "t": i * .4, "source": {"kind": "video_frame", "ref": filename}})
        keyframes.append({"index": i, "t": i * .4, "file_small": str(path)})
    return folder, {
        "schema_version": "2.0", "meta": {"clip": "synthetic_source", "duration": 6, "scale": "relative"},
        "frames": frames, "keyframes": keyframes,
        "camera": {"intrinsics": {"width": 640, "height": 360, "fov_deg": 60},
                   "poses": [{"t": 0, "pos": [1, 2, 6], "quat": [0, 0, 0, 1], "conf": "unknown"}]},
        "static": {"planes": []}, "objects": [],
    }


def test_declared_12_frames_metadata_camera_and_order_preserved(tmp_path):
    folder, ev = inputs(tmp_path)
    before = copy.deepcopy(ev)
    _, selected, stub = direct.prepare_direct_inputs(ev, folder)
    assert len(selected) == 12
    assert [f["t"] for f in selected] == [k["t"] for k in ev["keyframes"]]
    assert [Path(f["file"]).name for f in selected] != sorted(Path(f["file"]).name for f in selected)
    assert stub["meta"]["clip"] == "synthetic_source" and stub["meta"]["duration"] == 6
    assert stub["camera"]["keyframes"][0]["pos"] == [1, 2, 6]
    assert all(f["sha256"] == hashlib.sha256(Path(f["file"]).read_bytes()).hexdigest() for f in selected)
    assert not any(f["declared_hash_verified"] for f in selected)
    assert ev == before


@pytest.mark.parametrize("failure", ["duration", "no_plan", "missing_file", "duplicate", "time", "hash", "no_camera", "bad_image"])
def test_input_failures_rejected(tmp_path, failure):
    folder, ev = inputs(tmp_path)
    if failure == "duration": ev["meta"]["duration"] = 0
    if failure == "no_plan": ev["keyframes"] = []
    if failure == "missing_file": ev["keyframes"][0]["file_small"] = str(folder / "absent.png")
    if failure == "duplicate": ev["keyframes"][1]["file_small"] = ev["keyframes"][0]["file_small"]
    if failure == "time": ev["keyframes"][0]["t"] = 99
    if failure == "hash": ev["keyframes"][0]["file_small_sha256"] = "0" * 64
    if failure == "no_camera": ev["camera"]["poses"] = []
    if failure == "bad_image": Path(ev["keyframes"][0]["file_small"]).write_bytes(b"not an image")
    with pytest.raises((ValueError, OSError)):
        direct.prepare_direct_inputs(ev, folder)


def test_relocation_requires_opt_in_and_cannot_claim_hash_verification(tmp_path):
    folder, ev = inputs(tmp_path, count=1)
    name = Path(ev["keyframes"][0]["file_small"]).name
    ev["keyframes"][0]["file_small"] = str(tmp_path / "former_location" / name)
    with pytest.raises(ValueError): direct.prepare_direct_inputs(ev, folder)
    _, frames, _ = direct.prepare_direct_inputs(ev, folder, allow_relocated=True)
    assert frames[0]["resolution"] == "relocated_basename_unverified"
    assert not frames[0]["declared_hash_verified"]
    ev["keyframes"][0]["file_small_sha256"] = "0" * 64
    with pytest.raises(ValueError): direct.prepare_direct_inputs(ev, folder, allow_relocated=True)


def test_legacy_evidence_adapter_is_used(tmp_path):
    folder, ev = inputs(tmp_path)
    del ev["schema_version"]
    normalized, selected, _ = direct.prepare_direct_inputs(ev, folder)
    assert normalized["schema_version"] == "2.0" and len(selected) == 12


def test_legacy_frame_registry_is_revalidated_not_trusted_blindly(tmp_path):
    folder, ev = inputs(tmp_path)
    del ev["schema_version"]
    ev["frames"][0]["t"] = 2.0
    with pytest.raises(ValueError, match="Evidence"):
        direct.prepare_direct_inputs(ev, folder)


def test_declared_hash_verification_uses_selected_image_bytes(tmp_path):
    folder, ev = inputs(tmp_path, count=1)
    path = Path(ev["keyframes"][0]["file_small"])
    ev["keyframes"][0]["file_small_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _, frames, _ = direct.prepare_direct_inputs(ev, folder)
    assert frames[0]["declared_hash_verified"]


def test_producer_cwd_relative_image_reference(tmp_path, monkeypatch):
    folder, ev = inputs(tmp_path, count=1)
    monkeypatch.chdir(tmp_path)
    ev["keyframes"][0]["file_small"] = "frames/" + Path(ev["keyframes"][0]["file_small"]).name
    _, selected, _ = direct.prepare_direct_inputs(ev, folder)
    assert Path(selected[0]["file"]).parent == folder
    assert selected[0]["resolution"] == "declared_cwd_path"


def test_cwd_relative_image_cannot_escape_selected_directory(tmp_path, monkeypatch):
    folder, ev = inputs(tmp_path, count=1)
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere.png"
    Image.new("RGB", (4, 4)).save(elsewhere)
    ev["keyframes"][0]["file_small"] = "elsewhere.png"
    with pytest.raises(ValueError):
        direct.prepare_direct_inputs(ev, folder)


def test_ambiguous_relative_image_reference_rejected(tmp_path, monkeypatch):
    folder, ev = inputs(tmp_path, count=1)
    monkeypatch.chdir(tmp_path)
    name = Path(ev["keyframes"][0]["file_small"]).name
    (folder / "frames").mkdir()
    Image.new("RGB", (4, 4)).save(folder / "frames" / name)
    ev["keyframes"][0]["file_small"] = "frames/" + name
    with pytest.raises(ValueError, match="多个合法文件"):
        direct.prepare_direct_inputs(ev, folder)


def test_conflicting_video_source_claims_rejected(tmp_path):
    folder, ev = inputs(tmp_path, count=1)
    ev["meta"].update(source_video_sha256="a" * 64, video_sha256="b" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        direct.prepare_direct_inputs(ev, folder)


def cli_setup(tmp_path, monkeypatch, *, responses=None, packaging_error=False, syntax_error=False):
    import gwm.synthesis.vlm as vlm
    folder, ev = inputs(tmp_path)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(ev), encoding="utf-8")
    captured = {}
    cfg = {"perception": {"n_keyframes": 8}, "frames": {"keyframes": 8},
           "vlm": {"max_tokens": 1234, "seed": 0, "api_key": "synthetic-sensitive-placeholder"}}
    monkeypatch.setattr(direct, "load_config", lambda *args: cfg)
    replies = responses or ['export function describe(THREE) { throw new Error("runtime failure"); }']
    class FakeClient:
        model = "offline-test"
        calls = 0
        def chat(self, *args, **kwargs):
            captured["images"] = kwargs["images"]
            item = replies[self.calls]
            self.calls += 1
            if isinstance(item, Exception): raise item
            return {"text": item}
    client = FakeClient()
    monkeypatch.setattr(vlm, "make_client", lambda config: client)
    monkeypatch.setattr(direct.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 1 if syntax_error else 0, "", "syntax error" if syntax_error else ""))
    def fake_build(code, stub, out, cfg=None):
        captured.update(stub=stub, cfg=cfg)
        if packaging_error: raise RuntimeError("offline packaging failure")
    monkeypatch.setattr(direct, "build_game_dir", fake_build)
    out = tmp_path / "out"
    args = ["--evidence", str(path), "--keyframes", str(folder), "--out", str(out), "--attempts", str(len(replies))]
    return args, out, captured, path


def test_syntax_and_packaging_are_not_runtime_success(tmp_path, monkeypatch):
    args, out, captured, path = cli_setup(tmp_path, monkeypatch)
    assert direct.main(args) == 0  # Artifact workflow completed, not runtime passed.
    report = json.loads((out / "direct_code_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "packaged_unverified" and report["generation_ok"]
    assert report["ok"] is None and report["runtime_ok"] is None and report["runtime_status"] == "not_checked"
    assert report["attempts"][0]["syntax_ok"] and report["attempts"][0]["packaged"]
    assert len(captured["images"]) == 12
    assert captured["stub"]["meta"]["duration"] == 6
    assert captured["cfg"]["vlm"]["max_tokens"] == report["provenance"]["max_tokens"]
    assert report["provenance"]["seed"] == 0
    assert report["provenance"]["evidence_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["provenance"]["video_source_verification"] == "not_checked"
    assert "synthetic-sensitive-placeholder" not in (out / "direct_code_report.json").read_text(encoding="utf-8")


def test_producer_video_hash_is_recorded_as_claim_not_verification(tmp_path, monkeypatch):
    args, out, _, path = cli_setup(tmp_path, monkeypatch)
    ev = json.loads(path.read_text(encoding="utf-8"))
    ev["meta"]["source_video_sha256"] = "a" * 64
    path.write_text(json.dumps(ev), encoding="utf-8")
    assert direct.main(args) == 0
    report = json.loads((out / "direct_code_report.json").read_text(encoding="utf-8"))
    assert report["provenance"]["video_sha256"] == "a" * 64
    assert report["provenance"]["video_source_verification"] == "not_checked"


@pytest.mark.parametrize("failure", ["packaging", "syntax", "call", "response"])
def test_failure_rows_retained_and_reported(tmp_path, monkeypatch, failure):
    args, out, _, _ = cli_setup(tmp_path, monkeypatch,
        packaging_error=failure == "packaging", syntax_error=failure == "syntax",
        responses=[TimeoutError("offline")] if failure == "call" else ([123] if failure == "response" else None))
    assert direct.main(args) == 1
    report = json.loads((out / "direct_code_report.json").read_text(encoding="utf-8"))
    assert not report["generation_ok"] and report["ok"] is None
    expected = {"packaging": "packaging_failed", "syntax": "syntax_failed", "call": "call_failed", "response": "invalid_response"}
    assert report["attempts"][0]["status"] == expected[failure]
    assert report["attempts"][0]["error"]


def test_call_failure_can_retry_without_losing_record(tmp_path, monkeypatch):
    args, out, _, _ = cli_setup(tmp_path, monkeypatch, responses=[TimeoutError("offline"), "export function describe(THREE) { return []; }"])
    assert direct.main(args) == 0
    report = json.loads((out / "direct_code_report.json").read_text(encoding="utf-8"))
    assert report["calls"] == 2
    assert [r["status"] for r in report["attempts"]] == ["call_failed", "packaged_unverified"]


def test_bad_input_stops_before_client_creation(tmp_path, monkeypatch):
    import gwm.synthesis.vlm as vlm
    args, out, _, path = cli_setup(tmp_path, monkeypatch)
    ev = json.loads(path.read_text(encoding="utf-8")); ev["keyframes"] = []
    path.write_text(json.dumps(ev), encoding="utf-8")
    monkeypatch.setattr(vlm, "make_client", lambda config: pytest.fail("must not create client on bad input"))
    assert direct.main(args) == 1
    report = json.loads((out / "direct_code_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "input_failed" and report["calls"] == 0 and report["attempts"] == []


def test_existing_results_not_overwritten(tmp_path, monkeypatch):
    args, out, _, _ = cli_setup(tmp_path, monkeypatch)
    out.mkdir(); saved = out / "direct_code_report.json"; saved.write_text("old record", encoding="utf-8")
    with pytest.raises(SystemExit): direct.main(args)
    assert saved.read_text(encoding="utf-8") == "old record"
