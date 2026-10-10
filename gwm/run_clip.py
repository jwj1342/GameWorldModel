"""End-to-end orchestration for one clip: perception -> generation + feedback loop -> binding -> playtest -> report.
Usage: python -m gwm.run_clip --video data/clips/trimmed/x.mp4 --clip x [--phrases "a,b"] [--no-vlm] [--config configs/ablations/no_feedback.yaml] [--resume]"""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, random, re, shutil, subprocess, time, uuid
from pathlib import Path
import numpy as np
from .config import load_config, redact, site_name, REPO
from .errors import ErrorLog
from .perception.run import run_perception, load_masks
from .perception.provenance import file_sha256
from .perception.contract import format_evidence_errors
from .perception.quality import assess_evidence_quality
from .synthesis.direct import evidence_to_program
from .compiler.compile import compile_program
from .compiler.validate import validate, format_errors
from .binding.platformer import bind
from .playtest.autopilot import playtest
from .feedback.render import render
from .feedback.validate_render import validate_render_result

def git_commit() -> str:
    try: return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: return ""

def stage_done(d: Path, marker: str) -> bool: return (d / marker).exists()


def resolve_seed(cfg: dict, override: int | None = None) -> int | None:
    """Resolve CLI precedence without changing RNG state or accepting invalid seeds."""
    seed = override if override is not None else (cfg.get("vlm") or {}).get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32):
        raise ValueError("seed must be null or an integer in [0, 2**32)")
    cfg.setdefault("vlm", {})["seed"] = seed
    return seed


def configuration_identity(cfg: dict) -> str:
    """Fingerprint effective settings, excluding credential fields, not keyframes.

    The old display redactor also masks names containing 'key' (e.g. keyframes),
    so its output alone cannot establish that sampling settings are unchanged.
    """
    credentials = re.compile(r"(?:^|_)(?:key|token|secret|password|authorization|credential)(?:$|_)", re.I)
    def public(value):
        if isinstance(value, dict):
            return {key: "<redacted>" if credentials.search(key) else public(item) for key, item in value.items()}
        if isinstance(value, list):
            return [public(item) for item in value]
        return value
    payload = json.dumps(public(cfg), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_run_manifest(run_dir: Path, args, cfg: dict, seed: int | None) -> dict:
    """Validate before writing: cached Evidence must not acquire a new seed/source.

    Original identity/config/arguments stay intact on resume. A snapshot of the
    previous stage summaries preserves history when stages are rerun.
    """
    video = Path(args.video)
    video_hash = file_sha256(video)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    config = json.loads(json.dumps(redact(cfg), default=str))
    config_id = configuration_identity(cfg)
    if args.resume:
        path = run_dir / "run.json"
        if not path.is_file():
            raise ValueError("resume requires the original run.json; use a new output directory")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        required = ("run_id", "seed", "clip", "video", "config", "args", "started_at", "git_commit", "stages")
        if not isinstance(manifest, dict) or any(key not in manifest for key in required):
            raise ValueError("resume metadata is incomplete; use a new output directory")
        if not isinstance(manifest["args"], dict) or not isinstance(manifest["stages"], dict):
            raise ValueError("resume arguments/stages must be objects")
        recorded_seed = manifest["seed"]
        if recorded_seed is not None and (isinstance(recorded_seed, bool) or not isinstance(recorded_seed, int) or not 0 <= recorded_seed < 2 ** 32):
            raise ValueError("resume metadata contains an invalid seed")
        if recorded_seed != seed or manifest["clip"] != args.clip:
            raise ValueError("resume seed/clip differs from the original run; use a new output directory")
        if not isinstance(manifest["video"], str) or Path(manifest["video"]).resolve() != video.resolve():
            raise ValueError("resume video/config differs from the original run; use a new output directory")
        if manifest.get("config_identity") != config_id:
            raise ValueError("resume effective config identity is missing or mismatched; use a new output directory")
        for key in ("no_vlm", "phrases", "prompt"):
            if key not in manifest["args"] or manifest["args"][key] != getattr(args, key):
                raise ValueError(f"resume {key} differs from the original run; use a new output directory")
        recorded_hash = manifest.get("video_sha256")
        evidence_path = run_dir / "perception" / "evidence.json"
        reused = {}
        previous_stage = manifest["stages"].get("perception", {})
        if not isinstance(previous_stage, dict):
            raise ValueError("resume perception stage metadata must be an object")
        previous_hash = previous_stage.get("evidence_sha256")
        if previous_hash and not evidence_path.is_file():
            raise ValueError("resume recorded Evidence artifact is missing; use a new output directory")
        if evidence_path.is_file():
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(evidence, dict) or not isinstance(evidence.get("meta"), dict):
                raise ValueError("resume cached Evidence metadata must be an object")
            evidence_source = (evidence.get("meta") or {}).get("source_video_sha256")
            if evidence_source != video_hash:
                raise ValueError("resume cached Evidence source hash is missing or mismatched")
            evidence_hash = file_sha256(evidence_path)
            if not isinstance(previous_hash, str) or not previous_hash:
                raise ValueError("resume cached Evidence has no recorded content hash; use a new output directory")
            if previous_hash != evidence_hash:
                raise ValueError("resume cached Evidence content differs from the recorded stage")
            recorded_hash = recorded_hash or evidence_source
            reused["perception/evidence.json"] = evidence_hash
        if recorded_hash != video_hash:
            raise ValueError("resume video hash is missing or mismatched; source cannot be verified")
        attempts = manifest.setdefault("resume_attempts", [])
        if not isinstance(attempts, list):
            raise ValueError("resume_attempts must be an array")
        attempts.append({"started_at": now, "git_commit": git_commit(), "node": platform.node(),
                         "args": copy.deepcopy(vars(args)), "previous_stages": copy.deepcopy(manifest["stages"]),
                         "previous_result": {key: copy.deepcopy(manifest[key]) for key in
                                             ("status", "error_type", "exit_code", "updated_at") if key in manifest},
                         "reused_artifacts": reused})
        manifest["stages"] = {"perception": {**previous_stage, "status": "reused"}} if reused else {}
        return manifest
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("output contains an existing run; use --resume or a new output directory")
    return {"run_id": run_dir.name, "clip": args.clip, "video": str(video.resolve()), "video_sha256": video_hash,
            "started_at": now, "git_commit": git_commit(), "node": platform.node(), "seed": seed,
            "config": config, "config_identity": config_id, "args": copy.deepcopy(vars(args)), "stages": {}}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--out", default=None, help="run dir (default out/<clip>/<run_id>)")
    ap.add_argument("--phrases", default=None); ap.add_argument("--config", action="append", default=[]); ap.add_argument("--no-vlm", action="store_true"); ap.add_argument("--resume", default=None, help="existing run dir to resume")
    ap.add_argument("--prompt", default=None, help="gameplay prompt (MVP: platformer only)")
    ap.add_argument("--seed", type=int, default=None, help="随机种子；覆盖 vlm.seed，并固定 random / numpy。同一配置换 seed 重跑即可得到多次独立样本")
    a = ap.parse_args(argv)
    cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *[REPO / c for c in a.config]])
    try:
        seed = resolve_seed(cfg, a.seed)
    except ValueError as exc:
        ap.error(str(exc))
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    run_dir = Path(a.resume) if a.resume else Path(a.out or (Path(cfg["paths"]["out"]) / a.clip / run_id))
    try:
        manifest = prepare_run_manifest(run_dir, a, cfg, seed)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    def outcome(status, error=None):
        # Error fields describe this attempt only; prior results live in history.
        for key in ("error_type", "exit_code"):
            manifest.pop(key, None)
            if a.resume:
                manifest["resume_attempts"][-1].pop(key, None)
        record = {"status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if error is not None:
            record["error_type"] = type(error).__name__
            if isinstance(error, SystemExit):
                record["exit_code"] = error.code if isinstance(error.code, int) else 1
        manifest.update(record)
        if a.resume:
            manifest["resume_attempts"][-1].update(record)
            manifest["resume_attempts"][-1]["stages"] = copy.deepcopy(manifest["stages"])
    outcome("running")
    try:
        _execute_run(a, cfg, run_dir, manifest)
    except BaseException as exc:
        outcome("failed", exc)
        if (run_dir / "run.json").is_file():
            (run_dir / "run.json").write_text(json.dumps(manifest, indent=1, default=str))
        raise
    else:
        outcome("completed")
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=1, default=str))


def _execute_run(a, cfg: dict, run_dir: Path, manifest: dict):
    """Execute only after resume preflight; main records the attempt outcome."""
    seed = manifest["seed"]
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = ErrorLog(run_dir / "errors.jsonl")
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=1, default=str))
    def save(): manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S"); (run_dir / "run.json").write_text(json.dumps(manifest, indent=1, default=str))

    client = None
    if not a.no_vlm:
        try:
            from .synthesis.vlm import make_client
            client = make_client(cfg, run_dir / "vlm_calls.jsonl"); manifest["models"] = {"vlm": client.model}
        except Exception as e:
            log.record("setup", "vlm_unavailable", repr(e), action_taken="direct translation only (no VLM)"); client = None

    # ---- perception ----
    pdir = run_dir / "perception"; t0 = time.time()
    if stage_done(pdir, "evidence.json"):
        ev = json.loads((pdir / "evidence.json").read_text())
    else:
        ev = run_perception(a.video, a.clip, pdir, cfg, phrases=[p.strip() for p in a.phrases.split(",")] if a.phrases else None, client=client, log=log)
    # Record generated content before a quality block or exception can interrupt us.
    if (pdir / "evidence.json").is_file():
        manifest["stages"].setdefault("perception", {})["evidence_sha256"] = file_sha256(pdir / "evidence.json")
        save()
    quality_report = assess_evidence_quality(ev, cfg)
    evidence_report = quality_report["validation"]
    manifest["stages"]["evidence_validation"] = {
        "ok": evidence_report["ok"], "adapted": evidence_report["adapted"],
        "errors": len(evidence_report["errors"]), "warnings": len(evidence_report["warnings"]),
    }
    quality_artifact = {key: value for key, value in quality_report.items() if key not in ("evidence", "validation")}
    (pdir / "evidence_quality.json").write_text(json.dumps(quality_artifact, indent=1, ensure_ascii=False))
    manifest["stages"]["evidence_quality"] = {
        "decision": quality_report["decision"], "metrics": quality_report["metrics"],
        "diagnostic_codes": sorted({item["code"] for item in quality_report["diagnostics"]}),
    }
    if quality_report["decision"] == "block":
        log.record("evidence", "validation_failed", format_evidence_errors(evidence_report), recoverable=False,
                   action_taken="stop before Program generation")
        save(); raise SystemExit(2)
    if quality_report["decision"] == "warn":
        codes = sorted({item["code"] for item in quality_report["diagnostics"] if item["severity"] == "warning"})
        log.record("evidence", "quality_warning", ", ".join(codes), action_taken="proceed with explicit quality diagnostics")
    ev = quality_report["evidence"]
    frames = json.loads((pdir / "frames" / "frames.json").read_text())["frames"]
    sam_masks = load_masks(pdir)
    perception_status = manifest["stages"].get("perception", {}).get("status", "completed")
    manifest["stages"]["perception"] = {"status": perception_status, "seconds": round(time.time() - t0, 1), "objects": len(ev["objects"]), "geometry_backend": ev["meta"]["geometry_backend"], "fallbacks": ev["meta"]["fallbacks"], "evidence_sha256": file_sha256(pdir / "evidence.json")}; save()
    key_times = [k["t"] for k in ev["keyframes"]]
    kf_files = [k.get("file_small") or k["file"] for k in ev["keyframes"]]

    # ---- generation ----
    gdir = run_dir / "generation"; gdir.mkdir(exist_ok=True); t0 = time.time()
    if client is not None:
        from .synthesis.writer import Writer
        writer = Writer(client, cfg, gdir / "writer_log")
        program, winfo = writer.generate(ev, kf_files, a.clip)
    else:
        program, winfo = evidence_to_program(ev, a.clip, cfg), {"mode": "direct"}
    (gdir / "program_initial.json").write_text(json.dumps(program, indent=1, ensure_ascii=False))
    manifest["stages"]["writer"] = {"seconds": round(time.time() - t0, 1), **winfo}; save()

    # ---- feedback loop ----
    t0 = time.time()
    from .synthesis.loop import run_loop
    dino_dir = str(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])
    loop_res = run_loop(program, ev, frames, sam_masks, key_times, cfg, gdir / "rounds", client, dino_dir, log)
    program = loop_res["program"]
    manifest["stages"]["feedback"] = {"seconds": round(time.time() - t0, 1), "rounds_run": loop_res["rounds_run"], "history": loop_res["history"], "final_score": loop_res["best"].get("score")}; save()

    # ---- binding + final compile ----
    program = bind(program, ev, cfg, a.prompt)
    rep = validate(program)
    if not rep["ok"]:
        log.record("binding", "invalid_after_binding", format_errors(rep), action_taken="drop binding slots"); program["binding"] = {"template": "platformer_3p", "slots": {}}
    (run_dir / "program.json").write_text(json.dumps(program, indent=1, ensure_ascii=False))
    comp = compile_program(program, run_dir / "game", cfg)
    if not comp["ok"]:
        log.record("compile", "final_compile_failed", format_errors(comp["validation"]), recoverable=False); save(); raise SystemExit(2)
    # copy final render for the report
    if loop_res["best"].get("ok"):
        src = Path(loop_res["best"]["game_dir"]).parent / "render"
        if src.exists(): shutil.copytree(src, run_dir / "feedback", dirs_exist_ok=True)

    # Validate the final bound Program's own render, not an earlier feedback candidate.
    rv_dir = run_dir / "render_validation_frames"
    duration = float(program.get("meta", {}).get("duration") or 0.0)
    n_validation_frames = max(2, int(cfg.get("render_validation", {}).get("sample_frames", 5)))
    validation_times = [round(float(value), 3) for value in np.linspace(0.0, duration, n_validation_frames)]
    render_failure = None
    try:
        rv_index = render(run_dir / "game", validation_times, rv_dir,
                          cfg["feedback"]["render_width"], cfg["feedback"]["render_height"], ("rgb", "depth", "id"))
    except Exception as exc:
        rv_index = None
        render_failure = repr(exc)[:1000]
    rv_report = validate_render_result(program, rv_index, rv_dir, cfg, expected_times=validation_times)
    if render_failure:
        rv_report["render_failure"] = render_failure
    (run_dir / "render_validation.json").write_text(json.dumps(rv_report, indent=1, ensure_ascii=False))
    manifest["stages"]["render_validation"] = {
        "decision": rv_report["decision"], "errors": len(rv_report["errors"]), "warnings": len(rv_report["warnings"]),
        "diagnostic_codes": sorted({item["code"] for item in rv_report["diagnostics"]}),
        "artifact": "render_validation.json", "mode": rv_report["mode"],
    }
    save()
    if rv_report["decision"] == "block":
        log.record("render_validation", "render_invalid", ", ".join(manifest["stages"]["render_validation"]["diagnostic_codes"]),
                   recoverable=rv_report["mode"] != "enforce", action_taken="report only" if rv_report["mode"] == "report" else "stop before playtest")
        if rv_report["mode"] == "enforce":
            raise SystemExit(2)

    # ---- playtest ----
    t0 = time.time()
    verdict = playtest(run_dir / "game", run_dir / "playtest", cfg, client, video_frame=kf_files[0] if kf_files else None)
    manifest["stages"]["playtest"] = {"seconds": round(time.time() - t0, 1), **{k: v for k, v in verdict.items() if k != "review"}, "review": verdict.get("review")}; save()

    # ---- report ----
    write_report(run_dir, manifest, ev, program, loop_res, verdict, log)
    print(json.dumps({"run_dir": str(run_dir), "objects": len(program["objects"]), "score": loop_res["best"].get("score"), "reached_goal": verdict.get("reached_goal"), "errors": len(log.entries())}, indent=1))

def write_report(run_dir: Path, m: dict, ev: dict, program: dict, loop_res: dict, verdict: dict, log: ErrorLog):
    L = [f"# Run {m['run_id']} — clip `{m['clip']}`", "", f"video: `{m['video']}`  commit: `{m['git_commit']}`  node: {m['node']}  started: {m['started_at']}", "", "## Perception",
         f"- geometry backend: {ev['meta']['geometry_backend']} (scale {ev['meta']['scale']}); segmentation: {ev['meta']['segmentation_backend']}; fallbacks: {ev['meta']['fallbacks'] or 'none'}",
         f"- frames: {ev['meta']['n_frames']} @ {ev['meta']['fps_sampled']} fps, duration {ev['meta']['duration']:.1f} s; keyframes: {len(ev['keyframes'])}; timing: {ev['meta'].get('timing_s')}",
         f"- alignment: {ev['meta'].get('alignment')}", "", "| evidence object | class guess | dynamic | motion guess | conf | frames |", "|---|---|---|---|---|---|"]
    for o in ev["objects"]:
        mg = o["motion_guess"]; conf = mg.get("conf"); conf_text = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "unknown"; L.append(f"| {o['id']} | {o['class_guess']} | {o['is_dynamic']} | {mg.get('type','unknown')} ({mg.get('notes','')[:60]}) | {conf_text} | {len(o['obb'])} |")
    L += ["", "## Program", f"- writer: {json.dumps(m['stages'].get('writer'), default=str)[:600]}", f"- static: {len(program['static'])}, objects: {len(program['objects'])}", "", "| id | class | geom | motion |", "|---|---|---|---|"]
    for o in program["objects"]: L.append(f"| {o['id']} | {o.get('class')} | {o['geom'].get('kind')}:{o['geom'].get('shape') or o['geom'].get('query')} {o['geom'].get('extent')} | {(o.get('motion') or {}).get('type','static')} |")
    L += ["", "## Feedback loop", "| round | score | note |", "|---|---|---|"]
    for h in loop_res["history"]: L.append(f"| {h.get('round')} | {h.get('best_score', h.get('score'))} | {h.get('note') or h.get('critic_summary') or h.get('error') or ''} |")
    L += ["", f"final score: {loop_res['best'].get('score')}; binding slots: `{json.dumps(program.get('binding', {}).get('slots'))[:300]}`", "", "## Playtest", f"```json\n{json.dumps(verdict, indent=1, ensure_ascii=False)}\n```", "", "## Errors / fallbacks", ""]
    for e in log.entries(): L.append(f"- [{e['stage']}/{e['code']}] {e['message'][:160]} → {e['action_taken']}")
    L += ["", "## Files", "- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`"]
    (run_dir / "report.md").write_text("\n".join(L))

if __name__ == "__main__":
    main()
