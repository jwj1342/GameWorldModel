"""End-to-end orchestration for one clip: perception -> generation + feedback loop -> binding -> playtest -> report.
Usage: python -m gwm.run_clip --video data/clips/trimmed/x.mp4 --clip x [--phrases "a,b"] [--no-vlm] [--config configs/ablations/no_feedback.yaml] [--resume]"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, time, uuid
from pathlib import Path
import numpy as np
from .config import load_config, REPO
from .errors import ErrorLog
from .perception.run import run_perception, load_masks
from .perception.contract import format_evidence_errors
from .perception.quality import assess_evidence_quality
from .synthesis.direct import evidence_to_program
from .compiler.compile import compile_program
from .compiler.validate import validate, format_errors
from .binding.platformer import bind
from .playtest.autopilot import playtest

def git_commit() -> str:
    try: return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: return ""

def stage_done(d: Path, marker: str) -> bool: return (d / marker).exists()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--out", default=None, help="run dir (default out/<clip>/<run_id>)")
    ap.add_argument("--phrases", default=None); ap.add_argument("--config", action="append", default=[]); ap.add_argument("--no-vlm", action="store_true"); ap.add_argument("--resume", default=None, help="existing run dir to resume")
    ap.add_argument("--prompt", default=None, help="gameplay prompt (MVP: platformer only)")
    a = ap.parse_args(argv)
    cfg = load_config([REPO / "configs/default.yaml", REPO / "configs/vulcan.yaml", *[REPO / c for c in a.config]])
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    run_dir = Path(a.resume) if a.resume else Path(a.out or (Path(cfg["paths"]["out"]) / a.clip / run_id)); run_dir.mkdir(parents=True, exist_ok=True)
    log = ErrorLog(run_dir / "errors.jsonl")
    manifest = {"run_id": run_dir.name, "clip": a.clip, "video": str(a.video), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(), "node": os.uname().nodename, "config": cfg, "args": vars(a), "stages": {}}
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
    manifest["stages"]["perception"] = {"seconds": round(time.time() - t0, 1), "objects": len(ev["objects"]), "geometry_backend": ev["meta"]["geometry_backend"], "fallbacks": ev["meta"]["fallbacks"]}; save()
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
    comp = compile_program(program, run_dir / "game")
    if not comp["ok"]:
        log.record("compile", "final_compile_failed", format_errors(comp["validation"]), recoverable=False); save(); raise SystemExit(2)
    # copy final render for the report
    if loop_res["best"].get("ok"):
        src = Path(loop_res["best"]["game_dir"]).parent / "render"
        if src.exists(): shutil.copytree(src, run_dir / "feedback", dirs_exist_ok=True)

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
