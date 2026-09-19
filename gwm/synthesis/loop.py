"""Evaluator-optimizer loop: render -> metrics -> clauses -> critic -> JSON Patch candidates -> compile/render/score -> keep best or roll back."""
from __future__ import annotations
import copy, json, time
from pathlib import Path
import jsonpatch
from ..compiler.compile import compile_program
from ..compiler.validate import validate, format_errors
from ..feedback.render import render
from ..feedback.metrics import compute_metrics, Dino
from .critic import make_crops, critique

def evaluate(program: dict, round_dir: Path, evidence: dict, frames: list[dict], sam_masks: dict, cfg: dict, dino, key_times: list[float]) -> dict:
    """Compile + render + metrics for one program. Returns dict with game_dir, render_index, metrics, score (or error)."""
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "program.json").write_text(json.dumps(program, indent=1, ensure_ascii=False))
    comp = compile_program(program, round_dir / "game")
    if not comp["ok"]:
        (round_dir / "compile_errors.txt").write_text(format_errors(comp["validation"])); return {"ok": False, "error": "compile", "details": comp["validation"]}
    try:
        idx = render(round_dir / "game", key_times, round_dir / "render", cfg["feedback"]["render_width"], cfg["feedback"]["render_height"], tuple(cfg["feedback"]["passes"]))
    except Exception as e:
        (round_dir / "render_error.txt").write_text(repr(e)); return {"ok": False, "error": "render", "details": repr(e)[:1000]}
    m = compute_metrics(program, idx, round_dir / "render", evidence, frames, sam_masks, cfg, dino)
    (round_dir / "clauses.json").write_text(json.dumps(m, indent=1))
    return {"ok": True, "game_dir": str(round_dir / "game"), "render_index": idx, "metrics": m, "score": m["summary"]["score"]}

def apply_patches(program: dict, items: list[dict]) -> tuple[dict, list[str]]:
    p = copy.deepcopy(program); applied = []
    for it in items:
        ops = [op for op in (it.get("suggested_edit") or []) if not (isinstance(op.get("value"), list) and all(isinstance(v, (int, float)) and abs(v) < 1e-9 for v in op["value"]))]
        if not ops: continue
        try:
            cand = jsonpatch.JsonPatch(ops).apply(p)
            if validate(cand)["ok"]: p = cand; applied.append(it["object_id"])
        except Exception:
            continue
    return p, applied

def run_loop(program: dict, evidence: dict, frames: list[dict], sam_masks: dict, key_times: list[float], cfg: dict, out_dir: Path, client, dino_dir: str | None, log) -> dict:
    fb = cfg["feedback"]; out_dir.mkdir(parents=True, exist_ok=True)
    dino = None
    if dino_dir:
        try: dino = Dino(dino_dir)
        except Exception as e: log.record("feedback", "dino_unavailable", repr(e), action_taken="scoring without DINO")
    history = []
    best = evaluate(program, out_dir / "r0", evidence, frames, sam_masks, cfg, dino, key_times)
    best_prog = program; history.append({"round": 0, "ok": best["ok"], "score": best.get("score"), "error": best.get("error")})
    if not best["ok"]:
        return {"program": program, "best": best, "history": history, "rounds_run": 0}
    rounds = 0 if (cfg["ablations"].get("no_feedback") or client is None) else fb["rounds"]
    if client is None: history.append({"round": 0, "note": "no VLM client: feedback loop skipped"})
    for r in range(1, rounds + 1):
        clauses = best["metrics"]["clauses"]
        fail_ids = {c["object_id"] for c in clauses if c["status"] == "FAIL" and c["check"] != "presence"}
        if not fail_ids: history.append({"round": r, "note": "all clauses pass"}); break
        rdir = out_dir / f"r{r}"; rdir.mkdir(exist_ok=True)
        try:
            crops = make_crops(best_prog, best["render_index"], Path(best["game_dir"]).parent / "render", frames, sam_masks, fail_ids, rdir / "crops")
            crit = critique(client, best_prog, clauses, crops, rdir / "critic.json", evidence)
        except Exception as e:
            log.record("feedback", "critic_failed", repr(e), action_taken="stop loop"); history.append({"round": r, "error": "critic", "details": repr(e)[:300]}); break
        cand_results = []
        items = crit.get("items", [])
        if not items: history.append({"round": r, "note": "critic proposed nothing"}); break
        # candidate A: all patches; candidate B: only the worst object's patches
        iou_by_pid = {v.get("program_id"): v.get("mean_iou", 0) or 0 for v in best["metrics"]["summary"]["per_object"].values()}
        worst = min(fail_ids, key=lambda pid: iou_by_pid.get(pid, 0))
        subsets = [items] + ([[it for it in items if it["object_id"] == worst]] if len(items) > 1 else [])
        for k, sub in enumerate(subsets[: fb.get("candidates", 2)]):
            cand, applied = apply_patches(best_prog, sub)
            if not applied: continue
            res = evaluate(cand, rdir / f"cand{k}", evidence, frames, sam_masks, cfg, dino, key_times)
            cand_results.append({"k": k, "applied": applied, "ok": res["ok"], "score": res.get("score"), "res": res, "program": cand})
        ok_c = [c for c in cand_results if c["ok"]]
        if not ok_c: history.append({"round": r, "note": "no valid candidate", "candidates": [{k: v for k, v in c.items() if k not in ("res", "program")} for c in cand_results]}); break
        top = max(ok_c, key=lambda c: c["score"])
        improved = top["score"] > best["score"] + fb["min_improvement"]
        history.append({"round": r, "candidates": [{k: v for k, v in c.items() if k not in ("res", "program")} for c in cand_results], "chosen": top["k"] if improved else None, "best_score": max(top["score"], best["score"]), "improved": improved, "critic_summary": crit.get("summary", "")})
        if improved: best, best_prog = top["res"], top["program"]
        else: log.record("feedback", "no_improvement", f"round {r}: best candidate {top['score']:.3f} <= {best['score']:.3f}", action_taken="keep previous program")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "best_program.json").write_text(json.dumps(best_prog, indent=1, ensure_ascii=False))
    return {"program": best_prog, "best": best, "history": history, "rounds_run": len([h for h in history if "candidates" in h])}
