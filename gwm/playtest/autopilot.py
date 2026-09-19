"""Automated playtest via the Node harness + rule verdict (+ optional one-line VLM review)."""
from __future__ import annotations
import json
from pathlib import Path
from ..feedback.render import run_harness

REPO = Path(__file__).resolve().parents[2]

def playtest(game_dir: Path, out_dir: Path, cfg: dict, client=None, video_frame: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cp = run_harness("playtest.mjs", ["--game", str(game_dir), "--out", str(out_dir), "--seconds", str(cfg["playtest"]["seconds"]), "--shot-every", str(cfg["playtest"]["shot_every_s"])], timeout=3600)
    (out_dir / "harness_stdout.txt").write_text(cp.stdout + "\n--- stderr ---\n" + cp.stderr)
    vpath = out_dir / "verdict.json"
    verdict = json.loads(vpath.read_text()) if vpath.exists() else {"reached_goal": False, "stuck": False, "playable": False, "error": cp.stderr[-800:]}
    if client is not None and cfg["playtest"].get("vlm_review", True):
        frames = sorted((out_dir / "frames").glob("rgb_*.png"))
        pick = frames[:: max(1, len(frames) // 6)][:6]
        imgs = ([video_frame] if video_frame else []) + [str(p) for p in pick]
        try:
            r = client.chat((REPO / "prompts" / "playtest_review.md").read_text(), "First image (if present) is a frame of the source video; the rest are playthrough frames in time order.", images=imgs,
                            json_schema={"type": "object", "required": ["related_to_video", "playable", "defect"], "properties": {"related_to_video": {"type": "boolean"}, "playable": {"type": "boolean"}, "defect": {"type": "string"}}}, temperature=0.2)
            verdict["review"] = r["json"]
        except Exception as e:
            verdict["review_error"] = repr(e)[:300]
    vpath.write_text(json.dumps(verdict, indent=1, ensure_ascii=False))
    return verdict
