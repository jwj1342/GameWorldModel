"""Call the Node harness to render rgb/depth/id passes at given times. Returns the RenderSet index dict."""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def run_harness(script: str, args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    cmd = [str(REPO / "scripts" / "node_harness.sh"), script, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def render(game_dir: str | Path, times: list[float], out_dir: str | Path, width: int, height: int, passes=("rgb", "depth", "id")) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cp = run_harness("render.mjs", ["--game", str(game_dir), "--times", ",".join(f"{t:.3f}" for t in times), "--out", str(out), "--width", str(width), "--height", str(height), "--passes", ",".join(passes)])
    (out / "harness_stdout.txt").write_text(cp.stdout + "\n--- stderr ---\n" + cp.stderr)
    if cp.returncode != 0:
        raise RuntimeError(f"render harness failed (code {cp.returncode}): {cp.stderr[-1500:]}")
    return json.loads((out / "index.json").read_text())
