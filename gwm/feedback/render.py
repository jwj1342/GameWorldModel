"""Call the Node harness to render rgb/depth/id passes at given times. Returns the RenderSet index dict."""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def run_harness(script: str, args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    cmd = [str(REPO / "scripts" / "node_harness.sh"), script, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def render(game_dir: str | Path, times: list[float], out_dir: str | Path, width: int, height: int, passes=("rgb", "depth", "id"), *, unoccluded_id: bool = False, scene_only: bool = False) -> dict:
    if not isinstance(scene_only, bool):
        raise TypeError("scene_only must be a bool")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # 时间用 .17g 不用 .3f：#87 修的就是这个——0.366667 被截成 0.367 之后，
    # 渲染出来的时刻和请求的时刻对不上，时间校验会判成换了一帧。
    args = ["--game", str(game_dir), "--times", ",".join(format(float(t), ".17g") for t in times),
            "--out", str(out), "--width", str(width), "--height", str(height), "--passes", ",".join(passes)]
    if unoccluded_id:
        args += ["--unoccluded-id", "true"]
    if scene_only:
        args += ["--scene-only"]
    cp = run_harness("render.mjs", args)
    (out / "harness_stdout.txt").write_text(cp.stdout + "\n--- stderr ---\n" + cp.stderr)
    if cp.returncode != 0:
        raise RuntimeError(f"render harness failed (code {cp.returncode}): {cp.stderr[-1500:]}")
    return json.loads((out / "index.json").read_text())
