"""重建出来的场景，和原视频像不像。

这是整条链路最该有、却一直没有的那个数。现有的指标各自只看一个侧面：
掩码覆盖率测的是「有多少像素被检测声称占有」（大的虚检反而让它变好看），
物体召回测的是「检出来的有没有重建」，轨迹误差测的是运动。没有一个在回答
「把它渲染出来，和原视频像不像」——而那恰恰是看一眼就能判断、也最该被优化的东西。

做法直接：按**输入视频的相机轨迹**把重建的场景渲染出来，和对应时刻的原帧比。
比较用 DINOv2 的图像嵌入做余弦相似度，和渲染反馈回路里那一项同一个口径
（`gwm/feedback/metrics.py` 的 `Dino`），所以两边的数可以互相印证。

刻意不做的事：不做像素级的 L2 或 PSNR。重建出来的是低多边形的程序化场景，
配的是库里的资产，和真实视频在像素上本来就不可能接近；要看的是「结构和布局像不像」，
那正是自监督视觉特征擅长的。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def pick_times(frames: list[dict], count: int = 8) -> list[float]:
    """在片段上均匀取若干时刻。帧数不够就全要。"""
    times = sorted({round(float(f["t"]), 3) for f in frames if "t" in f})
    if not times:
        return []
    if len(times) <= count:
        return times
    step = (len(times) - 1) / (count - 1)
    return [times[round(i * step)] for i in range(count)]


def render_at(game_dir: Path, times: list[float], out_dir: Path, harness: str,
              width: int = 640, height: int = 360) -> dict:
    """按给定时刻渲染场景。渲染失败不抛异常，返回里带 error，让调用方决定要不要继续。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [harness, "render.mjs", "--game", str(game_dir), "--out", str(out_dir),
           "--times", ",".join(f"{t:.3f}" for t in times),
           "--width", str(width), "--height", str(height), "--passes", "rgb"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    index = out_dir / "index.json"
    if done.returncode != 0 or not index.is_file():
        return {"ok": False, "error": (done.stderr or done.stdout or "")[-300:]}
    return {"ok": True, "index": json.loads(index.read_text(encoding="utf-8"))}


def likeness(game_dir: Path, frames: list[dict], out_dir: Path, harness: str,
             weights_dir: str | Path, count: int = 8,
             width: int = 640, height: int = 360) -> dict:
    """返回重建场景和原视频的相似度。

    `frames` 是感知阶段抽出来的帧（带 `t` 和 `file`）。每个采样时刻各取一对：
    渲染图和最接近那个时刻的原帧。
    """
    import numpy as np
    from .metrics import Dino

    times = pick_times(frames, count)
    if not times:
        return {"ok": False, "reason": "no frames"}
    rendered = render_at(game_dir, times, out_dir, harness, width, height)
    if not rendered["ok"]:
        return {"ok": False, "reason": "render failed", "error": rendered.get("error")}

    by_time = {round(float(f["t"]), 3): f["file"] for f in frames if f.get("file")}
    dino = Dino(weights_dir)
    pairs, scores = [], []
    for entry in rendered["index"].get("frames", []):
        t = round(float(entry.get("t", entry.get("requested_t", 0.0))), 3)
        rgb = entry.get("files", {}).get("rgb")
        if not rgb:
            continue
        nearest = min(by_time, key=lambda k: abs(k - t)) if by_time else None
        if nearest is None or abs(nearest - t) > 0.5:
            continue
        score = float(dino.embed(out_dir / rgb) @ dino.embed(by_time[nearest]))
        scores.append(score)
        pairs.append({"t": t, "source_t": nearest, "similarity": round(score, 4)})
    if not scores:
        return {"ok": False, "reason": "no comparable frames"}
    ordered = sorted(scores)
    return {"ok": True, "n": len(scores),
            "median": round(float(np.median(ordered)), 4),
            "mean": round(float(np.mean(ordered)), 4),
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4),
            "per_frame": pairs}
