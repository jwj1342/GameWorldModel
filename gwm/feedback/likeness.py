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
             width: int = 640, height: int = 360,
             unrelated_frames: list[str] | None = None) -> dict:
    """返回重建场景和原视频的相似度，**连同两条标尺**。

    光报一个 0.19 没有意义：DINOv2 在两张毫不相干的图之间也有 0.1 上下的相似度。
    所以一起报两条参照：

    * `reference_self`：原视频不同时刻的两帧互相比。这是「同一个场景、不同视角」的
      相似度，也就是我们能指望的上限——重建得再好，视角和光照的差异也还在。
    * `reference_unrelated`：原帧和**另一段视频**的帧比。这是下限，相当于「完全没
      重建对」时这个数会是多少。

    重建的分落在这两条之间的什么位置，才是真正能读的东西。

    `frames` 是感知阶段抽出来的帧（带 `t` 和 `file`）。
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
    out = {"ok": True, "n": len(scores),
           "median": round(float(np.median(ordered)), 4),
           "mean": round(float(np.mean(ordered)), 4),
           "min": round(ordered[0], 4), "max": round(ordered[-1], 4),
           "per_frame": pairs}

    # 上限：同一段视频不同时刻的两帧。重建得再好也追不上这个数，因为视角差异还在。
    sampled = [by_time[t] for t in times if t in by_time]
    if len(sampled) >= 2:
        embeddings = [dino.embed(f) for f in sampled]
        self_scores = [float(embeddings[i] @ embeddings[i + 1]) for i in range(len(embeddings) - 1)]
        out["reference_self"] = round(float(np.median(self_scores)), 4)
        # 下限：和另一段视频的帧比，相当于完全没重建对时这个数是多少
        if unrelated_frames:
            others = [dino.embed(f) for f in unrelated_frames[:len(embeddings)]]
            cross = [float(a @ b) for a in embeddings for b in others]
            out["reference_unrelated"] = round(float(np.median(cross)), 4)
            span = out["reference_self"] - out["reference_unrelated"]
            if span > 1e-6:
                # 重建的分落在下限和上限之间的什么位置，0 是完全没重建对，1 是追平了
                out["normalised"] = round((out["median"] - out["reference_unrelated"]) / span, 3)
    return out
