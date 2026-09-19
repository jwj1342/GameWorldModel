"""Frame extraction (ffmpeg), working-frame subset for geometry/segmentation, keyframe selection for the VLM."""
from __future__ import annotations
import json, subprocess, shutil
from pathlib import Path
import numpy as np
from PIL import Image

def probe(video: str | Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,duration", "-show_entries", "format=duration", "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out); s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/"); fps = float(num) / float(den)
    dur = float(s.get("duration") or j["format"]["duration"])
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "duration": dur}

def extract_frames(video: str | Path, out_dir: str | Path, sample_fps: float = 4.0, max_seconds: float = 60.0, max_side: int = 960, start: float = 0.0) -> list[dict]:
    """Extract frames at sample_fps into out_dir/f_%05d.jpg; returns [{index, t, file, width, height}]."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("f_*.jpg"): f.unlink()
    info = probe(video)
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{max_seconds:.3f}", "-vf", f"fps={sample_fps},{scale}", "-q:v", "2", str(out / "f_%05d.jpg")]
    subprocess.run(cmd, check=True)
    files = sorted(out.glob("f_*.jpg"))
    frames = []
    for i, f in enumerate(files):
        with Image.open(f) as im: w, h = im.size
        frames.append({"index": i, "t": round(i / sample_fps, 4), "file": str(f), "width": w, "height": h})
    meta = {"video": str(video), "sample_fps": sample_fps, "start": start, "n_frames": len(frames), "source": info, "frames": frames}
    (out / "frames.json").write_text(json.dumps(meta, indent=1))
    return frames

def motion_profile(frames: list[dict], size: int = 96) -> np.ndarray:
    """Mean absolute grey difference between consecutive frames (downsampled)."""
    prev = None; diffs = [0.0]
    for fr in frames:
        im = np.asarray(Image.open(fr["file"]).convert("L").resize((size, size)), dtype=np.float32) / 255.0
        if prev is not None: diffs.append(float(np.abs(im - prev).mean()))
        prev = im
    return np.asarray(diffs)

def pick_uniform(frames: list[dict], n: int) -> list[dict]:
    if len(frames) <= n: return list(frames)
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))
    return [frames[i] for i in idx]

def select_keyframes(frames: list[dict], k: int, motion: np.ndarray | None = None) -> list[dict]:
    """Pick k frames spread along cumulative motion (more frames where things move), always including first and last."""
    if len(frames) <= k: return [dict(f, reason="all") for f in frames]
    if motion is None: motion = motion_profile(frames)
    w = motion + 0.25 * motion.mean() + 1e-6  # blend with uniform so static parts still get frames
    cum = np.cumsum(w); cum = cum / cum[-1]
    targets = np.linspace(0, 1, k)
    idx = sorted(set(int(np.searchsorted(cum, tg, side="left")) for tg in targets) | {0, len(frames) - 1})
    idx = [min(i, len(frames) - 1) for i in idx][:k]
    return [dict(frames[i], reason="motion_arclength") for i in idx]

def resize_copy(src: str | Path, dst: str | Path, max_side: int) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB"); s = max_side / max(im.size)
        if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        Path(dst).parent.mkdir(parents=True, exist_ok=True); im.save(dst, quality=90)
