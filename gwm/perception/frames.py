"""Frame extraction (ffmpeg), working-frame subset for geometry/segmentation, keyframe selection for the VLM."""
from __future__ import annotations
import json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

from .provenance import file_sha256

def probe(video: str | Path, ffprobe_bin: str = "ffprobe") -> dict:
    out = subprocess.run([ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,duration", "-show_entries", "format=duration", "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out); s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/"); fps = float(num) / float(den)
    dur = float(s.get("duration") or j["format"]["duration"])
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "duration": dur}

def extract_frames(video: str | Path, out_dir: str | Path, sample_fps: float = 4.0, max_seconds: float = 60.0, max_side: int = 960, start: float = 0.0,
                   ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> list[dict]:
    """Extract sampled decoded frames with source PTS, decoded index, and video hash.

    The final decoded frame before each sampling bucket's half-tick boundary
    reproduces FFmpeg fps-filter downsampling (e.g. 30 fps to 1 fps starts at
    source frame 14 / PTS 0.467, not source frame 0 / PTS 0).
    """
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
           for value in (sample_fps, max_seconds, start)) or sample_fps <= 0 or max_seconds <= 0 or start < 0:
        raise ValueError("sampling FPS, duration, and start must be finite and valid")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("f_*.jpg")):
        raise FileExistsError("frame output directory already contains sampled images")
    info = probe(video, ffprobe_bin)
    video_hash = file_sha256(video)
    effective_seconds = min(max_seconds, max(0.0, info["duration"] - start))
    frame_json = subprocess.run(
        [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    decoded = json.loads(frame_json).get("frames", [])
    bucket_count = math.ceil(effective_seconds * sample_fps)
    chosen: dict[int, tuple[int, float]] = {}
    for source_index, item in enumerate(decoded):
        raw_pts = item.get("best_effort_timestamp_time")
        if raw_pts is None:
            raise ValueError(f"decoded source frame {source_index} has no PTS")
        pts = float(raw_pts)
        if not math.isfinite(pts):
            raise ValueError(f"decoded source frame {source_index} has invalid PTS")
        if not start <= pts < start + effective_seconds:
            continue
        bucket = math.floor((pts - start) * sample_fps + 0.5)
        if 0 <= bucket < bucket_count:
            chosen[bucket] = (source_index, pts)
    selected = sorted(chosen.values(), key=lambda item: item[0])
    if not selected:
        raise ValueError("no decoded source frames in requested sampling range")
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    select = "+".join(f"eq(n\\,{index})" for index, _ in selected)
    cmd = [ffmpeg_bin, "-loglevel", "error", "-y", "-i", str(video), "-vf", f"select='{select}',{scale}",
           "-vsync", "0", "-q:v", "2", str(out / "f_%05d.jpg")]
    subprocess.run(cmd, check=True)
    if file_sha256(video) != video_hash:
        raise RuntimeError("source video changed during frame extraction")
    files = sorted(out.glob("f_*.jpg"))
    if len(files) != len(selected):
        raise RuntimeError("extracted image count differs from decoded PTS selection")
    frames = []
    for i, (f, (source_index, pts)) in enumerate(zip(files, selected)):
        with Image.open(f) as im: w, h = im.size
        frames.append({"index": i, "t": round(pts - start, 6), "source_pts_s": pts,
                       "source_frame_index": source_index, "video_sha256": video_hash,
                       "file": str(f), "image_sha256": file_sha256(f), "width": w, "height": h})
    meta = {"video": str(video), "video_sha256": video_hash, "sample_fps": sample_fps, "start": start,
            "n_frames": len(frames), "source": info, "frames": frames}
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
