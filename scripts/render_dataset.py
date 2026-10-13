#!/usr/bin/env python
"""把生成的程序渲染成视频，并逐条记下耗时。

帧序列只落在节点本地盘（$SLURM_TMPDIR），只有 clip.mp4、gt_states.jsonl 和 record.json
会拷回共享盘。一条 8 秒 30fps 的片段是 240 张 PNG，2000 条就是 48 万个小文件，
直接写 Lustre 会把元数据服务器拖垮。

用法:
  python -m scripts.render_dataset --root $GWM_SCRATCH/dataset/probe50 --out $GWM_SCRATCH/dataset/probe50_rendered
  # 作业数组切片：--shard 3 --shards 8 只做第 3 份
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HARNESS = REPO / "scripts" / "node_harness.sh"


def _work_root() -> Path:
    base = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    root = Path(base) / "gwm_render"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _integrity_error(rec: Path, program_path: Path) -> str:
    """录完之后、拷回共享盘之前的一道检查。只查能从产物本身查出来的事实。

    录制器自己已经校验采样计划，这里查的是另外三件它不管的事：真值行数和它自报的帧数
    对不对得上、真值里的物体是不是程序里那一批、每一帧是不是都带了物体状态。
    """
    try:
        meta = json.loads((rec / "record.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in
                (rec / "gt_states.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        return f"产物读不出来：{error}"[:200]

    if len(rows) != meta.get("frames"):
        return f"真值 {len(rows)} 行，record.json 说 {meta.get('frames')} 帧"
    if not rows:
        return "真值是空的"

    program = json.loads(program_path.read_text(encoding="utf-8"))
    expected = {node["id"] for node in program.get("static", []) + program.get("objects", [])}
    for index, row in enumerate(rows):
        objects = row.get("objects")
        if not objects:
            return f"第 {index} 帧没有物体状态"
        seen = {o.get("id") for o in objects}
        if seen != expected:
            return f"第 {index} 帧的物体集合对不上：多 {sorted(seen - expected)} 少 {sorted(expected - seen)}"
    return ""


def render_one(program_path: Path, out_dir: Path, work: Path, width: int, height: int,
               fps: int, duration: float, keep_frames: bool) -> dict:
    """返回这一条的耗时拆解。失败不抛，记进 row 里让上层继续。"""
    row: dict = {"clip": program_path.parent.name, "ok": False}
    game, rec = work / "game", work / "rec"
    for path in (game, rec):
        shutil.rmtree(path, ignore_errors=True)

    t0 = time.time()
    compiled = subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(program_path), str(game)],
                              cwd=REPO, capture_output=True, text=True)
    row["compile_s"] = round(time.time() - t0, 3)
    if compiled.returncode != 0:
        row["error"] = f"compile: {compiled.stderr[-400:]}"
        return row

    t1 = time.time()
    recorded = subprocess.run([str(HARNESS), "record.mjs", "--game", str(game), "--out", str(rec),
                               "--fps", str(fps), "--duration", str(duration),
                               "--width", str(width), "--height", str(height)],
                              cwd=REPO, capture_output=True, text=True)
    row["record_s"] = round(time.time() - t1, 3)
    if recorded.returncode != 0:
        row["error"] = f"record: {recorded.stderr[-400:]}"
        return row

    # 写出之前先验一遍。2000 条规模上，坏样本要在这一刻拦住——等到训练时才发现，
    # 已经分不清是数据坏了还是模型学不会。
    bad = _integrity_error(rec, program_path)
    if bad:
        row["error"] = f"integrity: {bad}"
        return row

    out_dir.mkdir(parents=True, exist_ok=True)
    t2 = time.time()
    video = out_dir / "clip.mp4"
    encoded = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                              "-i", str(rec / "frames" / "f_%05d.png"),
                              "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                             capture_output=True, text=True)
    row["encode_s"] = round(time.time() - t2, 3)
    if encoded.returncode != 0:
        row["error"] = f"ffmpeg: {encoded.stderr[-400:]}"
        return row

    for name in ("gt_states.jsonl", "record.json"):
        if (rec / name).is_file():
            shutil.copy2(rec / name, out_dir / name)
    shutil.copy2(program_path, out_dir / "program.json")
    if keep_frames:
        shutil.copytree(rec / "frames", out_dir / "frames", dirs_exist_ok=True)

    row.update(ok=True, video_bytes=video.stat().st_size,
               gt_bytes=(out_dir / "gt_states.jsonl").stat().st_size if (out_dir / "gt_states.jsonl").is_file() else 0,
               total_s=round(row["compile_s"] + row["record_s"] + row["encode_s"], 3))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="generate_dataset 的输出目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()

    manifest = json.loads((Path(a.root) / "manifest.json").read_text(encoding="utf-8"))
    clips = [c["clip"] for c in manifest["clips"]]
    if a.limit:
        clips = clips[:a.limit]
    clips = [c for i, c in enumerate(clips) if i % a.shards == a.shard]

    work = _work_root()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, clip in enumerate(clips, 1):
        row = render_one(Path(a.root) / clip / "program.json", out / clip, work,
                         a.width, a.height, int(manifest["fps"]), float(manifest["duration_s"]),
                         a.keep_frames)
        rows.append(row)
        status = "ok" if row["ok"] else row.get("error", "failed")[:80]
        print(f"[{i}/{len(clips)}] {clip} {row.get('total_s', '-')}s {status}", flush=True)

    done = [r for r in rows if r["ok"]]
    summary = {
        "root": a.root, "width": a.width, "height": a.height,
        "shard": a.shard, "shards": a.shards,
        "clips": len(rows), "ok": len(done), "wall_s": round(time.time() - t0, 1),
        "per_clip_s": round(sum(r["total_s"] for r in done) / len(done), 2) if done else None,
        "per_clip_record_s": round(sum(r["record_s"] for r in done) / len(done), 2) if done else None,
        "per_clip_video_bytes": round(sum(r["video_bytes"] for r in done) / len(done)) if done else None,
        "rows": rows,
    }
    (out / f"render_report_{a.shard}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    return 0 if len(done) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
