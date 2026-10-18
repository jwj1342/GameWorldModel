#!/usr/bin/env python
"""小批量端到端：合成片段跑完整条链路，出一张表。

这是 #5 里说的「先五十条跑通再谈两千条」的那一步。合成片段的好处是每条都有标准答案，
所以除了场景完整度之外，还能直接算重建和真值的差距。

检测短语取自真值程序的类名。这是**命名这一步的上界**，不是生产路径——生产路径用视觉
语言模型来命名。这么选是因为这一轮要量的是几何、资产和摆位的质量，把命名的误差混进来
会看不清哪一层的问题。命名本身的代价之前单独量过。

用法:
  python -m scripts.pilot_report --rendered $GWM_SCRATCH/dataset/pilot50_rendered \
         --out $GWM_SCRATCH/dataset/pilot50_report --limit 50
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def gt_phrases(program: dict, limit: int = 20) -> list[str]:
    classes = {str(n.get("class", "")).strip().lower()
               for n in program.get("objects", []) + program.get("static", [])}
    return sorted(c for c in classes if c)[:limit]


def run_one(clip_dir: Path, out_dir: Path, extra: list[str]) -> dict:
    program = json.loads((clip_dir / "program.json").read_text(encoding="utf-8"))
    phrases = gt_phrases(program)
    target = out_dir / clip_dir.name
    cmd = [sys.executable, "-m", "scripts.rebuild_with_assets",
           "--video", str(clip_dir / "clip.mp4"), "--clip", clip_dir.name,
           "--out", str(target), "--phrases", ",".join(phrases), *extra]
    started = time.time()
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600)
    row = {"clip": clip_dir.name, "ok": done.returncode == 0,
           "seconds": round(time.time() - started, 1), "gt_objects": len(program.get("objects", []))}
    report = target / "rebuild.json"
    if report.is_file():
        info = json.loads(report.read_text(encoding="utf-8"))
        fidelity = info["completeness"]["fidelity"]
        inter = info["completeness"].get("interactivity") or {}
        # 实测可交互性只有带 --probe-interaction 跑的时候才有。没有就留空，不要拿
        # 声明值顶上——这两个数差多少正是这一项存在的理由。
        row["measured_interactive"] = inter.get("measured_success_fraction")
        row["style_swap_deviation"] = inter.get("style_swap_max_deviation")
        row.update(objects=info["objects"], assets_resolved=info["assets_resolved"],
                   asset_coverage=fidelity.get("asset_coverage"),
                   pixel_coverage=fidelity.get("evidence_pixel_coverage"),
                   size_log_error=fidelity.get("size_log_error_median"),
                   object_recall=fidelity.get("object_recall"),
                   metric_scale=info["metric_scale"].get("factor"),
                   resized=len(info.get("resized") or []),
                   stabilised=sum(1 for r in info.get("stabilised") or [] if r.get("applied")))
    else:
        row["error"] = (done.stderr or done.stdout or "")[-400:]
    return row


def summarise(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("asset_coverage") is not None]
    def med(key):
        values = [r[key] for r in done if isinstance(r.get(key), (int, float))]
        return round(statistics.median(values), 3) if values else None
    return {"clips": len(rows), "ok": len(done),
            "median_objects": med("objects"), "median_gt_objects": med("gt_objects"),
            "median_asset_coverage": med("asset_coverage"),
            "median_pixel_coverage": med("pixel_coverage"),
            "median_size_log_error": med("size_log_error"),
            "median_measured_interactive": med("measured_interactive"),
            "median_metric_scale": med("metric_scale"),
            "median_seconds": med("seconds"),
            "total_hours": round(sum(r["seconds"] for r in rows) / 3600, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rebuild-arg", action="append", default=[],
                    help="透传给 rebuild_with_assets 的额外参数。值本身以 - 开头时要用等号写，"
                         "比如 --rebuild-arg=--no-settle，分开写 argparse 会当成另一个选项")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in Path(a.rendered).iterdir() if (p / "clip.mp4").is_file())[:a.limit]
    rows = []
    for i, clip_dir in enumerate(clips, 1):
        row = run_one(clip_dir, out, a.rebuild_arg)
        rows.append(row)
        print(f"[{i}/{len(clips)}] {row['clip']:<16} {row['seconds']:>6}s "
              f"物体 {row.get('objects', '-'):>3}/{row['gt_objects']:<3} "
              f"资产 {row.get('asset_coverage', '-')} 画面 {row.get('pixel_coverage', '-')}"
              f"{' 能拖 ' + str(row['measured_interactive']) if row.get('measured_interactive') is not None else ''}"
              f"{'' if row['ok'] else '  失败'}", flush=True)
        (out / "pilot.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    summary = summarise(rows)
    (out / "pilot.json").write_text(json.dumps({"summary": summary, "rows": rows},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
