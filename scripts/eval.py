#!/usr/bin/env python3
"""评测入口：给一批运行目录，出一张论文能用的表。

把散落的三件事串起来——对每个运行目录算真值指标、按配置分组聚合、渲染成表格。
在这之前每次要出一个数都得手写一个作业脚本，而且结果散在不同 JSON 里没法横向比。

用法：
    python scripts/eval.py --run out/clipA/run1 --run out/clipB/run2 --work $SLURM_TMPDIR/ev
    python scripts/eval.py --runs-from runs.txt --group-by config

真值从哪来：每个运行目录的 run.json 里记着 clip 名，按 clip 名去
examples/<clip>/program.json 找真值程序。找不到就跳过并说明——
真实视频没有真值，这是预期内的，不是错误。
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

from gwm.config import REPO
from gwm.feedback.gt_metrics import evaluate, load_states
from gwm.feedback.report import compare, as_markdown

GT_DIRS = ["examples", "data/ground_truth"]


def find_ground_truth(clip: str) -> Path | None:
    for d in GT_DIRS:
        p = REPO / d / clip / "program.json"
        if p.exists(): return p
    # 合成调试场景的目录名和 clip 名可能不一致，再按 meta.clip 找一遍
    for d in GT_DIRS:
        for p in (REPO / d).glob("*/program.json"):
            try:
                if json.loads(p.read_text()).get("meta", {}).get("clip") == clip: return p
            except Exception:
                continue
    return None


def states_of(program_path: Path, out: Path, duration: float, harness: str, fps: int = 10) -> list[dict]:
    """轨迹不在 Python 里重算，走内核录一遍。运动语义只有内核那一份实现。"""
    out.mkdir(parents=True, exist_ok=True)
    game, rec = out / "game", out / "rec"
    if not (rec / "gt_states.jsonl").exists():
        subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(program_path), str(game)],
                       check=True, cwd=REPO, capture_output=True)
        subprocess.run([harness, "record.mjs", "--game", str(game), "--out", str(rec),
                        "--fps", str(fps), "--duration", str(duration), "--width", "320", "--height", "180"],
                       check=True, cwd=REPO, capture_output=True)
    return load_states(rec / "gt_states.jsonl")


def eval_one(run: Path, work: Path, harness: str) -> dict:
    manifest = json.loads((run / "run.json").read_text())
    clip = manifest.get("clip", run.parent.name)
    cfg_name = ",".join(Path(c).stem for c in (manifest.get("args") or {}).get("config", [])) or "default"
    gt_path = find_ground_truth(clip)
    row = {"run": str(run), "clip": clip, "config": cfg_name}
    if gt_path is None:
        row["skipped"] = "没有真值程序（真实视频本来就没有，不是错误）"
        return row
    gt = json.loads(gt_path.read_text())
    duration = float(gt.get("meta", {}).get("duration") or 8.0)
    gt_states = states_of(gt_path, work / f"{clip}_gt", duration, harness)
    pred_states = states_of(run / "program.json", work / f"{run.name}_pred", duration, harness)
    row["metrics"] = evaluate(gt, json.loads((run / "program.json").read_text()), gt_states, pred_states)
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="一个运行目录，可以给多次")
    ap.add_argument("--runs-from", default=None, help="一个文件，每行一个运行目录")
    ap.add_argument("--work", required=True, help="中间产物放哪，用 $SLURM_TMPDIR")
    ap.add_argument("--harness", default=str(REPO / "scripts" / "node_harness.sh"))
    ap.add_argument("--group-by", default="config", choices=["config", "clip"])
    ap.add_argument("--out", default="docs/results/eval_table.json")
    a = ap.parse_args(argv)

    runs = [Path(r) for r in a.run]
    if a.runs_from:
        runs += [Path(l.strip()) for l in Path(a.runs_from).read_text().splitlines() if l.strip()]
    if not runs: ap.error("至少要给一个 --run 或 --runs-from")

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for r in runs:
        try:
            row = eval_one(r, work, a.harness)
        except Exception as e:
            skipped.append({"run": str(r), "error": repr(e)[:200]}); continue
        (skipped if "skipped" in row else rows).append(row)
        if "skipped" not in row:
            w = row["metrics"]["full"]
            t = w["trajectory"]["median_m"]
            print(f"  {row['clip']:24s} {row['config']:16s} 轨迹中位 "
                  f"{'无法匹配' if t is None else f'{t:.3f} m'}  "
                  f"类型 {w['motion_type']['accuracy']}  召回 {w['objects']['recall']}")

    groups: dict[str, list[dict]] = {}
    for row in rows: groups.setdefault(row[a.group_by], []).append(row)
    table_full = compare(groups, "full")
    table_hold = compare(groups, "holdout")

    print("\n整段：\n" + as_markdown(table_full))
    print("\n留出段（视频末尾 20%，模型没看过）：\n" + as_markdown(table_hold))
    if skipped:
        print(f"\n跳过 {len(skipped)} 条：")
        for s in skipped: print(f"  {s['run']}  {s.get('skipped') or s.get('error')}")

    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"full": table_full, "holdout": table_hold,
                               "rows": rows, "skipped": skipped,
                               "markdown_full": as_markdown(table_full),
                               "markdown_holdout": as_markdown(table_hold)},
                              ensure_ascii=False, indent=1) + "\n")
    print(f"\n写到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
