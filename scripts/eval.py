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

名字只用来定位，不足以证明这份真值就是这段视频的来源，所以每一行都记下匹配
是靠什么建立的（`truth_link`）。靠名字匹配上的进表但单独标出来，有多份候选
同时匹配的直接拒绝，不猜。

录制结果会缓存复用，但缓存按输入内容认，不按文件在不在认：换了程序、时长、
帧率或者内核版本都会重录，录到一半断掉的也不会被当成完整结果。
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

from gwm.config import REPO
from gwm.feedback.gt_metrics import evaluate, load_states
from gwm.feedback.report import compare, as_markdown

GT_DIRS = ["examples", "data/ground_truth"]
RECORD_W, RECORD_H = 320, 180


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def kernel_identity() -> str:
    """内核改了，同一份程序录出来的轨迹就可能不一样，所以它要进缓存键。"""
    h = hashlib.sha256()
    for p in sorted((REPO / "kernel").rglob("*.js")):
        h.update(p.relative_to(REPO).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def find_ground_truth(clip: str) -> tuple[Path | None, str, list[str]]:
    """返回 (真值程序路径, 匹配依据, 所有候选)。

    有多份候选同时匹配就不返回——两份不同的真值选错一份，算出来的误差是假的，
    而且从表上完全看不出来。宁可标成不可评测。
    """
    exact = [REPO / d / clip / "program.json" for d in GT_DIRS]
    exact = [p for p in exact if p.exists()]
    if len(exact) == 1:
        return exact[0], "directory_name", [str(exact[0])]
    if len(exact) > 1:
        return None, "ambiguous_directory_name", [str(p) for p in exact]

    # 合成调试场景的目录名和 clip 名可能不一致，再按 meta.clip 找一遍
    declared = []
    for d in GT_DIRS:
        for p in sorted((REPO / d).glob("*/program.json")):
            try:
                if json.loads(p.read_text()).get("meta", {}).get("clip") == clip:
                    declared.append(p)
            except (OSError, ValueError):
                continue
    if len(declared) == 1:
        return declared[0], "program_meta_clip", [str(declared[0])]
    if len(declared) > 1:
        return None, "ambiguous_program_meta_clip", [str(p) for p in declared]
    return None, "not_found", []


def record_key(program_bytes: bytes, duration: float, fps: int) -> dict:
    return {"program_sha256": _sha256_bytes(program_bytes), "duration_s": float(duration),
            "fps": int(fps), "width": RECORD_W, "height": RECORD_H,
            "kernel_sha256": kernel_identity()}


def _cached(rec: Path, key: dict) -> bool:
    """缓存只有在输入一致、而且上次确实录完了的情况下才算数。"""
    states, stamp, meta = rec / "gt_states.jsonl", rec / "eval_record_key.json", rec / "record.json"
    if not (states.is_file() and stamp.is_file() and meta.is_file()):
        return False
    try:
        if json.loads(stamp.read_text()) != key:
            return False
        frames = json.loads(meta.read_text()).get("frames")
    except (OSError, ValueError):
        return False
    if not isinstance(frames, int) or frames <= 0:
        return False
    with states.open() as fh:
        written = sum(1 for line in fh if line.strip())
    return written == frames        # 录到一半断掉的不算完整


def states_of(program_path: Path, out: Path, duration: float, harness: str, fps: int = 10) -> list[dict]:
    """轨迹不在 Python 里重算，走内核录一遍。运动语义只有内核那一份实现。"""
    out.mkdir(parents=True, exist_ok=True)
    game, rec = out / "game", out / "rec"
    key = record_key(program_path.read_bytes(), duration, fps)
    if not _cached(rec, key):
        (rec / "eval_record_key.json").unlink(missing_ok=True)
        subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(program_path), str(game)],
                       check=True, cwd=REPO, capture_output=True)
        subprocess.run([harness, "record.mjs", "--game", str(game), "--out", str(rec),
                        "--fps", str(fps), "--duration", str(duration),
                        "--width", str(RECORD_W), "--height", str(RECORD_H)],
                       check=True, cwd=REPO, capture_output=True)
        rec.mkdir(parents=True, exist_ok=True)
        (rec / "eval_record_key.json").write_text(json.dumps(key, sort_keys=True) + "\n")
        if not _cached(rec, key):
            # 录制进程退 0 也不代表录完了，帧数对不上就当失败，别把半截结果留着复用
            raise RuntimeError(f"录制没有产出完整的状态序列：{rec}")
    return load_states(rec / "gt_states.jsonl")


def work_dir(work: Path, tag: str, identity: str) -> Path:
    """目录名带上身份摘要。只用 run.name 的话，两个不同片段里同名的运行会共用缓存。"""
    return work / f"{tag}_{hashlib.sha256(identity.encode()).hexdigest()[:12]}"


def eval_one(run: Path, work: Path, harness: str) -> dict:
    manifest = json.loads((run / "run.json").read_text())
    clip = manifest.get("clip", run.parent.name)
    configs = (manifest.get("args") or {}).get("config") or []
    cfg_identity = manifest.get("config_identity")
    row = {"run": str(run.resolve()), "clip": clip, "seed": manifest.get("seed"),
           "config": config_label(configs, cfg_identity), "config_files": list(configs),
           "config_identity": cfg_identity, "video_sha256": manifest.get("video_sha256")}

    gt_path, link, candidates = find_ground_truth(clip)
    row["truth_link"] = link
    if gt_path is None:
        row["skipped"] = ("找到多份都匹配的真值程序，不猜：" + ", ".join(candidates)) if candidates \
            else "没有真值程序（真实视频本来就没有，不是错误）"
        return row

    gt_bytes = gt_path.read_bytes()
    row["ground_truth"] = {"path": str(gt_path.relative_to(REPO)), "sha256": _sha256_bytes(gt_bytes),
                           "link": link,
                           "verified": False,
                           "note": "按名字定位的，没有证据表明这段视频确实由这份程序渲染而来"}
    gt = json.loads(gt_bytes)
    duration = float(gt.get("meta", {}).get("duration") or 8.0)
    pred_bytes = (run / "program.json").read_bytes()
    gt_states = states_of(gt_path, work_dir(work, f"{clip}_gt", row["ground_truth"]["sha256"]),
                          duration, harness)
    pred_states = states_of(run / "program.json",
                            work_dir(work, f"{run.name}_pred", row["run"]), duration, harness)
    row["metrics"] = evaluate(gt, json.loads(pred_bytes), gt_states, pred_states)
    return row


def config_label(files: list[str], identity: str | None) -> str:
    """分组用的名字要能区分内容。

    只取文件名主干的话，`configs/ablations/no_feedback.yaml` 和别处同名的另一份
    会被并进同一组，表里两行其实混了两个实验。所以同一主干不同内容时带上摘要。
    """
    stem = ",".join(Path(c).stem for c in files) or "default"
    return f"{stem}@{identity[:8]}" if identity else f"{stem}@unknown"


def usable(row: dict) -> bool:
    """算出数来才叫可用。evaluate 自己判不了的会返回 error，不能当成零误差。"""
    metrics = row.get("metrics")
    return isinstance(metrics, dict) and "error" not in metrics and isinstance(metrics.get("full"), dict)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="一个运行目录，可以给多次")
    ap.add_argument("--runs-from", default=None, help="一个文件，每行一个运行目录")
    ap.add_argument("--work", required=True, help="中间产物放哪，用 $SLURM_TMPDIR")
    ap.add_argument("--harness", default=str(REPO / "scripts" / "node_harness.sh"))
    ap.add_argument("--group-by", default="config", choices=["config", "clip"])
    ap.add_argument("--out", default="docs/results/eval_table.json")
    a = ap.parse_args(argv)

    given = [Path(r) for r in a.run]
    if a.runs_from:
        given += [Path(l.strip()) for l in Path(a.runs_from).read_text().splitlines() if l.strip()]
    if not given: ap.error("至少要给一个 --run 或 --runs-from")

    runs, seen = [], set()
    for r in given:                                  # 同一个目录给两遍不该在表里占两行
        key = str(r.resolve())
        if key not in seen: seen.add(key); runs.append(r)
    duplicates = len(given) - len(runs)

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rows, skipped, failed = [], [], []
    for r in runs:
        try:
            row = eval_one(r, work, a.harness)
        except Exception as e:
            failed.append({"run": str(r), "error": repr(e)[:300]}); continue
        if "skipped" in row:
            skipped.append(row); continue
        if not usable(row):
            reason = (row.get("metrics") or {}).get("error") or "指标没有产出 full 窗口"
            failed.append({"run": str(r), "clip": row["clip"], "error": f"指标不可评测：{reason}"})
            continue
        rows.append(row)
        w = row["metrics"]["full"]
        t = (w.get("trajectory") or {}).get("median_m")
        print(f"  {row['clip']:24s} {row['config']:24s} 轨迹中位 "
              f"{'无法匹配' if t is None else f'{t:.3f} m'}  "
              f"类型 {(w.get('motion_type') or {}).get('accuracy')}  "
              f"召回 {(w.get('objects') or {}).get('recall')}  来源 {row['truth_link']}")

    groups: dict[str, list[dict]] = {}
    for row in rows: groups.setdefault(row[a.group_by], []).append(row)
    table_full = compare(groups, "full")
    table_hold = compare(groups, "holdout")

    print("\n整段：\n" + as_markdown(table_full))
    print("\n留出段（视频末尾 20%）：\n" + as_markdown(table_hold))
    if skipped:
        print(f"\n跳过 {len(skipped)} 条（没有真值，预期内）：")
        for s in skipped: print(f"  {s['run']}  {s['skipped']}")
    if failed:
        print(f"\n失败 {len(failed)} 条：")
        for s in failed: print(f"  {s['run']}  {s['error']}")
    if duplicates:
        print(f"\n（去掉了 {duplicates} 条重复给出的运行目录）")
    name_only = sorted({r["clip"] for r in rows if r["truth_link"] in ("directory_name", "program_meta_clip")})
    if name_only:
        print(f"\n注意：{len(name_only)} 个片段的真值是按名字定位的，没有来源证据："
              f"{', '.join(name_only)}")

    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"full": table_full, "holdout": table_hold,
                               "rows": rows, "skipped": skipped, "failed": failed,
                               "duplicate_inputs": duplicates,
                               "truth_link_unverified": name_only,
                               "markdown_full": as_markdown(table_full),
                               "markdown_holdout": as_markdown(table_hold)},
                              ensure_ascii=False, indent=1) + "\n")
    print(f"\n写到 {out}")
    # 一条都没评出来还退 0 的话，作业脚本会把一次全盘失败当成成功。
    if failed or not rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
