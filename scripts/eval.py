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
帧率、内核或者录制器版本都会重录，录到一半断掉的也不会被当成完整结果。
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys
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


def recorder_identity() -> str:
    """录制器及共享启动逻辑决定状态时间语义，也必须参与缓存身份。"""
    h = hashlib.sha256()
    paths = sorted((REPO / "harness").glob("*.mjs"))
    paths += [REPO / "harness" / "package.json", REPO / "scripts" / "node_harness.sh"]
    for p in paths:
        h.update(p.relative_to(REPO).as_posix().encode("utf-8"))
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()


def _display_path(path: Path) -> str:
    """仓库内的写相对路径，仓库外（比如 $SCRATCH 上的数据集）写绝对路径。

    仓库内那条统一用正斜杠：它会进 run.json 和报告，是要跨机器比对的标识，
    不该因为在 Windows 上跑一遍就变成反斜杠。仓库外是本机绝对路径，保持原样。
    """
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def gt_roots(extra: list[str] | None = None) -> list[Path]:
    """真值根目录。仓库里那两个是默认，额外的由调用方显式给。

    合成数据集产在 $SCRATCH 上，不在仓库里，所以必须能从外面指进来。
    但不从 run.json 里读——那里记的 `gt_program` 是这次运行自己用过的东西，
    拿它当真值等于自己给自己判卷。
    """
    out = [REPO / d for d in GT_DIRS]
    for d in extra or []:
        path = Path(d)
        out.append(path if path.is_absolute() else REPO / path)
    return out


def find_ground_truth(clip: str, extra_roots: list[str] | None = None) -> tuple[Path | None, str, list[str]]:
    """返回 (真值程序路径, 匹配依据, 所有候选)。

    有多份候选同时匹配就不返回——两份不同的真值选错一份，算出来的误差是假的，
    而且从表上完全看不出来。宁可标成不可评测。
    """
    roots = gt_roots(extra_roots)
    exact = [root / clip / "program.json" for root in roots]
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
                if json.loads(p.read_text(encoding="utf-8")).get("meta", {}).get("clip") == clip:
                    declared.append(p)
            except (OSError, ValueError):
                continue
    if len(declared) == 1:
        return declared[0], "program_meta_clip", [str(declared[0])]
    if len(declared) > 1:
        return None, "ambiguous_program_meta_clip", [str(p) for p in declared]
    return None, "not_found", []


def record_key(program_bytes: bytes, duration: float, fps: int, *, include_endpoint: bool = False) -> dict:
    if type(include_endpoint) is not bool:
        raise TypeError("include_endpoint must be a boolean")
    return {"program_sha256": _sha256_bytes(program_bytes), "duration_s": float(duration),
            "fps": int(fps), "width": RECORD_W, "height": RECORD_H,
            "include_endpoint": include_endpoint,
            "kernel_sha256": kernel_identity(), "recorder_sha256": recorder_identity()}


def _cached(rec: Path, key: dict) -> bool:
    """缓存只有在输入一致、而且上次确实录完了的情况下才算数。"""
    states, stamp, meta = rec / "gt_states.jsonl", rec / "eval_record_key.json", rec / "record.json"
    if not (states.is_file() and stamp.is_file() and meta.is_file()):
        return False
    try:
        if json.loads(stamp.read_text(encoding="utf-8")) != key:
            return False
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        frames = metadata.get("frames")
        # 旧录制未记此字段时按默认 false；字符串 "false" 不能当布尔值。
        endpoint = metadata.get("include_endpoint", False)
        requested_endpoint = key.get("include_endpoint", False)
        if type(endpoint) is not bool or type(requested_endpoint) is not bool or endpoint != requested_endpoint:
            return False
        # 与 record.mjs 的 Math.round(duration * fps) 一致，不能用 Python 的银行家舍入。
        duration, fps = float(key["duration_s"]), float(key["fps"])
        if not math.isfinite(duration * fps) or duration <= 0 or fps <= 0:
            return False
        expected = math.floor(duration * fps + 0.5)
        if endpoint:
            # 逐项复现 record.mjs：空序列先采 0，再按相同容差补终点。
            expected = max(1, expected)
            if abs((expected - 1) / fps - duration) > 1e-9:
                expected += 1
        if type(frames) is not int or frames <= 0 or frames != expected:
            return False
        with states.open(encoding="utf-8") as fh:
            written = sum(1 for line in fh if line.strip())
    except (OSError, ValueError, KeyError, TypeError, OverflowError, AttributeError):
        return False
    return written == frames        # 录到一半断掉的不算完整


def states_of(program_path: Path, out: Path, duration: float, harness: str, fps: int = 10,
              *, include_endpoint: bool = False) -> list[dict]:
    """轨迹不在 Python 里重算，走内核录一遍。运动语义只有内核那一份实现。"""
    out.mkdir(parents=True, exist_ok=True)
    game, rec = out / "game", out / "rec"
    key = record_key(program_path.read_bytes(), duration, fps, include_endpoint=include_endpoint)
    if not _cached(rec, key):
        (rec / "eval_record_key.json").unlink(missing_ok=True)
        subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(program_path), str(game)],
                       check=True, cwd=REPO, capture_output=True)
        subprocess.run([harness, "record.mjs", "--game", str(game), "--out", str(rec),
                        "--fps", str(fps), "--duration", str(duration),
                        "--width", str(RECORD_W), "--height", str(RECORD_H)] +
                       (["--include-endpoint", "true"] if include_endpoint else []),
                       check=True, cwd=REPO, capture_output=True)
        rec.mkdir(parents=True, exist_ok=True)
        (rec / "eval_record_key.json").write_text(json.dumps(key, sort_keys=True) + "\n", encoding="utf-8")
        if not _cached(rec, key):
            # 录制进程退 0 也不代表录完了，帧数对不上就当失败，别把半截结果留着复用
            raise RuntimeError(f"录制没有产出完整的状态序列：{rec}")
    return load_states(rec / "gt_states.jsonl")


def work_dir(work: Path, tag: str, identity: str) -> Path:
    """目录名带上身份摘要。只用 run.name 的话，两个不同片段里同名的运行会共用缓存。"""
    return work / f"{tag}_{hashlib.sha256(identity.encode()).hexdigest()[:12]}"


def eval_one(run: Path, work: Path, harness: str, gt_roots_extra: list[str] | None = None,
             max_dist_m: float | None = None, match_on: str | None = None) -> dict:
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    clip = manifest.get("clip", run.parent.name)
    configs = (manifest.get("args") or {}).get("config") or []
    cfg_identity = manifest.get("config_identity")
    row = {"run": str(run.resolve()), "clip": clip, "seed": manifest.get("seed"),
           "config": config_label(configs, cfg_identity), "config_files": list(configs),
           "config_identity": cfg_identity, "video_sha256": manifest.get("video_sha256")}

    gt_path, link, candidates = find_ground_truth(clip, gt_roots_extra)
    row["truth_link"] = link
    if gt_path is None:
        row["skipped"] = ("找到多份都匹配的真值程序，不猜：" + ", ".join(candidates)) if candidates \
            else "没有真值程序（真实视频本来就没有，不是错误）"
        return row

    gt_bytes = gt_path.read_bytes()
    row["ground_truth"] = {"path": _display_path(gt_path), "sha256": _sha256_bytes(gt_bytes),
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
    # 配对容差默认 3 米，是绝对量。场景尺度变了它就不再等价——在一个 3 米见方的场景里
    # 3 米等于「什么都能配上」，在 21 米的场景里又很紧。跨尺度比较必须把它按场景一起缩，
    # 否则比出来的是口径差异不是方法差异。不给就沿用原来的默认值。
    extra = {} if max_dist_m is None else {"max_dist_m": float(max_dist_m)}
    # 配对判据。默认沿用历史口径（整段平均距离），但那个口径把被评测的量写进了配对
    # 条件——运动判错的物体配不上，于是既算漏检又被踢出运动准确率的分母。详见 #7。
    if match_on is not None:
        extra["match_on"] = match_on
    row["metrics"] = evaluate(gt, json.loads(pred_bytes), gt_states, pred_states, **extra)
    if max_dist_m is not None:
        row["max_dist_m"] = float(max_dist_m)
    if match_on is not None:
        row["match_on"] = match_on
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
    ap.add_argument("--gt-root", action="append", default=[],
                    help="额外的真值根目录，可给多次。合成数据集产在 $SCRATCH 上，不在仓库里，"
                         "所以要从外面指进来；不从 run.json 读，那是这次运行自己用过的东西")
    ap.add_argument("--out", default="docs/results/eval_table.json")
    ap.add_argument("--match-on", choices=["mean", "start"], default=None,
                    help="物体配对按什么算距离。mean（默认，历史口径）是观察窗内逐时刻"
                         "距离的平均；它把被评测的量写进了配对条件——运动判错的物体配不上，"
                         "于是既算漏检、又被踢出运动准确率的分母（实测这是个闭环，见 #7）。"
                         "start 只看观察窗起点的位置，和运动无关。")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="物体配对的距离容差（米），默认 3.0。它是绝对量，所以跨尺度"
                         "比较时必须按场景尺度一起缩，否则小场景里等于「什么都能配上」。")
    a = ap.parse_args(argv)

    given = [Path(r) for r in a.run]
    if a.runs_from:
        given += [Path(l.strip()) for l in Path(a.runs_from).read_text(encoding="utf-8").splitlines() if l.strip()]
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
            # 只在真的给了容差时才往下传：测试里有用三到四个位置参数替换 eval_one 的，
            # 无条件多传一个会把它们打断，而那些测试盯的是入口行为不是签名
            tol = {} if a.max_dist is None else {"max_dist_m": a.max_dist}
            if a.match_on is not None:
                tol["match_on"] = a.match_on
            row = eval_one(r, work, a.harness, a.gt_root, **tol)
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
                              ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n写到 {out}")
    # 一条都没评出来还退 0 的话，作业脚本会把一次全盘失败当成成功。
    if failed or not rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
