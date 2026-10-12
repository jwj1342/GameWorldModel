#!/usr/bin/env python3
"""无真值的奖励信号，和有真值的指标，排序一不一致（issue #17）。

为什么要问这个：#17 想用渲染比对的分数当训练信号，因为它在真实视频上也能算，
不需要标注。但 #4 指出那个分数是拿程序和感知结果比的，而感知本身有误差。
于是 #17 的前提就悬着——「等 #4 做完奖励问题就解决了」在逻辑上不成立，
因为真实视频根本没有真值可用。

本入口仅作探索：分别统计两族扰动候选，不将不同基准的分数混池。
源视频/真值时间对应、坐标约定及身份映射尚未核验时，正式结论必须为无法判断。
一次相关性不能证明真实视频训练有效，也不能证明某种信号永远不可用。
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, subprocess, sys
from pathlib import Path

from scipy.stats import spearmanr

from gwm.config import REPO, load_config, site_name
from gwm.feedback.gt_metrics import evaluate, load_states


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def assess_agreement(rows: list[dict], min_samples: int = 3) -> dict:
    """Descriptive ranks within one family; callers must verify the experiment protocol separately."""
    if type(min_samples) is not int or min_samples < 3:
        raise ValueError("min_samples must be an integer >= 3")
    included, excluded = [], []
    families = {r.get("family") for r in rows}
    if len(families) > 1:
        raise ValueError("different candidate families must not be pooled")
    for index, row in enumerate(rows):
        reasons = []
        for field, status in (("reward_no_gt", "reward_status"), ("gt_traj_median_m", "gt_status")):
            if row.get(status) != "available":
                reasons.append(status + ":" + str(row.get(status, "unknown")))
            elif not _finite(row.get(field)):
                reasons.append(field + ":non_finite_or_missing")
        if reasons:
            excluded.append({"index": index, "label": row.get("label"), "reasons": reasons})
        else:
            included.append(index)
    report = {"status": "unavailable", "reason": "insufficient_samples", "n_comparable": len(included),
              "included_indices": included, "excluded": excluded, "spearman_rho": None, "p_value": None}
    if len(included) < min_samples:
        return report
    xs = [rows[i]["reward_no_gt"] for i in included]
    ys = [rows[i]["gt_traj_median_m"] for i in included]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        report["reason"] = "constant_reward" if len(set(xs)) < 2 else "constant_gt_error"
        return report
    rho, p = spearmanr(xs, ys)
    if not math.isfinite(float(rho)) or not math.isfinite(float(p)):
        report["reason"] = "non_finite_statistic"
        return report
    report.update(status="available", reason="", spearman_rho=float(rho), p_value=float(p))
    return report


def reward_measurement(metrics: dict) -> tuple[float | None, str]:
    """Conservative availability: no observations or unresolved ID mappings are not measured zero."""
    summary = metrics.get("summary", {})
    objects = summary.get("per_object", {})
    if not objects:
        return None, "no_observations"
    if any(obj.get("program_id") is None for obj in objects.values()):
        return None, "identity_unresolved"
    value = summary.get("score")
    return (float(value), "available") if _finite(value) else (None, "non_finite_score")


def graded_candidates(program: dict) -> list[tuple[str, dict]]:
    """受控扰动；不预设相对未知真值的质量顺序。"""
    out = [("原样", copy.deepcopy(program))]
    for shift in (0.15, 0.4, 0.8, 1.6):
        p = copy.deepcopy(program)
        for o in p["objects"]:
            if "pose" in o: o["pose"]["pos"][0] += shift
            for inst in o.get("instances", []): inst["pos"][0] += shift
        out.append((f"对象整体平移 {shift} 米", p))
    p = copy.deepcopy(program)
    for o in p["objects"]:
        if (o.get("motion") or {}).get("type") not in (None, "static"): o["motion"] = {"type": "static"}
    out.append(("所有运动改成静止", p))
    return out


class SourceRejected(ValueError):
    def __init__(self, resolution):
        self.resolution = resolution


def source_failure(code, path, hint):
    return {"ok": False, "status": "rejected", "diagnostics": [{
        "stage": "ground_truth_source", "severity": "error", "code": code, "path": path, "hint": hint}]}


def metric_frames(evidence):
    """Adapt existing frame references to the legacy metric API, never invent indices."""
    frames = evidence.get("frames") or evidence.get("keyframes") or []
    normalized, seen = [], set()
    for frame in frames:
        index = frame.get("frame_index", frame.get("index"))
        if (type(index) is not int or index < 0 or index in seen or not _finite(frame.get("t"))
                or ("index" in frame and frame["index"] != index)):
            raise SourceRejected(source_failure("SOURCE_INVALID_FRAME_REFERENCE", "/frames",
                "Frames require unique existing indices, consistent aliases and finite times"))
        seen.add(index)
        normalized.append({**frame, "index": index})
    if not normalized or any(a["t"] >= b["t"] for a, b in zip(normalized, normalized[1:])):
        raise SourceRejected(source_failure("SOURCE_INVALID_FRAME_REFERENCE", "/frames", "Frames must be time ordered"))
    by_index = {f["index"]: f for f in normalized}
    for keyframe in evidence.get("keyframes", []):
        index = keyframe.get("frame_index", keyframe.get("index"))
        if (type(index) is not int or index not in by_index or
                ("index" in keyframe and keyframe["index"] != index) or
                not _finite(keyframe.get("t")) or abs(keyframe["t"] - by_index[index]["t"]) > 1e-4):
            raise SourceRejected(source_failure("SOURCE_INVALID_FRAME_REFERENCE", "/keyframes",
                "Keyframes must reference the same registered frame and time"))
    return normalized


def prepare_sources(run, video, associations, duration):
    """Verify reviewed content identity before any compilation or rendering."""
    from gwm.feedback.ground_truth_source import resolve_ground_truth
    if not video or not associations or not _finite(duration) or duration <= 0:
        raise SourceRejected(source_failure("SOURCE_INPUT_REQUIRED", "/",
            "Supply --video, --associations and a finite positive --duration; no filename fallback"))
    resolution = resolve_ground_truth(video, associations, (0, duration))
    if not resolution["ok"]:
        raise SourceRejected(resolution)
    ev_path, pred_path = run / "perception/evidence.json", run / "program.json"
    ev_bytes, pred_bytes = ev_path.read_bytes(), pred_path.read_bytes()
    ev = json.loads(ev_bytes)
    pipeline = json.loads(pred_bytes)
    claims = [ev.get("meta", {}).get(k) for k in ("source_video_sha256", "video_sha256")
              if ev.get("meta", {}).get(k) not in (None, "unknown")]
    if not claims or any(c != resolution["sources"]["video_sha256"] for c in claims):
        raise SourceRejected(source_failure("SOURCE_EVIDENCE_VIDEO_MISMATCH", "/meta",
            "Evidence must declare the verified video hash; missing/conflicting claims are rejected"))
    if not _finite(pipeline.get("meta", {}).get("duration")) or pipeline["meta"]["duration"] < duration:
        raise SourceRejected(source_failure("SOURCE_PREDICTION_TOO_SHORT", "/meta/duration",
            "Prediction duration must cover the explicit target interval"))
    times = [k.get("t") for k in ev.get("keyframes", [])]
    if not times or any(not _finite(t) or not 0 <= t <= duration for t in times) or any(
            a >= b for a, b in zip(times, times[1:])):
        raise SourceRejected(source_failure("SOURCE_INVALID_FRAME_TIMES", "/keyframes",
            "Keyframe times must be finite, strictly increasing and within the target interval"))
    metric_frames(ev)
    gt_path = Path(resolution["ground_truth_path"])
    if gt_path.resolve() == pred_path.resolve():
        raise SourceRejected(source_failure("SOURCE_REFERENCE_IS_PREDICTION", "/ground_truth",
            "The current prediction file cannot serve as its own independent reference"))
    gt_bytes = gt_path.read_bytes()
    gt = json.loads(gt_bytes)
    fingerprints = {"evidence_sha256": hashlib.sha256(ev_bytes).hexdigest(),
                    "prediction_sha256": hashlib.sha256(pred_bytes).hexdigest()}
    if hashlib.sha256(gt_bytes).hexdigest() != resolution["sources"]["ground_truth_sha256"]:
        raise SourceRejected(source_failure("SOURCE_CHANGED_DURING_CHECK", "/ground_truth", "Reference changed"))
    return gt, pipeline, ev, resolution, fingerprints


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--video", help="Actual source video, not a clip-name lookup")
    ap.add_argument("--associations", help="Human-reviewed ground-truth association manifest")
    ap.add_argument("--duration", type=float, help="Explicit target interval [0, duration]")
    ap.add_argument("--harness", default=str(REPO / "scripts" / "node_harness.sh"))
    ap.add_argument("--out", help="Fresh report path; defaults to work/reward_agreement.json")
    a = ap.parse_args(argv)
    run, work = Path(a.run).resolve(), Path(a.work).resolve()
    out = (REPO / a.out).resolve() if a.out else work / "reward_agreement.json"
    if out.exists() or (work.exists() and any(work.iterdir())):
        ap.error("Use a fresh work directory and report path; do not overwrite prior experiments")
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {"rows": rows, "spearman_rho": None, "p_value": None, "n_comparable": 0,
              "agreement_by_family": {}, "monotonicity": {}, "diagnostics": [],
              "source_verification": {"status": "not_checked"}, "protocol_status": "not_verified",
              "conclusion": "目前无法判断奖励是否可靠或感知是否为唯一瓶颈；坐标及身份对应尚未核验。"}

    def finish(status, exit_code):
        report["status"] = status
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {out} ({status})")
        return exit_code

    stage = "source_input"
    try:
        gt, pipeline, ev, resolution, fingerprints = prepare_sources(
            run, a.video, a.associations, a.duration)
        report.update(source_verification=resolution, input_fingerprints=fingerprints)
        stage = "setup"
        cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"])
        from gwm.feedback.metrics import compute_metrics
        from gwm.feedback.render import render
        from gwm.perception.run import load_masks
        masks = load_masks(run / "perception")
        frames = metric_frames(ev)
        key_times = [k["t"] for k in ev["keyframes"]]

        def compile_at(prog, directory):
            directory.mkdir(parents=True, exist_ok=True)
            pj = directory / "program.json"
            pj.write_text(json.dumps(prog, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(pj), str(directory / "game")],
                           check=True, cwd=REPO, capture_output=True)

        def record_at(directory):
            subprocess.run([a.harness, "record.mjs", "--game", str(directory / "game"),
                            "--out", str(directory / "rec"), "--fps", "10",
                            "--duration", str(a.duration), "--width", "320", "--height", "180", "--include-endpoint"],
                           # Explicit closed-interval sampling; default recorder behavior stays unchanged.
                           check=True, cwd=REPO, capture_output=True)
            states = load_states(directory / "rec/gt_states.jsonl")
            if not states:
                raise ValueError("Empty recorded states")
            times = [s.get("t") for s in states]
            expected = [i / 10 for i in range(math.floor(a.duration * 10 + .5))] or [0.0]
            if abs(expected[-1] - a.duration) > 1e-9:
                expected.append(a.duration)
            if (any(not _finite(t) for t in times) or any(x >= y for x, y in zip(times, times[1:]))
                    or len(times) != len(expected) or any(abs(t - e) > 1e-6 for t, e in zip(times, expected))):
                raise ValueError("Recorded states do not cover the explicit interval")
            return states

        stage = "reference_compile"
        compile_at(gt, work / "gt")
        stage = "reference_record"
        gt_states = record_at(work / "gt")
    except SourceRejected as error:
        report["source_verification"] = error.resolution
        report["diagnostics"].extend(error.resolution["diagnostics"])
        return finish("source_rejected", 1)
    except Exception as error:
        report["diagnostics"].append({"stage": stage, "severity": "error", "code": "EXECUTION_FAILED",
                                      "path": "/", "hint": type(error).__name__})
        return finish("source_rejected" if stage == "source_input" else "execution_failed", 1)

    families = [("以管线程序为基准", pipeline), ("以真值程序为基准", gt)]
    for prefix, (family, base) in zip(("p", "g"), families):
        for index, (label, candidate) in enumerate(graded_candidates(base)):
            row = {"family": family, "label": label, "status": "execution_failed",
                   "reward_status": "not_checked", "gt_status": "not_checked",
                   "reward_no_gt": None, "reward_raw": None, "gt_traj_median_m": None,
                   "gt_motion_acc": None, "diagnostics": []}
            rows.append(row)
            directory = work / f"c{prefix}{index}"
            try:
                stage = "candidate_compile"; compile_at(candidate, directory)
                stage = "candidate_render"
                rendered = render(directory / "game", key_times, directory / "render",
                                  cfg["feedback"]["render_width"], cfg["feedback"]["render_height"],
                                  tuple(cfg["feedback"]["passes"]))
                stage = "reward_metrics"
                measured = compute_metrics(candidate, rendered, directory / "render", ev, frames, masks, cfg)
                row["reward_no_gt"], row["reward_status"] = reward_measurement(measured)
                raw = measured["summary"]["score"]
                row["reward_raw"] = raw if _finite(raw) else None
                stage = "candidate_record"; states = record_at(directory)
                stage = "gt_metrics"; truth = evaluate(gt, candidate, gt_states, states)
                if "error" in truth:
                    row["gt_status"] = "trajectory_unavailable"
                else:
                    traj, motion = truth["full"]["trajectory"]["median_m"], truth["full"]["motion_type"]["accuracy"]
                    row.update(gt_traj_median_m=traj if _finite(traj) else None,
                               gt_motion_acc=motion if _finite(motion) else None,
                               gt_status="available" if _finite(traj) else "trajectory_unavailable")
                row["status"] = "measured"
            except Exception as error:
                row["reward_status"] = row["gt_status"] = "execution_failed"
                row["reward_no_gt"] = row["gt_traj_median_m"] = row["gt_motion_acc"] = None
                row["diagnostics"].append({"stage": stage, "severity": "error", "code": "EXECUTION_FAILED",
                                          "path": f"/rows/{len(rows)-1}", "hint": type(error).__name__})

    # Recheck content identity after acquisition before accepting any statistic.
    try:
        _, _, _, after, current = prepare_sources(run, a.video, a.associations, a.duration)
        if after["sources"] != resolution["sources"] or current != fingerprints:
            raise SourceRejected(source_failure("SOURCE_CHANGED_DURING_ACQUISITION", "/sources", "Inputs changed"))
    except Exception as error:
        failure = error.resolution if isinstance(error, SourceRejected) else source_failure(
            "SOURCE_RECHECK_FAILED", "/sources", type(error).__name__)
        report["source_verification"] = failure
        report["diagnostics"].extend(failure["diagnostics"])
        for row in rows:
            row.update(status="source_rejected", reward_status="source_rejected", gt_status="source_rejected",
                       reward_no_gt=None, gt_traj_median_m=None, gt_motion_acc=None)
        return finish("source_rejected", 1)

    for family, _ in families:
        selected = [r for r in rows if r["family"] == family]
        report["agreement_by_family"][family] = assess_agreement(selected)
        xs = [r["reward_no_gt"] for r in selected[:5]]
        usable = len(xs) >= 2 and all(_finite(x) for x in xs)
        report["monotonicity"][family] = {"rewards": xs, "status": "available" if usable else "unavailable",
            "单调下降": (all(x >= y for x, y in zip(xs, xs[1:])) and len(set(xs)) > 1) if usable else None}
    report["n_comparable"] = sum(r["n_comparable"] for r in report["agreement_by_family"].values())
    report["diagnostics"].append({"stage": "reward_agreement", "severity": "warning",
        "code": "protocol_not_verified", "path": "/", "hint": "来源通过不代表坐标/身份或解码PTS已核验；相关系数仅供探索。"})
    failed = any(r["status"] == "execution_failed" for r in rows)
    return finish("partial_execution_failure" if failed else "exploratory_complete", 1 if failed else 0)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
