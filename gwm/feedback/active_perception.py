"""Plan review-only requests for more video evidence from verification findings."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from ..perception.contract import UNKNOWN, validate_evidence
from ..perception.provenance import declared_video_sha256, evidence_sha256, file_sha256


DEFAULTS = {
    "max_requests": 12,
    "max_time_range_s": 4.0,
    "time_padding_s": 0.5,
    "merge_gap_s": 0.5,
    "max_keyframes_per_request": 3,
    "min_observation_frames": 2,
    "min_observation_coverage": 0.3,
}
VISIBILITY_SIGNALS = {"object_never_visible", "object_not_observed_in_available_id_frames",
                      "object_temporarily_not_visible"}
MOTION_SIGNALS = {"dynamic_object_not_moving"}
EXECUTION_CODES = {
    "missing_render_index", "invalid_frames", "invalid_frame", "invalid_frame_size",
    "too_few_render_frames", "frame_count_mismatch", "missing_render_pass",
    "missing_pass_file", "undecodable_pass", "pass_size_mismatch", "blank_pass",
    "unknown_render_id", "invalid_frame_time", "timestamp_mismatch",
    "state_time_mismatch", "non_monotonic_render_times", "missing_frame_state",
    "invalid_object_pose", "invalid_object_rotation", "state_id_mismatch",
    "invalid_browser_events", "insufficient_motion_samples", "trigger_motion_not_exercised",
    "object_visibility_unverifiable",
}
PRIORITY = {"high": 0, "medium": 1, "low": 2}


def _settings(config: dict | None) -> dict:
    if config is None:
        supplied = {}
    elif not isinstance(config, dict):
        raise ValueError("active perception config must be a mapping")
    else:
        supplied = config.get("active_perception", config)
    if not isinstance(supplied, dict):
        raise ValueError("active_perception config must be a mapping")
    values = {**DEFAULTS, **supplied}
    for key in ("max_requests", "max_keyframes_per_request", "min_observation_frames"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_time_range_s", "time_padding_s", "merge_gap_s", "min_observation_coverage"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if values["max_time_range_s"] == 0 or values["min_observation_coverage"] > 1:
        raise ValueError("max_time_range_s must be positive and min_observation_coverage must be <= 1")
    return values


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _issue(code: str, message: str, path: str, stage: str = "active_perception") -> dict:
    return {"stage": stage, "severity": "warning", "code": code, "path": path,
            "message": message, "hint": "review the source artifact before requesting more perception"}


def _signal(item: dict, source: str) -> dict:
    return {"source": source, "stage": item.get("stage", "verification"),
            "severity": item.get("severity", "warning"), "code": item["code"],
            "path": item.get("path", ""), "message": item.get("message", "")}


def _range(times: list[float], duration: float | None, settings: dict) -> dict | None:
    finite = sorted(float(t) for t in times if _number(t) and t >= 0)
    if not finite and duration is None:
        return None
    if not finite:
        finite = [duration / 2]
    start = max(0.0, finite[0] - settings["time_padding_s"])
    end = finite[-1] + settings["time_padding_s"]
    if duration is not None:
        start, end = min(start, duration), min(end, duration)
    if end - start > settings["max_time_range_s"]:
        center = (finite[0] + finite[-1]) / 2
        half = settings["max_time_range_s"] / 2
        start, end = max(0.0, center - half), center + half
        if duration is not None and end > duration:
            end = duration
            start = max(0.0, end - settings["max_time_range_s"])
    return {"start_s": round(start, 3), "end_s": round(end, 3)}


def _keyframes(frames: list[dict], interval: dict | None, limit: int) -> list[dict]:
    candidates = [frame for frame in frames if _number(frame.get("t")) and isinstance(frame.get("frame_index"), int)]
    if not candidates:
        return []
    candidates.sort(key=lambda frame: (frame["t"], frame["frame_index"]))
    if interval is None:
        selected = candidates[:limit]
    else:
        inside = [frame for frame in candidates if interval["start_s"] <= frame["t"] <= interval["end_s"]]
        pool = inside or sorted(candidates, key=lambda frame: (abs(frame["t"] - (interval["start_s"] + interval["end_s"]) / 2), frame["frame_index"]))[:1]
        if len(pool) <= limit:
            selected = pool
        else:
            positions = [round(i * (len(pool) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [len(pool) // 2]
            selected = [pool[position] for position in positions]
    return [{"frame_index": frame["frame_index"], "t": frame["t"],
             "source_ref": frame.get("source", {}).get("ref", UNKNOWN)} for frame in selected]


def _candidate(target: dict, signal: dict, needed: list[str], priority: str, reason: str,
               times: list[float], duration: float | None, settings: dict,
               uncertainty: str | None = None) -> dict:
    return {"target": deepcopy(target), "verification_signals": [signal],
            "needed_evidence": sorted(set(needed)), "suggested_video_range": _range(times, duration, settings),
            "suggested_keyframes": [], "priority": priority, "rationale": [reason],
            "uncertainties": [uncertainty] if uncertainty else [], "review_status": "pending"}


def _overlap(left: dict | None, right: dict | None, gap: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left["start_s"] <= right["end_s"] + gap and right["start_s"] <= left["end_s"] + gap


def _merge(candidates: list[dict], settings: dict) -> list[dict]:
    """Merge same-object, overlapping requests before applying the count cap."""
    ordered = sorted(candidates, key=lambda item: (item["target"]["program_object_id"],
                                                    (item["suggested_video_range"] or {}).get("start_s", -1),
                                                    PRIORITY[item["priority"]]))
    merged: list[dict] = []
    for candidate in ordered:
        match = next((item for item in merged if item["target"] == candidate["target"] and
                      _overlap(item["suggested_video_range"], candidate["suggested_video_range"], settings["merge_gap_s"]) and
                      (item["suggested_video_range"] is None or candidate["suggested_video_range"] is None or
                       max(item["suggested_video_range"]["end_s"], candidate["suggested_video_range"]["end_s"]) -
                       min(item["suggested_video_range"]["start_s"], candidate["suggested_video_range"]["start_s"]) <=
                       settings["max_time_range_s"])), None)
        if match is None:
            merged.append(candidate)
            continue
        match["verification_signals"].extend(signal for signal in candidate["verification_signals"]
                                              if signal not in match["verification_signals"])
        match["needed_evidence"] = sorted(set(match["needed_evidence"] + candidate["needed_evidence"]))
        match["rationale"].extend(reason for reason in candidate["rationale"] if reason not in match["rationale"])
        match["priority"] = min((match["priority"], candidate["priority"]), key=PRIORITY.get)
        match["uncertainties"] = sorted(set(match["uncertainties"] + candidate["uncertainties"]))
        a, b = match["suggested_video_range"], candidate["suggested_video_range"]
        if a is not None and b is not None:
            match["suggested_video_range"] = {"start_s": min(a["start_s"], b["start_s"]),
                                              "end_s": max(a["end_s"], b["end_s"])}
    return merged


def request_identifier(request: dict) -> str:
    """Stable identifier for human selection, including legacy request files."""
    identity = {"target": request.get("target"), "range": request.get("suggested_video_range"),
                "signals": sorted(str(item.get("code")) for item in request.get("verification_signals", [])
                                  if isinstance(item, dict))}
    content = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "apr-" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def plan_active_perception_requests(render_validation: dict, evidence: dict, program: dict,
                                    config: dict | None = None, *, source_video: str | None = None) -> dict:
    """Plan requests for human review without running detectors or mutating inputs.

    ``render_validation`` is the content of render_validation.json. Only exact
    Program/Evidence object IDs are linked; no class-based identity guess is made.
    """
    settings = _settings(config)
    result = {"version": "1.0", "mode": "review_only", "status": "ready", "requests": [],
              "execution_issues": [], "diagnostics": [], "truncated": 0}
    if not isinstance(render_validation, dict) or not isinstance(render_validation.get("diagnostics"), list):
        result["execution_issues"].append(_issue("invalid_render_validation", "render validation is missing or invalid", "/render_validation"))
        result["status"] = "execution_blocked"
        return result
    render_diagnostics = [item for item in render_validation["diagnostics"] if isinstance(item, dict)]
    if render_validation.get("decision") == "block" and not render_diagnostics:
        result["execution_issues"].append(_issue("unexplained_render_block", "render validation blocked without diagnostics", "/decision"))
    for item in render_diagnostics:
        code = item.get("code")
        if not isinstance(code, str):
            continue
        if code.startswith("browser_") or code in EXECUTION_CODES or (item.get("severity") == "error" and code not in MOTION_SIGNALS):
            result["execution_issues"].append(_signal(item, "render_validation"))
    if render_validation.get("render_failure"):
        result["execution_issues"].append(_issue("render_failure", "render harness failed", "/render_failure"))
    if result["execution_issues"]:
        result["status"] = "execution_blocked"
        return result

    validation = validate_evidence(evidence)
    if not validation["ok"]:
        result["status"] = "invalid_input"
        result["diagnostics"].append(_issue("invalid_evidence", "Evidence is structurally invalid", "/evidence"))
        return result
    normalized = validation["evidence"]
    try:
        declared_video_hash = declared_video_sha256(normalized)
    except ValueError as exc:
        result["status"] = "invalid_input"
        result["diagnostics"].append(_issue("evidence_video_hash_conflict", str(exc), "/evidence/meta"))
        return result
    result["source_binding"] = {"evidence_sha256": evidence_sha256(normalized),
                                "video_sha256": declared_video_hash,
                                "status": "evidence_declared" if declared_video_hash else "unverified_evidence_origin"}
    if source_video is not None:
        actual_video_hash = file_sha256(source_video)
        if declared_video_hash is not None and declared_video_hash.lower() != actual_video_hash:
            result["status"] = "invalid_input"
            result["diagnostics"].append(_issue("evidence_video_hash_mismatch",
                                                "Evidence source video hash differs from the supplied original video",
                                                "/evidence/meta/source_video_sha256"))
            return result
        result["source_binding"]["checked_video_sha256"] = actual_video_hash
        if declared_video_hash is not None:
            result["source_binding"]["status"] = "verified"
    if not isinstance(program, dict) or not isinstance(program.get("objects"), list) or any(
        not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or not obj["id"]
        for obj in program["objects"]
    ) or len({obj["id"] for obj in program["objects"]}) != len(program["objects"]):
        result["status"] = "invalid_input"
        result["diagnostics"].append(_issue("invalid_program", "Program objects are invalid", "/program/objects"))
        return result
    program_objects = program["objects"]
    evidence_by_id = {obj["id"]: obj for obj in normalized["objects"]}
    frames = normalized.get("frames") or []
    duration_value = normalized.get("meta", {}).get("duration")
    duration = float(duration_value) if _number(duration_value) else None
    render_metrics = render_validation.get("metrics")
    if not isinstance(render_metrics, dict):
        render_metrics = {}
    render_frames = render_metrics.get("frames")
    if not isinstance(render_frames, list):
        render_frames = []
    render_times_by_index = {frame["index"]: frame["t"] for frame in render_frames
                             if isinstance(frame, dict) and isinstance(frame.get("index"), int)
                             and _number(frame.get("t"))}
    render_times = list(render_times_by_index.values())
    object_render_metrics = render_metrics.get("objects")
    if not isinstance(object_render_metrics, dict):
        object_render_metrics = {}
    candidates = []

    for index, obj in enumerate(program_objects):
        object_id = obj["id"]
        ev_obj = evidence_by_id.get(object_id)
        if ev_obj is None:
            result["diagnostics"].append(_issue("unmatched_evidence_object",
                                                f"Program object {object_id} has no exact Evidence ID match; no identity was guessed",
                                                f"/program/objects/{index}"))
        target = {"program_object_id": object_id, "evidence_object_id": ev_obj["id"] if ev_obj else None,
                  "track_id": ev_obj["track_id"] if ev_obj else None,
                  "class_guess": ev_obj.get("class_guess", UNKNOWN) if ev_obj else UNKNOWN}
        observations = (ev_obj.get("observations") or []) if ev_obj else []
        observation_times = [item["t"] for item in observations if _number(item.get("t"))]
        for diagnostic in render_diagnostics:
            code, path = diagnostic.get("code"), diagnostic.get("path")
            if not isinstance(path, str) or not (path == f"/objects/{index}" or path.startswith(f"/objects/{index}/")):
                continue
            signal = _signal(diagnostic, "render_validation")
            if code in VISIBILITY_SIGNALS:
                times = render_times
                if code == "object_temporarily_not_visible":
                    object_metrics = object_render_metrics.get(object_id, {})
                    if not isinstance(object_metrics, dict):
                        object_metrics = {}
                    absent = object_metrics.get("id_absent_frames") or []
                    times = [render_times_by_index[i] for i in absent if isinstance(i, int) and i in render_times_by_index] or render_times
                candidates.append(_candidate(target, signal, ["bbox", "mask", "visible_fraction", "camera_pose"],
                                             "high" if code == "object_never_visible" else "medium",
                                             "ID pass does not establish whether the object was occluded, outside the view, or absent; inspect source video and camera context.",
                                             times, duration, settings, "occlusion_vs_out_of_view_unknown"))
            elif code in MOTION_SIGNALS and (obj.get("motion") or {}).get("type") not in (None, "static"):
                candidates.append(_candidate(target, signal, ["world_position", "orientation", "motion_timing", "camera_pose"],
                                             "high", "Expected Program motion did not appear in sampled rendered states; compare original-video trajectory and implementation timing.",
                                             render_times or observation_times, duration, settings, "source_motion_vs_runtime_unknown"))

        if ev_obj is None:
            continue
        hypotheses = [hypothesis for hypothesis in ev_obj.get("hypotheses", []) if hypothesis.get("kind") == "identity"]
        if hypotheses:
            signal = {"source": "evidence_crosscheck", "stage": "evidence", "severity": "warning",
                      "code": "identity_candidates_unresolved", "path": f"/objects/{index}/hypotheses", "message": "multiple identity candidates remain"}
            candidates.append(_candidate(target, signal, ["track_identity", "appearance", "bbox", "mask"], "high",
                                         "Evidence retains multiple identity candidates; inspect frames around the track and preserve alternatives until resolved.",
                                         observation_times, duration, settings, "identity_unresolved"))
        known_frames = {item["frame_index"] for item in observations if isinstance(item.get("frame_index"), int)}
        coverage = len(known_frames) / len(frames) if frames else 0.0
        if len(known_frames) < settings["min_observation_frames"] or coverage < settings["min_observation_coverage"]:
            signal = {"source": "evidence_crosscheck", "stage": "evidence", "severity": "warning",
                      "code": "insufficient_object_observations", "path": f"/objects/{index}/observations",
                      "message": f"{len(known_frames)} observed frames; coverage {coverage:.3f}"}
            candidates.append(_candidate(target, signal, ["bbox", "mask", "depth", "visible_fraction"], "medium",
                                         "Evidence has too few registered object observations to support a reliable visibility or motion comparison.",
                                         [frame["t"] for frame in frames if frame["frame_index"] not in known_frames] or observation_times,
                                         duration, settings))

    merged = _merge(candidates, settings)
    merged.sort(key=lambda item: (PRIORITY[item["priority"]], item["target"]["program_object_id"],
                                  (item["suggested_video_range"] or {}).get("start_s", -1)))
    result["truncated"] = max(0, len(merged) - settings["max_requests"])
    result["requests"] = merged[:settings["max_requests"]]
    for request in result["requests"]:
        request["request_id"] = request_identifier(request)
        request["suggested_keyframes"] = _keyframes(frames, request["suggested_video_range"], settings["max_keyframes_per_request"])
    return result
