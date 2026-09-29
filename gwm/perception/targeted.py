"""Targeted, review-gated observation from original video; never updates Evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from ..feedback.active_perception import request_identifier
from .contract import validate_evidence
from .base import DetectionProvider, DetectionRecord
from .frames import extract_frames, pick_uniform, probe
from .provenance import declared_video_sha256, evidence_sha256, file_sha256


DEFAULTS = {"sample_fps": 4.0, "max_frames_per_request": 5, "max_side": 960,
            "ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"}


def _settings(config: dict | None) -> dict:
    supplied = (config or {}).get("targeted_observation", config or {})
    if not isinstance(supplied, dict):
        raise ValueError("targeted_observation config must be a mapping")
    settings = {**DEFAULTS, **supplied}
    for key in ("max_frames_per_request", "max_side"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    fps = settings["sample_fps"]
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    for key in ("ffmpeg_bin", "ffprobe_bin"):
        if not isinstance(settings[key], str) or not settings[key]:
            raise ValueError(f"{key} must be a nonempty executable path")
    return settings


def _range(request: dict, duration: float) -> tuple[float, float]:
    interval = request.get("suggested_video_range")
    if not isinstance(interval, dict):
        raise ValueError("request lacks a suggested_video_range")
    start, end = interval.get("start_s"), interval.get("end_s")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (start, end)) or start < 0 or end <= start:
        raise ValueError("request time range must have finite 0 <= start_s < end_s")
    if start >= duration:
        raise ValueError("request time range starts outside the source video")
    return float(start), min(float(end), duration)


def _phrases(request: dict) -> list[str]:
    supplied = request.get("search_phrases")
    if supplied is None:
        supplied = [(request.get("target") or {}).get("class_guess")]
    if not isinstance(supplied, list):
        raise ValueError("search_phrases must be an approved list")
    phrases = sorted({" ".join(value.casefold().strip().rstrip(".").split()) for value in supplied
                      if isinstance(value, str) and value.strip() and value != "unknown"})
    if not phrases:
        raise ValueError("no target search phrase; approve search_phrases before detection")
    return phrases


def _choose(frames: list[dict], request: dict, limit: int, start: float) -> list[dict]:
    """Prefer reviewed keyframe times, then fill remaining slots uniformly."""
    selected: dict[int, dict] = {}
    times = [item.get("t") for item in request.get("suggested_keyframes", []) if isinstance(item, dict)]
    for time in times:
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time):
            continue
        nearest = min(frames, key=lambda frame: (abs(frame.get("source_pts_s", start + frame["t"]) - time), frame["index"]))
        selected[nearest["index"]] = nearest
        if len(selected) >= limit:
            break
    for frame in pick_uniform(frames, limit):
        if len(selected) >= limit:
            break
        selected.setdefault(frame["index"], frame)
    return [selected[index] for index in sorted(selected)]


def _bbox(record: DetectionRecord, width: int, height: int) -> dict:
    values = tuple(float(value) for value in record.bbox)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("provider returned invalid bbox")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("provider bbox lies outside decoded frame")
    return {"format": "xyxy", "values": list(values), "space": "pixel", "image_size": [width, height]}


def _observation(record: DetectionRecord, frame: dict, target_id: str, provider_name: str,
                 request_dir: Path) -> dict:
    if not isinstance(record, DetectionRecord) or not isinstance(record.detection_id, str) or not record.detection_id:
        raise ValueError("provider must return DetectionRecord items with detection IDs")
    score = record.score
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("provider returned invalid confidence")
    box = _bbox(record, frame["width"], frame["height"])
    position = None
    if record.position_3d is not None:
        if len(record.position_3d) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                                               not math.isfinite(value) for value in record.position_3d):
            raise ValueError("provider returned invalid 3D position")
        position = [float(value) for value in record.position_3d]
    mask_ref = None
    if record.mask is not None:
        mask = np.asarray(record.mask)
        if mask.shape != (frame["height"], frame["width"]):
            raise ValueError("provider mask size differs from decoded frame")
        mask_dir = request_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        safe_id = hashlib.sha256(record.detection_id.encode("utf-8")).hexdigest()[:12]
        mask_file = mask_dir / f"f{frame['extracted_index']:05d}_{safe_id}.png"
        Image.fromarray((mask.astype(bool) * 255).astype(np.uint8)).save(mask_file)
        mask_ref = str(mask_file.resolve())
    return {"target_object_id": target_id, "identity_status": "candidate_unverified",
            "video_time_s": frame["video_time_s"], "frame_ref": frame["file"],
            "detection": {"detection_id": record.detection_id, "class_label": record.class_label,
                          "confidence": float(score), "bbox": box,
                          "mask_ref": mask_ref, "position_3d": position,
                          "position_frame": "provider_unspecified" if position is not None else None,
                          "source": record.source or provider_name}}


def _write(out_dir: Path, report: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "targeted_observations.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _source_issues(submitted: dict, selected: list[tuple[str, dict]], video_hash: str,
                   evidence_file: str | Path | None, require_evidence: bool) -> tuple[list[dict], dict | None, dict[str, list[dict]]]:
    """Reject mismatched provenance before any Provider call or frame extraction."""
    issues: list[dict] = []
    request_issues: dict[str, list[dict]] = {}
    binding = submitted.get("source_binding")
    binding = binding if isinstance(binding, dict) else {}
    declared = binding.get("video_sha256")
    if declared is not None and declared != video_hash:
        issues.append({"code": "request_video_hash_mismatch", "reason": "request video hash differs from supplied video"})
    if evidence_file is None:
        if require_evidence:
            issues.append({"code": "missing_evidence", "reason": "a validated Evidence v2 file is required before Provider inference"})
        return issues, None, request_issues
    try:
        evidence = json.loads(Path(evidence_file).read_text(encoding="utf-8"))
        validation = validate_evidence(evidence, adapt_v1=False)
    except (OSError, ValueError, TypeError) as exc:
        issues.append({"code": "evidence_unavailable", "reason": str(exc)[:500]})
        return issues, None, request_issues
    if not validation["ok"]:
        issues.append({"code": "invalid_evidence", "reason": "Evidence v2 validation failed"})
        return issues, None, request_issues
    normalized = validation["evidence"]
    if not declared:
        issues.append({"code": "request_source_unbound", "reason": "request has no source video hash"})
    try:
        evidence_origin = declared_video_sha256(normalized)
    except ValueError as exc:
        issues.append({"code": "evidence_origin_conflict", "reason": str(exc)})
        evidence_origin = None
    if not evidence_origin:
        issues.append({"code": "evidence_origin_unverified", "reason": "Evidence has no source video hash"})
    elif evidence_origin != video_hash:
        issues.append({"code": "evidence_video_hash_mismatch", "reason": "Evidence source hash differs from supplied video"})
    expected_evidence_hash = binding.get("evidence_sha256")
    if not expected_evidence_hash:
        issues.append({"code": "request_evidence_unbound", "reason": "request has no Evidence fingerprint"})
    elif expected_evidence_hash != evidence_sha256(normalized):
        issues.append({"code": "request_evidence_hash_mismatch", "reason": "Evidence content differs from the request source"})
    objects = {item["id"]: item for item in normalized.get("objects", [])}
    for request_id, request in selected:
        target = request.get("target") or {}
        evidence_id = target.get("evidence_object_id") if isinstance(target, dict) else None
        track_id = target.get("track_id") if isinstance(target, dict) else None
        program_id = target.get("program_object_id") if isinstance(target, dict) else None
        if not evidence_id or evidence_id not in objects:
            request_issues.setdefault(request_id, []).append(
                {"code": "request_object_mismatch", "reason": f"{request_id}: target Evidence object ID is absent"})
        elif track_id != objects[evidence_id]["track_id"]:
            request_issues.setdefault(request_id, []).append(
                {"code": "request_track_mismatch", "reason": f"{request_id}: target track ID differs from Evidence"})
        if program_id != evidence_id:
            request_issues.setdefault(request_id, []).append(
                {"code": "request_program_object_mismatch", "reason": f"{request_id}: Program and Evidence object IDs differ; no mapping was verified"})
    return issues, normalized, request_issues


def run_targeted_observations(requests_file: str | Path, video_file: str | Path, out_dir: str | Path,
                              provider: DetectionProvider | None = None, config: dict | None = None,
                              *, selected_request_ids: Iterable[str] | None = None,
                              provider_unavailable_reason: str | None = None,
                              evidence_file: str | Path | None = None) -> dict:
    """Read human-approved requests, extract frames, and collect candidate detections.

    No detector is invoked when ``provider`` is None. Output observations are
    *candidates*, not Evidence v2, because identity/depth may remain unknown.
    """
    settings = _settings(config)
    source = Path(requests_file)
    video = Path(video_file)
    target_dir = Path(out_dir)
    submitted = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(submitted, dict) or not isinstance(submitted.get("requests"), list):
        raise ValueError("active perception request file must contain a requests array")
    selection = set(selected_request_ids) if selected_request_ids is not None else None
    selected: list[tuple[str, dict]] = []
    duplicates = []
    seen = set()
    for request in submitted["requests"]:
        if not isinstance(request, dict):
            continue
        request_id = request.get("request_id") or request_identifier(request)
        if not isinstance(request_id, str):
            continue
        if selection is not None and request_id not in selection:
            continue
        if selection is None and request.get("review_status") != "approved":
            continue
        if request_id in seen:
            duplicates.append(request_id)
            continue
        seen.add(request_id)
        selected.append((request_id, request))
    report = {"version": "1.0", "status": "nothing_selected", "source_request_file": str(source.resolve()),
              "source_video": str(video.resolve()), "selected_request_ids": [item[0] for item in selected],
              "duplicate_request_ids": duplicates, "unknown_selected_request_ids": sorted(selection - seen) if selection else [],
              "source_issues": [], "results": [], "provenance": {"frame_extractor": "gwm.perception.frames.extract_frames",
              "provider": getattr(provider, "name", None), "sample_fps": settings["sample_fps"],
              "request_file_sha256": file_sha256(source),
              "provider_model_dir": str(Path(provider.model_dir).resolve()) if provider is not None and hasattr(provider, "model_dir") else None,
              "video_time_basis": "per-frame time_basis; decoded_source_pts only if extractor supplies source_pts_s",
              "ffmpeg_bin": settings["ffmpeg_bin"], "ffprobe_bin": settings["ffprobe_bin"]}}
    if not selected:
        return _write(target_dir, report)
    if submitted.get("status") == "execution_blocked" or submitted.get("execution_issues"):
        report["status"] = "source_execution_blocked"
        for request_id, request in selected:
            report["results"].append({"request_id": request_id, "target": request.get("target"), "status": "skipped",
                                      "reason_code": "source_execution_issue", "reason": "resolve verification execution issues first",
                                      "selected_frames": [], "observations": [], "frame_failures": []})
        return _write(target_dir, report)
    try:
        if not video.is_file():
            raise FileNotFoundError(f"source video not found: {video}")
        video_info = probe(video, settings["ffprobe_bin"])
        report["provenance"]["video_sha256"] = file_sha256(video)
        report["provenance"]["video_probe"] = video_info
    except (OSError, ValueError, KeyError, IndexError) as exc:
        report["status"] = "video_unavailable"
        for request_id, request in selected:
            report["results"].append({"request_id": request_id, "target": request.get("target"), "status": "failed",
                                      "reason_code": "video_unavailable", "reason": str(exc)[:500],
                                      "selected_frames": [], "observations": [], "frame_failures": []})
        return _write(target_dir, report)

    issues, checked_evidence, request_issues = _source_issues(submitted, selected, report["provenance"]["video_sha256"],
                                                               evidence_file, provider is not None)
    report["source_issues"] = issues + [issue for findings in request_issues.values() for issue in findings]
    if checked_evidence is not None:
        report["provenance"]["evidence_file"] = str(Path(evidence_file).resolve())
        report["provenance"]["evidence_sha256"] = evidence_sha256(checked_evidence)
    if issues:
        report["status"] = "source_mismatch"
        for request_id, request in selected:
            report["results"].append({"request_id": request_id, "target": request.get("target"), "status": "rejected",
                                      "reason_code": issues[0]["code"], "reason": "; ".join(item["reason"] for item in issues),
                                      "selected_frames": [], "observations": [], "frame_failures": []})
        return _write(target_dir, report)

    for ordinal, (request_id, request) in enumerate(selected):
        if request_id in request_issues:
            findings = request_issues[request_id]
            report["results"].append({"request_id": request_id, "target": request.get("target"), "status": "rejected",
                                      "reason_code": findings[0]["code"], "reason": "; ".join(item["reason"] for item in findings),
                                      "selected_frames": [], "observations": [], "frame_failures": []})
            continue
        result = {"request_id": request_id, "target": request.get("target"), "status": "failed", "reason_code": None,
                  "reason": None, "requested_range": request.get("suggested_video_range"), "actual_range": None,
                  "selected_frames": [], "observations": [], "frame_failures": [], "search_phrases": []}
        report["results"].append(result)
        try:
            target = request.get("target")
            if not isinstance(target, dict) or not isinstance(target.get("program_object_id"), str) or not target["program_object_id"]:
                raise ValueError("request lacks a target Program object ID")
            start, end = _range(request, video_info["duration"])
            result["actual_range"] = {"start_s": start, "end_s": end}
            request_dir = target_dir / f"request_{ordinal:03d}_{hashlib.sha256(request_id.encode('utf-8')).hexdigest()[:10]}"
            extracted = extract_frames(video, request_dir / "frames", settings["sample_fps"], end - start,
                                       settings["max_side"], start, settings["ffmpeg_bin"], settings["ffprobe_bin"])
            if not extracted:
                raise RuntimeError("frame extractor returned no frames in the requested range")
            selected_frames = _choose(extracted, request, settings["max_frames_per_request"], start)
            for frame in selected_frames:
                pts = frame.get("source_pts_s")
                if pts is not None and (isinstance(pts, bool) or not isinstance(pts, (int, float)) or not math.isfinite(pts)):
                    raise ValueError("extractor returned invalid source PTS")
                selected_frame = {"extracted_index": frame["index"],
                                  "video_time_s": float(pts) if pts is not None else round(start + frame["t"], 4),
                                  "time_basis": "decoded_source_pts" if pts is not None else "estimated_sample_time",
                                  "file": str(Path(frame["file"]).resolve()),
                                  "width": frame["width"], "height": frame["height"]}
                if "source_frame_index" in frame:
                    selected_frame["source_frame_index"] = frame["source_frame_index"]
                result["selected_frames"].append(selected_frame)
            if provider is None:
                result.update(status="provider_unavailable", reason_code="provider_unavailable",
                              reason=provider_unavailable_reason or "frames extracted; no Perception Provider was supplied")
                continue
            phrases = _phrases(request)
            result["search_phrases"] = phrases
            observed = set()
            for frame in result["selected_frames"]:
                try:
                    with Image.open(frame["file"]) as source_image:
                        image = source_image.convert("RGB")
                    records = provider.detect(image, frame["extracted_index"], phrases, config or {})
                    if not isinstance(records, list):
                        raise ValueError("provider returned a non-list detection result")
                    for record in records:
                        if not isinstance(record, DetectionRecord):
                            raise ValueError("provider returned an item that is not DetectionRecord")
                        if not isinstance(record.class_label, str):
                            raise ValueError("provider returned a non-string class label")
                        label = " ".join(record.class_label.casefold().strip().rstrip(".").split())
                        if label not in phrases:
                            continue
                        identity = (frame["extracted_index"], record.detection_id)
                        if identity in observed:
                            continue
                        observation = _observation(record, frame, target["program_object_id"],
                                                   getattr(provider, "name", "unnamed_provider"), request_dir)
                        observed.add(identity)
                        result["observations"].append(observation)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    result["frame_failures"].append({"extracted_index": frame["extracted_index"],
                                                     "reason": str(exc)[:500]})
            if result["frame_failures"]:
                result.update(status="partial" if result["observations"] else "failed",
                              reason_code="provider_frame_failure", reason="one or more selected frames failed")
            elif result["observations"]:
                result.update(status="observed", reason_code=None, reason=None)
            else:
                result.update(status="no_match", reason_code="no_target_detection",
                              reason="provider returned no detections matching approved search phrases")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            result.update(status="failed", reason_code="request_failed", reason=str(exc)[:500])
    states = {item["status"] for item in report["results"]}
    report["status"] = ("source_mismatch" if states == {"rejected"} else
                        "provider_unavailable" if states == {"provider_unavailable"} else
                        "completed" if states <= {"observed", "no_match"} else
                        "partial" if states & {"observed", "no_match", "partial", "provider_unavailable"} else "failed")
    return _write(target_dir, report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract requested video frames; detections require a supplied provider")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--evidence", help="validated Evidence v2 JSON bound to this video; required for detection")
    parser.add_argument("--out", required=True)
    parser.add_argument("--select", action="append", default=None, help="manually selected request ID; repeat for multiple")
    parser.add_argument("--provider", choices=("none", "grounding-dino"), default="none")
    parser.add_argument("--model-dir", help="existing local Grounding DINO model directory; never downloaded")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    args = parser.parse_args(argv)
    provider = None
    unavailable_reason = None
    if args.provider == "grounding-dino":
        model_dir = Path(args.model_dir) if args.model_dir else None
        if model_dir is None or not model_dir.is_dir():
            unavailable_reason = "local Grounding DINO model directory is missing; no model was called"
        elif not (model_dir / "config.json").is_file() or not any(
            (model_dir / filename).is_file() for filename in
            ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
        ):
            unavailable_reason = "local Grounding DINO config or weights are missing; no model was called"
        else:
            try:
                from .grounded_sam2_backend import GroundingDinoDetectionProvider
                provider = GroundingDinoDetectionProvider(model_dir)
            except ImportError as exc:
                unavailable_reason = f"local Grounding DINO dependency unavailable: {exc}"
    result = run_targeted_observations(args.requests, args.video, args.out, provider,
                                       selected_request_ids=args.select,
                                       config={"ffmpeg_bin": args.ffmpeg_bin, "ffprobe_bin": args.ffprobe_bin},
                                       provider_unavailable_reason=unavailable_reason, evidence_file=args.evidence)
    print(json.dumps({"status": result["status"], "results": len(result["results"]),
                      "artifact": str((Path(args.out) / "targeted_observations.json").resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
