"""Model-independent checks for a render harness index and its saved passes."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..compiler.ids import id_to_color, registry_order


DEFAULTS = {
    "mode": "report", "min_frames": 2, "time_tolerance_s": 0.005,
    "min_rgb_spatial_range": 2, "min_depth_spatial_range": 1,
    "min_id_spatial_range": 1, "min_object_pixels": 4,
    "max_unknown_id_fraction": 0.05, "min_motion_frames": 2,
    "min_motion_span_s": 0.05, "min_translation_m": 0.01,
    "min_rotation_deg": 1.0, "max_browser_events_per_code": 10,
    "sample_frames": 5,
}
PASSES = ("rgb", "depth", "id")


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _config(config: dict | None) -> dict:
    supplied = (config or {}).get("render_validation", config or {})
    if not isinstance(supplied, dict):
        raise ValueError("render_validation config must be a mapping")
    settings = {**DEFAULTS, **supplied}
    if settings["mode"] not in {"report", "enforce"}:
        raise ValueError("render_validation.mode must be report or enforce")
    integer_keys = ("min_frames", "min_rgb_spatial_range", "min_depth_spatial_range", "min_id_spatial_range",
                    "min_object_pixels", "min_motion_frames", "max_browser_events_per_code", "sample_frames")
    if any(isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1 for key in integer_keys):
        raise ValueError("render validation count and pixel thresholds must be positive integers")
    positive = ("time_tolerance_s", "min_motion_span_s", "min_translation_m", "min_rotation_deg")
    if any(not _number(settings[key]) or settings[key] <= 0 for key in positive):
        raise ValueError("render validation temporal and motion thresholds must be finite and positive")
    fraction = settings["max_unknown_id_fraction"]
    if not _number(fraction) or not 0 <= fraction <= 1:
        raise ValueError("max_unknown_id_fraction must be in [0, 1]")
    return settings


def _diag(stage: str, severity: str, code: str, path: str, message: str, hint: str) -> dict:
    return {"stage": stage, "severity": severity, "code": code, "path": path, "message": message, "hint": hint}


def _safe_image_path(render_dir: Path, name: Any) -> Path | None:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        return None
    root = render_dir.resolve()
    path = (root / name).resolve()
    return path if path.is_relative_to(root) else None


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _spatial_range(pixels: np.ndarray) -> int:
    return int(max(np.ptp(pixels[:, :, channel]) for channel in range(3)))


def _packed(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.uint32)
    return (values[:, :, 0] << 16) | (values[:, :, 1] << 8) | values[:, :, 2]


def _color(index: int) -> int:
    red, green, blue = id_to_color(index)
    return (red << 16) | (green << 8) | blue


def _state_name(entry: dict) -> str:
    return f"{entry['id']}#{entry['instance']}" if entry["instance"] else entry["id"]


def _vector(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == length and all(_number(item) for item in value)


def _rotation_degrees(a: list[float], b: list[float]) -> float:
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    first_norm, second_norm = np.linalg.norm(first), np.linalg.norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("nan")
    cosine = abs(float(np.dot(first, second) / (first_norm * second_norm)))
    return math.degrees(2 * math.acos(min(1.0, max(-1.0, cosine))))


def _browser_events(index: dict, render_dir: Path, diagnostics: list[dict], settings: dict) -> dict:
    events = index.get("browser_events")
    if events is None:
        log_path = render_dir / "browser.log"
        events = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else []
    if not isinstance(events, list):
        diagnostics.append(_diag("browser", "error", "invalid_browser_events", "/browser_events",
                                 "browser_events is not a list", "record browser console, pageerror and requestfailed events as lines"))
        return {"error": 0, "pageerror": 0, "requestfailed": 0}
    counts = {"error": 0, "pageerror": 0, "requestfailed": 0}
    for index_value, line in enumerate(events):
        if not isinstance(line, str):
            continue
        for kind in counts:
            if line.startswith(f"[{kind}]"):
                counts[kind] += 1
                if counts[kind] <= settings["max_browser_events_per_code"]:
                    diagnostics.append(_diag("browser", "error", f"browser_{kind}", f"/browser_events/{index_value}",
                                             line[len(kind) + 2:].strip(), "inspect browser.log and the failing resource or script"))
                break
    return counts


def validate_render_result(program: dict, render_index: dict | None, render_dir: str | Path,
                           config: dict | None = None, *, expected_times: list[float] | None = None) -> dict:
    """Return per-pass, per-object and browser diagnostics; never alter Program.

    An ID-pass absence proves only that an object was not sampled as visible.
    It cannot by itself separate occlusion from leaving the camera frustum.
    """
    settings = _config(config)
    root = Path(render_dir)
    diagnostics: list[dict] = []
    metrics: dict = {"frames": [], "objects": {}, "browser": {}, "expected_frame_count": len(expected_times) if expected_times is not None else None}
    if not isinstance(render_index, dict):
        diagnostics.append(_diag("render", "error", "missing_render_index", "/render_index",
                                 "render index is missing or invalid", "inspect harness failure and index.json"))
        metrics["browser"] = _browser_events({}, root, diagnostics, settings)
        return _report(settings, diagnostics, metrics)

    frames = render_index.get("frames")
    if not isinstance(frames, list):
        diagnostics.append(_diag("render", "error", "invalid_frames", "/frames",
                                 "render index has no frames array", "write one indexed entry per requested time"))
        frames = []
    width, height = render_index.get("width"), render_index.get("height")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (width, height)):
        diagnostics.append(_diag("render", "error", "invalid_frame_size", "/width",
                                 "render index width and height must be positive integers", "record the actual output dimensions"))
        width = height = None
    if len(frames) < settings["min_frames"]:
        diagnostics.append(_diag("render", "error", "too_few_render_frames", "/frames",
                                 f"only {len(frames)} frames were indexed", "render enough distinct times to inspect visibility and motion"))
    if expected_times is not None and len(frames) != len(expected_times):
        diagnostics.append(_diag("render", "error", "frame_count_mismatch", "/frames",
                                 f"expected {len(expected_times)} frames, found {len(frames)}", "re-render all requested timestamps"))
    declared_passes = render_index.get("passes")
    if not isinstance(declared_passes, list) or any(name not in declared_passes for name in PASSES):
        diagnostics.append(_diag("render", "error", "missing_render_pass", "/passes",
                                 "RGB, depth and ID passes were not all declared", "render all three required passes"))

    registry = registry_order(program)
    expected_names = Counter(_state_name(entry) for entry in registry)
    expected_colors = {_color(entry["idx"]) for entry in registry}
    object_entries: dict[str, list[dict]] = {}
    for entry in registry:
        if entry["kind"] == "object":
            object_entries.setdefault(entry["id"], []).append(entry)
    presence = {object_id: [] for object_id in object_entries}
    state_poses: dict[str, list[tuple[float, list[float], list[float]]]] = {name: [] for name in expected_names}
    indexed_times: list[float] = []

    for frame_index, frame in enumerate(frames):
        base = f"/frames/{frame_index}"
        if not isinstance(frame, dict):
            diagnostics.append(_diag("render", "error", "invalid_frame", base,
                                     "frame entry is not an object", "record frame time, files and state"))
            continue
        time_value = frame.get("t")
        if not _number(time_value):
            diagnostics.append(_diag("render", "error", "invalid_frame_time", f"{base}/t",
                                     "frame timestamp is not finite", "record the requested replay time"))
        else:
            indexed_times.append(float(time_value))
            if expected_times is not None and frame_index < len(expected_times):
                expected = expected_times[frame_index]
                if not _number(expected) or abs(float(time_value) - float(expected)) > settings["time_tolerance_s"]:
                    diagnostics.append(_diag("render", "error", "timestamp_mismatch", f"{base}/t",
                                             f"frame timestamp {time_value} differs from requested {expected}", "render and index the same requested time"))
        frame_metric = {"index": frame_index, "t": time_value, "passes": {}}
        metrics["frames"].append(frame_metric)
        arrays: dict[str, np.ndarray] = {}
        files = frame.get("files") if isinstance(frame.get("files"), dict) else {}
        for pass_name in PASSES:
            file_path = _safe_image_path(root, files.get(pass_name))
            file_ref = f"{base}/files/{pass_name}"
            if file_path is None:
                diagnostics.append(_diag("render", "error", "missing_pass_file", file_ref,
                                         f"{pass_name} filename is absent or escapes the render directory", "record a relative pass filename"))
                frame_metric["passes"][pass_name] = "missing"
                continue
            if not file_path.is_file():
                diagnostics.append(_diag("render", "error", "missing_pass_file", file_ref,
                                         f"{pass_name} image does not exist: {file_path.name}", "re-render this pass"))
                frame_metric["passes"][pass_name] = "missing"
                continue
            try:
                pixels = _image(file_path)
            except (OSError, ValueError, UnidentifiedImageError):
                diagnostics.append(_diag("render", "error", "undecodable_pass", file_ref,
                                         f"{pass_name} image cannot be decoded", "replace the corrupt image by re-rendering"))
                frame_metric["passes"][pass_name] = "undecodable"
                continue
            arrays[pass_name] = pixels
            observed_size = [int(pixels.shape[1]), int(pixels.shape[0])]
            frame_metric["passes"][pass_name] = {"size": observed_size, "spatial_range": _spatial_range(pixels)}
            if width is not None and observed_size != [width, height]:
                diagnostics.append(_diag("render", "error", "pass_size_mismatch", file_ref,
                                         f"{pass_name} image is {observed_size}, index declares {[width, height]}", "render all passes at the indexed size"))
            minimum_range = settings[f"min_{pass_name}_spatial_range"]
            if _spatial_range(pixels) < minimum_range:
                diagnostics.append(_diag("render", "error", "blank_pass", file_ref,
                                         f"{pass_name} image has no meaningful spatial variation", "inspect camera, scene visibility and pass encoder"))
        sizes = {tuple(pixels.shape[:2]) for pixels in arrays.values()}
        if len(sizes) > 1:
            diagnostics.append(_diag("render", "error", "pass_size_mismatch", f"{base}/files",
                                     "RGB, depth and ID images have different dimensions", "render all passes with one viewport"))

        if "id" in arrays:
            packed = _packed(arrays["id"])
            colored = packed != 0
            unknown_count = int(np.count_nonzero(colored & ~np.isin(packed, list(expected_colors))))
            colored_count = int(np.count_nonzero(colored))
            unknown_fraction = unknown_count / colored_count if colored_count else 0.0
            frame_metric["unknown_id_fraction"] = round(unknown_fraction, 6)
            if unknown_fraction > settings["max_unknown_id_fraction"]:
                diagnostics.append(_diag("render", "error", "unknown_render_id", f"{base}/files/id",
                                         f"{unknown_count} non-background ID pixels do not belong to Program registry",
                                         "check Program registry order and ID pass colour encoding"))
            for object_id, entries in object_entries.items():
                count = sum(int(np.count_nonzero(packed == _color(entry["idx"]))) for entry in entries)
                presence[object_id].append((frame_index, count >= settings["min_object_pixels"], count))

        state = frame.get("state")
        if not isinstance(state, dict) or not isinstance(state.get("objects"), list):
            diagnostics.append(_diag("render", "error", "missing_frame_state", f"{base}/state",
                                     "frame lacks a usable object state", "save window.__game.state() for every frame"))
            continue
        if _number(time_value) and (not _number(state.get("t")) or abs(float(state["t"]) - float(time_value)) > settings["time_tolerance_s"]):
            diagnostics.append(_diag("render", "error", "state_time_mismatch", f"{base}/state/t",
                                     "state time does not match indexed frame time", "capture state immediately after seeking the replay time"))
        observed_names = Counter(item.get("name") for item in state["objects"]
                                 if isinstance(item, dict) and isinstance(item.get("name"), str))
        if observed_names != expected_names:
            diagnostics.append(_diag("render", "error", "state_id_mismatch", f"{base}/state/objects",
                                     "rendered object names/counts differ from Program registry", "recompile and render the same Program"))
        for state_index, item in enumerate(state["objects"]):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name not in expected_names:
                continue
            if _number(time_value) and _vector(item.get("pos"), 3) and _vector(item.get("quat"), 4):
                state_poses[name].append((float(time_value), item["pos"], item["quat"]))
            else:
                diagnostics.append(_diag("render", "error", "invalid_object_pose", f"{base}/state/objects/{state_index}",
                                         f"object {name} has no finite position and quaternion", "save finite world-space transforms"))
        state_ids = Counter((item.get("id"), item.get("kind")) for item in state["objects"]
                            if isinstance(item, dict) and isinstance(item.get("id"), str)
                            and isinstance(item.get("kind"), str))
        expected_ids = Counter((entry["id"], entry["kind"]) for entry in registry)
        if state_ids != expected_ids:
            diagnostics.append(_diag("render", "error", "state_id_mismatch", f"{base}/state/objects",
                                     "rendered object IDs or kinds differ from Program", "check the compiled registry and Program IDs"))

    if len(indexed_times) > 1 and any(right <= left for left, right in zip(indexed_times, indexed_times[1:])):
        diagnostics.append(_diag("render", "error", "non_monotonic_render_times", "/frames",
                                 "render timestamps are duplicated or out of order", "request distinct increasing replay times"))
    for object_index, obj in enumerate(program.get("objects") or []):
        object_id = obj["id"]
        observed = presence.get(object_id, [])
        visible = [index for index, shown, _ in observed if shown]
        absent = [index for index, shown, _ in observed if not shown]
        metric = {"id_visible_frames": visible, "id_absent_frames": absent, "id_checked_frames": len(observed),
                  "state_samples": {}, "dynamic": bool(obj.get("motion") and obj["motion"].get("type") != "static")}
        metrics["objects"][object_id] = metric
        if not observed:
            diagnostics.append(_diag("visibility", "warning", "object_visibility_unverifiable", f"/objects/{object_index}",
                                     f"no valid ID pass can establish visibility of {object_id}", "restore the ID pass before drawing visibility conclusions"))
        elif not visible:
            code = "object_never_visible" if len(observed) == len(frames) else "object_not_observed_in_available_id_frames"
            diagnostics.append(_diag("visibility", "warning", code, f"/objects/{object_index}",
                                     f"{object_id} never appears in the available ID frames", "check camera framing, occlusion and object visibility"))
        elif absent:
            diagnostics.append(_diag("visibility", "warning", "object_temporarily_not_visible", f"/objects/{object_index}",
                                     f"{object_id} appears in some ID frames but not {absent}", "inspect occlusion, camera frustum and timing at the absent frames"))

        if not metric["dynamic"]:
            continue
        for entry in object_entries.get(object_id, []):
            name = _state_name(entry)
            poses = state_poses.get(name, [])
            metric["state_samples"][name] = len(poses)
            if len(poses) < settings["min_motion_frames"] or poses[-1][0] - poses[0][0] < settings["min_motion_span_s"]:
                diagnostics.append(_diag("motion", "warning", "insufficient_motion_samples", f"/objects/{object_index}/motion",
                                         f"{name} has insufficient sampled state for motion verification", "render more distinct times across the motion interval"))
                continue
            translations = [math.dist(poses[0][1], sample[1]) for sample in poses]
            rotations = [_rotation_degrees(poses[0][2], sample[2]) for sample in poses]
            if not all(math.isfinite(value) for value in rotations):
                diagnostics.append(_diag("motion", "error", "invalid_object_rotation", f"/objects/{object_index}/motion",
                                         f"{name} has a zero or invalid state quaternion", "inspect runtime pose generation"))
                continue
            max_translation, max_rotation = max(translations), max(rotations)
            metric.setdefault("motion", {})[name] = {"max_translation_m": round(max_translation, 6),
                                                      "max_rotation_deg": round(max_rotation, 6)}
            if max_translation < settings["min_translation_m"] and max_rotation < settings["min_rotation_deg"]:
                triggered = bool((obj.get("motion") or {}).get("trigger"))
                diagnostics.append(_diag("motion", "warning" if triggered else "error",
                                         "trigger_motion_not_exercised" if triggered else "dynamic_object_not_moving",
                                         f"/objects/{object_index}/motion",
                                         f"{name} did not move or rotate across sampled replay times",
                                         "check motion parameters and sampled times; triggered motion needs a play-mode event"))

    metrics["browser"] = _browser_events(render_index, root, diagnostics, settings)
    return _report(settings, diagnostics, metrics)


def _report(settings: dict, diagnostics: list[dict], metrics: dict) -> dict:
    errors = [item for item in diagnostics if item["severity"] == "error"]
    warnings = [item for item in diagnostics if item["severity"] == "warning"]
    return {"version": "1.0", "mode": settings["mode"], "ok": not errors,
            "decision": "block" if errors else ("warn" if warnings else "proceed"),
            "diagnostics": diagnostics, "errors": errors, "warnings": warnings, "metrics": metrics}
