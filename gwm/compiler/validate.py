"""Schema + semantic validation of a scene program. Returns structured errors that can be fed back to the VLM."""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
import jsonschema

SCHEMA_PATH = Path(__file__).parent / "schema" / "program.schema.json"
_SCHEMA = None

def schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(SCHEMA_PATH.read_text())
    return _SCHEMA

def _err(path: str, code: str, message: str, suggestion: str = "") -> dict:
    return {"path": path, "code": code, "message": message, "suggestion": suggestion}

def schema_errors(program: Any) -> list[dict]:
    v = jsonschema.Draft202012Validator(schema())
    out = []
    for e in sorted(v.iter_errors(program), key=lambda e: list(e.absolute_path)):
        path = "/" + "/".join(str(p) for p in e.absolute_path)
        out.append(_err(path, "schema", e.message[:300], "fix the field to match the schema"))
    return out

def _finite(v) -> bool:
    try:
        return all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
    except TypeError:
        return False

def semantic_errors(program: dict) -> tuple[list[dict], list[dict]]:
    errors: list[dict] = []
    warnings: list[dict] = []
    ids: dict[str, str] = {}
    for i, s in enumerate(program.get("static", [])):
        p = f"/static/{i}"
        if s["id"] in ids: errors.append(_err(p + "/id", "duplicate_id", f"id '{s['id']}' already used at {ids[s['id']]}", "rename to a unique id"))
        ids[s["id"]] = p
        _check_geom(s.get("geom", {}), p + "/geom", errors)
        if not _finite(s.get("pose", {}).get("pos", [])): errors.append(_err(p + "/pose/pos", "non_finite", "position must be finite numbers"))
    for i, o in enumerate(program.get("objects", [])):
        p = f"/objects/{i}"
        if o["id"] in ids: errors.append(_err(p + "/id", "duplicate_id", f"id '{o['id']}' already used at {ids[o['id']]}", "rename to a unique id"))
        ids[o["id"]] = p
        _check_geom(o.get("geom", {}), p + "/geom", errors)
        if "pose" not in o and not o.get("instances"): errors.append(_err(p, "missing_pose", "object needs 'pose' or 'instances'", "add pose.pos"))
        if o.get("pose") and not _finite(o["pose"].get("pos", [])): errors.append(_err(p + "/pose/pos", "non_finite", "position must be finite numbers"))
        m = o.get("motion")
        if m: _check_motion(m, p + "/motion", program.get("meta", {}).get("duration"), errors, warnings)
        for k, ev in enumerate(o.get("events", [])):
            if ev["type"] == "trigger_on_enter" and not ev.get("target"): errors.append(_err(f"{p}/events/{k}", "missing_target", "trigger_on_enter needs 'target'"))
    # references
    for i, o in enumerate(program.get("objects", [])):
        p = f"/objects/{i}"
        sup = o.get("support")
        if sup and sup not in ids: errors.append(_err(p + "/support", "unknown_ref", f"support '{sup}' is not a known id", "use an existing static or object id"))
        for k, ev in enumerate(o.get("events", [])):
            tgt = ev.get("target")
            if tgt and tgt not in ids: errors.append(_err(f"{p}/events/{k}/target", "unknown_ref", f"target '{tgt}' is not a known id"))
            w = ev.get("with")
            if w and w != "player" and w not in ids: errors.append(_err(f"{p}/events/{k}/with", "unknown_ref", f"with '{w}' is not 'player' or a known id"))
    # support cycles
    sup = {o["id"]: o.get("support") for o in program.get("objects", []) if o.get("support")}
    for start in sup:
        seen, cur = set(), start
        while cur in sup and cur not in seen:
            seen.add(cur); cur = sup[cur]
            if cur == start: errors.append(_err(f"/objects", "support_cycle", f"support relation cycles through '{start}'", "remove one support link")); break
    # binding refs
    slots = (program.get("binding") or {}).get("slots") or {}
    for key in ("hazards", "collectibles"):
        for k, rid in enumerate(slots.get(key) or []):
            if rid not in ids: errors.append(_err(f"/binding/slots/{key}/{k}", "unknown_ref", f"'{rid}' is not a known id"))
    # camera
    kf = (program.get("camera") or {}).get("keyframes") or []
    ts = [k["t"] for k in kf]
    if ts != sorted(ts): errors.append(_err("/camera/keyframes", "not_monotonic", "camera keyframe times must be non-decreasing"))
    if not kf: warnings.append(_err("/camera", "no_camera", "no camera keyframes; replay renders use a default camera"))
    if not program.get("static"): warnings.append(_err("/static", "no_static", "no static structure; the player has nothing to stand on unless binding adds ground"))
    return errors, warnings

def _check_geom(g: dict, path: str, errors: list[dict]) -> None:
    kind = g.get("kind")
    if kind == "primitive":
        shape = g.get("shape")
        if shape in ("box",) and not g.get("extent"): errors.append(_err(path, "missing_extent", "box needs 'extent'"))
        if shape == "plane" and not g.get("size"): errors.append(_err(path, "missing_size", "plane needs 'size' [w, d]"))
        if shape in ("sphere",) and not g.get("radius"): errors.append(_err(path, "missing_radius", "sphere needs 'radius'"))
        if shape in ("cylinder", "cone") and not (g.get("height") and (g.get("radius") or g.get("radius_top") is not None)): errors.append(_err(path, "missing_dims", f"{shape} needs 'radius' and 'height'"))
        for key in ("extent", "size"):
            if key in g and any((not isinstance(x, (int, float))) or x <= 0 for x in g[key]): errors.append(_err(f"{path}/{key}", "non_positive", f"{key} must be positive"))
    elif kind in ("asset", "generated"):
        if any(x <= 0 for x in g.get("extent", [1])): errors.append(_err(path + "/extent", "non_positive", "extent must be positive"))

def _check_motion(m: dict, path: str, duration, errors: list[dict], warnings: list[dict]) -> None:
    t = m.get("type")
    if t == "trajectory":
        kf = m.get("keyframes") or []
        if len(kf) < 2: errors.append(_err(path, "too_few_keyframes", "trajectory needs at least 2 keyframes"))
        ts = [k["t"] for k in kf]
        if ts != sorted(ts): errors.append(_err(path + "/keyframes", "not_monotonic", "keyframe times must be non-decreasing"))
    if t in ("revolute", "prismatic", "periodic_translate", "periodic_rotate", "spin"):
        ax = m.get("axis")
        if ax is not None and (not _finite(ax) or sum(x * x for x in ax) < 1e-9): errors.append(_err(path + "/axis", "zero_axis", "axis must be a non-zero vector"))
    if t == "revolute":
        r = m.get("range_deg")
        if r and r[0] > r[1]: errors.append(_err(path + "/range_deg", "bad_range", "range_deg must be [min, max]"))
        if not m.get("schedule") and not m.get("rate_dps"): warnings.append(_err(path, "no_drive", "revolute has neither schedule nor rate_dps; it will stay still"))
    if t == "prismatic" and not m.get("schedule") and not m.get("rate"): warnings.append(_err(path, "no_drive", "prismatic has neither schedule nor rate"))
    sch = m.get("schedule") or []
    ts = [s["t"] for s in sch]
    if ts != sorted(ts): errors.append(_err(path + "/schedule", "not_monotonic", "schedule times must be non-decreasing"))
    for k, s in enumerate(sch):
        if t == "revolute" and "to_deg" not in s: errors.append(_err(f"{path}/schedule/{k}", "missing_to", "revolute schedule entries need 'to_deg'"))
        if t == "prismatic" and "to" not in s: errors.append(_err(f"{path}/schedule/{k}", "missing_to", "prismatic schedule entries need 'to'"))
        if duration and s["t"] > duration * 1.5: warnings.append(_err(f"{path}/schedule/{k}", "after_clip", "schedule time is far beyond the clip duration"))
    if t in ("periodic_translate", "periodic_rotate") and not m.get("period"): errors.append(_err(path, "missing_period", f"{t} needs 'period'"))

def validate(program: Any) -> dict:
    """Return {'ok': bool, 'errors': [...], 'warnings': [...]} ."""
    errs = schema_errors(program)
    warns: list[dict] = []
    if not errs and isinstance(program, dict):
        e2, warns = semantic_errors(program)
        errs.extend(e2)
    return {"ok": not errs, "errors": errs, "warnings": warns}

def format_errors(report: dict, limit: int = 30) -> str:
    lines = [f"- {e['path']}: [{e['code']}] {e['message']}" + (f" -> {e['suggestion']}" if e.get('suggestion') else "") for e in report["errors"][:limit]]
    return "\n".join(lines) if lines else "(no errors)"
