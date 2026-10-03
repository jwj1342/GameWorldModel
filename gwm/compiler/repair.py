"""Conservative, auditable repairs for independently validated candidate programs.

Only local geometric invariants are eligible.  No field is edited in place, and
each proposed edit is checked by the existing candidate validator before it is
accepted.  The default mode is off.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from .geometry import _aabb, _overlaps_xz
from .validate import validate_candidate
from ..perception.quality import assess_evidence_quality


DEFAULTS = {
    "mode": "off",
    "max_passes": 3,
    "max_quaternion_norm_delta": 0.02,
    "max_contact_shift_m": 0.12,
    "max_evidence_position_error_m": 0.10,
    "min_contact_confidence": 0.80,
    "min_geometry_confidence": 0.80,
    "min_support_overlap_fraction": 0.50,
    "max_support_tilt_deg": 2.0,
}
_QUAT_CODE = "non_unit_quaternion"
_CONTACT_CODES = {"support_penetration", "support_gap", "ground_penetration"}


def _settings(config: dict | None) -> dict:
    supplied = (config or {}).get("candidate_repair", config or {})
    if not isinstance(supplied, dict):
        raise ValueError("candidate_repair config must be a mapping")
    settings = {**DEFAULTS, **supplied}
    if settings["mode"] not in {"off", "report", "apply"}:
        raise ValueError("candidate_repair.mode must be off, report, or apply")
    passes = settings["max_passes"]
    if isinstance(passes, bool) or not isinstance(passes, int) or not 1 <= passes <= 10:
        raise ValueError("candidate_repair.max_passes must be an integer in [1, 10]")
    positive = ("max_quaternion_norm_delta", "max_contact_shift_m", "max_evidence_position_error_m", "max_support_tilt_deg")
    if any(not _number(settings[key]) or settings[key] <= 0 for key in positive):
        raise ValueError("candidate repair tolerances must be finite and positive")
    fractions = ("min_contact_confidence", "min_geometry_confidence", "min_support_overlap_fraction")
    if any(_number(settings[key]) is None or not 0 <= settings[key] <= 1 for key in fractions):
        raise ValueError("candidate repair confidence thresholds must be in [0, 1]")
    return settings


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == length and all(_number(item) for item in value)


def _at(document: Any, path: str) -> Any:
    node = document
    for token in path.strip("/").split("/"):
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def _unresolved(report: dict, reasons: dict[tuple[str, str], str] | None = None) -> list[dict]:
    reasons = reasons or {}
    return [{**item, "reason": reasons.get((item["path"], item["code"]), "not eligible for automatic repair")}
            for item in report["findings"]]


def _quality(evidence: Any, config: dict | None) -> dict:
    try:
        return assess_evidence_quality(evidence, config)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return {"decision": "block", "diagnostics": [{"code": "evidence_quality_error", "message": str(exc)}],
                "evidence": None}


def _quaternion_action(program: dict, finding: dict, settings: dict) -> tuple[dict | None, str]:
    path = finding["path"]
    if not path.endswith("/quat"):
        return None, "diagnostic does not identify a quaternion field"
    try:
        before = _at(program, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None, "quaternion path no longer resolves"
    if not _finite_vector(before, 4):
        return None, "quaternion is missing or non-finite"
    norm = math.sqrt(sum(float(value) ** 2 for value in before))
    if norm <= 1e-12:
        return None, "zero quaternion has no recoverable orientation"
    if abs(norm - 1.0) > settings["max_quaternion_norm_delta"]:
        return None, "quaternion norm is too far from one"
    after = [float(value) / norm for value in before]
    return {"path": path, "code": _QUAT_CODE, "before": copy.deepcopy(before), "after": after,
            "rule": "normalize_near_unit_quaternion", "evidence_refs": [{"path": "/meta", "role": "quality_gate"}],
            "confidence": 1.0, "reason": "unit-quaternion invariant fixes only scale, preserving orientation"}, ""


def _horizontal_overlap_fraction(a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]) -> float:
    width = max(0.0, min(a[1][0], b[1][0]) - max(a[0][0], b[0][0]))
    depth = max(0.0, min(a[1][2], b[1][2]) - max(a[0][2], b[0][2]))
    own_area = max((a[1][0] - a[0][0]) * (a[1][2] - a[0][2]), 1e-12)
    return width * depth / own_area


def _horizontal_surface(pose: dict, settings: dict) -> bool:
    quat = pose.get("quat", [0, 0, 0, 1])
    if not _finite_vector(quat, 4):
        return False
    norm = math.sqrt(sum(float(value) ** 2 for value in quat))
    if norm <= 1e-12:
        return False
    x, z = float(quat[0]) / norm, float(quat[2]) / norm
    up_y = 1 - 2 * (x * x + z * z)
    return up_y >= math.cos(math.radians(settings["max_support_tilt_deg"]))


def _contact_action(program: dict, evidence: dict, index: int, codes: set[str], settings: dict) -> tuple[dict | None, str]:
    path = f"/objects/{index}/pose/pos"
    obj = program["objects"][index]
    support_id = obj.get("support")
    if not support_id:
        return None, "object has no explicit support relation; floating may be intentional"
    if (obj.get("motion") or {}).get("type", "static") != "static" or obj.get("events"):
        return None, "moving or interactive object must not be snapped to a static surface"
    if (program.get("meta") or {}).get("units") != "m" or (program.get("meta") or {}).get("up") != "y":
        return None, "program does not declare metre and y-up world coordinates"
    if (evidence.get("meta") or {}).get("scale") != "metric":
        return None, "Evidence lacks metric scale for a metre-valued correction"
    target_matches = [(i, node) for i, node in enumerate(program.get("static") or []) if node.get("id") == support_id]
    if len(target_matches) != 1:
        return None, "support is missing, dynamic, or not unique"
    support_index, support = target_matches[0]
    support_geometry = support.get("geom") or {}
    if support_geometry.get("kind") != "primitive" or support_geometry.get("shape") not in {"box", "plane"}:
        return None, "support geometry does not define a flat surface"
    if not _horizontal_surface(support.get("pose") or {}, settings):
        return None, "support surface is not reliably horizontal"
    obj_pose = obj.get("pose") or {}
    obj_bounds = _aabb(obj, obj_pose)
    support_bounds = _aabb(support, support.get("pose") or {})
    if obj_bounds is None or support_bounds is None or not all(_number(value) for bounds in (obj_bounds, support_bounds) for end in bounds for value in end):
        return None, "object or support geometry is missing or non-finite"
    if not _overlaps_xz(obj_bounds, support_bounds) or _horizontal_overlap_fraction(obj_bounds, support_bounds) < settings["min_support_overlap_fraction"]:
        return None, "horizontal support overlap is insufficient"
    delta = support_bounds[1][1] - obj_bounds[0][1]
    if abs(delta) > settings["max_contact_shift_m"]:
        return None, "required vertical correction exceeds configured small-error limit"
    if not any(code in codes for code in _CONTACT_CODES):
        return None, "no current geometric contact finding"

    competing = []
    for node in program.get("static") or []:
        if node.get("id") == support_id:
            continue
        bounds = _aabb(node, node.get("pose") or {})
        if bounds is None or not all(_number(value) for end in bounds for value in end):
            return None, "another static surface has unknown geometry"
        if _horizontal_surface(node.get("pose") or {}, settings) and _horizontal_overlap_fraction(obj_bounds, bounds) >= settings["min_support_overlap_fraction"]:
            if abs(bounds[1][1] - obj_bounds[0][1]) <= settings["max_contact_shift_m"]:
                competing.append(node.get("id"))
    if competing:
        return None, "multiple plausible support surfaces overlap: " + ", ".join(sorted(str(value) for value in competing))

    evidence_objects = [(i, item) for i, item in enumerate(evidence.get("objects") or []) if item.get("id") == obj.get("id")]
    if len(evidence_objects) != 1:
        return None, "Evidence has no unique object identity matching the program"
    evidence_index, observed = evidence_objects[0]
    if observed.get("is_dynamic") is not False or (observed.get("motion_guess") or {}).get("type") != "static":
        return None, "Evidence does not establish that the object is static"
    geometry_conf = (observed.get("attribute_confidence") or {}).get("geometry")
    if not _number(geometry_conf) or geometry_conf < settings["min_geometry_confidence"]:
        return None, "Evidence geometry confidence is insufficient"
    contacts = observed.get("contacts") or []
    matching = [(i, item) for i, item in enumerate(contacts) if item.get("with_id") == support_id]
    if len(matching) != 1 or any(item.get("with_id") != support_id for item in contacts):
        return None, "Evidence contact is missing, ambiguous, or semantically conflicting"
    contact_index, contact = matching[0]
    contact_conf = contact.get("conf")
    if not _number(contact_conf) or contact_conf < settings["min_contact_confidence"]:
        return None, "Evidence contact confidence is insufficient"
    measured = [(i, obb) for i, obb in enumerate(observed.get("obb") or [])
                if _finite_vector(obb.get("center"), 3) and _number(obb.get("conf")) and obb["conf"] >= settings["min_geometry_confidence"]]
    if not measured:
        return None, "Evidence has no reliable world-space object centre"
    proposed_y = float(obj_pose["pos"][1]) + delta
    before = copy.deepcopy(obj_pose["pos"])
    after = [float(value) for value in before]
    after[1] = proposed_y
    if any(math.dist(obb["center"], after) > settings["max_evidence_position_error_m"] for _, obb in measured):
        return None, "Evidence world-space object centres disagree with the proposed contact pose"
    code = "support_penetration" if "support_penetration" in codes else ("support_gap" if "support_gap" in codes else "ground_penetration")
    refs = [{"path": f"/objects/{evidence_index}/contacts/{contact_index}", "role": "support_contact"}]
    refs.extend({"path": f"/objects/{evidence_index}/obb/{index}", "role": "world_geometry"} for index, _ in measured)
    return {"path": path, "code": code, "before": before, "after": after,
            "rule": "snap_small_static_support_offset", "evidence_refs": refs,
            "confidence": round(min(float(contact_conf), float(geometry_conf), *(float(obb["conf"]) for _, obb in measured)), 6),
            "reason": f"unique static support '{support_id}' at program /static/{support_index}; world-y correction {delta:+.4f} m"}, ""


def _proposals(program: dict, evidence: dict, validation: dict, settings: dict) -> tuple[list[dict], dict[tuple[str, str], str]]:
    findings = validation["findings"]
    reasons: dict[tuple[str, str], str] = {}
    quaternions = [item for item in findings if item["code"] == _QUAT_CODE]
    actions: list[dict] = []
    if quaternions:
        for item in quaternions:
            action, reason = _quaternion_action(program, item, settings)
            if action:
                actions.append(action)
            else:
                reasons[(item["path"], item["code"])] = reason
        return actions, reasons

    contact_by_object: dict[int, set[str]] = {}
    for item in findings:
        parts = item["path"].strip("/").split("/")
        if item["code"] in _CONTACT_CODES and len(parts) >= 3 and parts[0] == "objects" and parts[1].isdigit():
            contact_by_object.setdefault(int(parts[1]), set()).add(item["code"])
    for index, codes in sorted(contact_by_object.items()):
        action, reason = _contact_action(program, evidence, index, codes, settings)
        if action:
            actions.append(action)
        else:
            for code in codes:
                reasons[(f"/objects/{index}/pose", code)] = reason
    return actions, reasons


def _apply_actions(program: dict, actions: list[dict]) -> dict:
    candidate = copy.deepcopy(program)
    for action in actions:
        parent, key = action["path"].rsplit("/", 1)
        _at(candidate, parent)[key] = copy.deepcopy(action["after"])
    return candidate


def repair_candidate(program: Any, evidence: Any, diagnostics: dict | list[dict] | None = None,
                     config: dict | None = None) -> tuple[Any, dict, list[dict]]:
    """Return (program copy, audit report, unresolved findings).

    ``report`` mode dry-runs the same bounded validation loop and reports its
    proposed edits, while returning an unchanged copy.  ``apply`` accepts an
    edit only when revalidation introduces no new errors.
    """
    settings = _settings(config)
    original = copy.deepcopy(program)
    initial = validate_candidate(original)
    report = {"mode": settings["mode"], "initial_validation": initial, "final_validation": initial,
              "repairs": [], "proposed_repairs": [], "passes": 0, "evidence_quality": None,
              "diagnostics_verified": True}
    if settings["mode"] == "off":
        return original, report, _unresolved(initial, {(item["path"], item["code"]): "automatic repair mode is off" for item in initial["findings"]})

    if diagnostics is not None:
        given = diagnostics.get("findings", []) if isinstance(diagnostics, dict) else diagnostics
        given_keys = sorted((item.get("path"), item.get("code"), item.get("stage"), item.get("severity")) for item in given)
        fresh_keys = sorted((item["path"], item["code"], item["stage"], item["severity"]) for item in initial["findings"])
        report["diagnostics_verified"] = given_keys == fresh_keys
        if not report["diagnostics_verified"]:
            unresolved = _unresolved(initial)
            unresolved.append({"path": "/", "code": "stale_diagnostics", "reason": "supplied diagnostics differ from fresh validation"})
            return original, report, unresolved

    quality = _quality(evidence, config)
    report["evidence_quality"] = {"decision": quality["decision"],
                                  "diagnostic_codes": sorted({item["code"] for item in quality.get("diagnostics", [])})}
    if quality["decision"] != "proceed":
        unresolved = _unresolved(initial, {(item["path"], item["code"]): "Evidence quality is not proceed" for item in initial["findings"]})
        unresolved.append({"path": "/evidence", "code": "evidence_quality_not_proceed",
                           "reason": "Evidence quality decision is " + quality["decision"]})
        return original, report, unresolved

    program_clip = (original.get("meta") or {}).get("clip") if isinstance(original, dict) else None
    evidence_clip = ((quality.get("evidence") or {}).get("meta") or {}).get("clip")
    if not program_clip or program_clip != evidence_clip:
        unresolved = _unresolved(initial, {(item["path"], item["code"]): "Program and Evidence clip identities do not match" for item in initial["findings"]})
        unresolved.append({"path": "/meta/clip", "code": "evidence_clip_mismatch",
                           "reason": "Program and Evidence must name the same clip"})
        return original, report, unresolved
    safe_error_codes = {_QUAT_CODE, "support_penetration", "ground_penetration"}
    unsafe = [item for item in initial["errors"] if item["code"] not in safe_error_codes]
    if unsafe:
        return original, report, _unresolved(initial, {
            (item["path"], item["code"]): "candidate has an unrelated schema, semantic, or unsafe geometry error"
            for item in initial["findings"]})

    working = copy.deepcopy(original)
    validation = initial
    refusal_reasons: dict[tuple[str, str], str] = {}
    for _ in range(settings["max_passes"]):
        actions, reasons = _proposals(working, quality["evidence"], validation, settings)
        refusal_reasons.update(reasons)
        if not actions:
            break
        attempted = _apply_actions(working, actions)
        next_validation = validate_candidate(attempted)
        old_errors = {(item["path"], item["code"]) for item in validation["errors"]}
        new_errors = {(item["path"], item["code"]) for item in next_validation["errors"]}
        remaining = {(item["path"], item["code"]) for item in next_validation["findings"]}
        target_left = any(
            (action["path"], _QUAT_CODE) in remaining if action["code"] == _QUAT_CODE
            else any(path == action["path"].rsplit("/", 1)[0] and code in _CONTACT_CODES for path, code in remaining)
            for action in actions
        )
        if new_errors - old_errors or target_left:
            for action in actions:
                finding_path = action["path"] if action["code"] == _QUAT_CODE else action["path"].rsplit("/", 1)[0]
                refusal_reasons[(finding_path, action["code"])] = "revalidation did not clear the finding or introduced a new error; proposed pass was rolled back"
            break
        working, validation = attempted, next_validation
        report["proposed_repairs"].extend(actions)
        report["passes"] += 1
    if settings["mode"] == "apply":
        report["repairs"] = copy.deepcopy(report["proposed_repairs"])
        report["final_validation"] = validation
        return working, report, _unresolved(validation, refusal_reasons)
    report["simulated_validation"] = validation
    for action in report["proposed_repairs"]:
        finding_path = action["path"] if action["code"] == _QUAT_CODE else action["path"].rsplit("/", 1)[0]
        refusal_reasons[(finding_path, action["code"])] = "proposed in report mode; not applied"
    return original, report, _unresolved(initial, refusal_reasons)
