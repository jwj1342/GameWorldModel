"""Read-only geometry diagnostics; never infer a correction from a conflict."""
from __future__ import annotations

import math
from copy import deepcopy
from statistics import median

from gwm.compiler.geometry import _CONTACT_TOLERANCE_M, geometry_findings, rotation_matrix


def _vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(not isinstance(v, bool) and isinstance(v, (int, float))
                    and math.isfinite(v) for v in value))


def audit_evidence_geometry(evidence: dict, program: dict | None = None, *,
                            tolerance: float = _CONTACT_TOLERANCE_M,
                            legacy_contact_dist: float | None = None) -> dict:
    """Return OBB/ground gaps, contact conflicts and optional Candidate findings.

    Gaps use the plane equation n.p + offset = 0 and the rotated box support
    radius. Negative gaps are below the declared plane, not intersection depth
    with a finite ground solid. This cannot decide which observation is wrong.
    ``legacy_contact_dist`` reproduces the historical producer's tolerance only
    when the caller provides its actual config value; it is not a new rule.
    """
    for value in (tolerance, legacy_contact_dist):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError("tolerances must be finite non-negative numbers")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("objects"), list):
        raise ValueError("evidence must contain an objects array")
    diagnostics, rows = [], []

    def finding(code, path, hint):
        diagnostics.append({"stage": "evidence_geometry_audit", "severity": "warning",
                            "code": code, "path": path, "hint": hint})

    planes = [p for p in (evidence.get("static") or {}).get("planes", [])
              if isinstance(p, dict) and p.get("kind") == "ground"]
    plane = None
    if len(planes) != 1:
        finding("ground_unresolved", "/static/planes", "Require one declared ground plane; do not choose between candidates.")
    else:
        p = planes[0]
        n, d = p.get("normal"), p.get("offset")
        if (_vector(n, 3) and _vector([d], 1)
                and math.isfinite(math.hypot(*n)) and math.hypot(*n) > 1e-12):
            norm = math.hypot(*n)
            plane = ([v / norm for v in n], d / norm)
        else:
            finding("ground_unresolved", "/static/planes", "Ground normal/offset missing or invalid; do not assume y=0.")

    object_ids = set()
    for i, obj in enumerate(evidence["objects"]):
        path = f"/objects/{i}"
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or obj["id"] in object_ids:
            raise ValueError("object IDs must be present and unique")
        object_ids.add(obj["id"])
        obbs = obj.get("obb")
        if not isinstance(obbs, list) or not obbs:
            finding("observations_unavailable", path + "/obb", "No OBB observations available.")
            continue
        heights = [b["size"][1] for b in obbs if isinstance(b, dict)
                   and _vector(b.get("size"), 3) and min(b["size"]) > 0]
        legacy_tol = (legacy_contact_dist * max(1., median(heights) / .1)
                      if legacy_contact_dist is not None and heights else None)
        contacts = obj.get("contacts") if isinstance(obj.get("contacts"), list) else []
        intervals = []
        for j, c in enumerate(contacts):
            if not isinstance(c, dict) or c.get("with_id") != "ground":
                continue
            start, end = c.get("t_start"), c.get("t_end")
            if _vector([start, end], 2) and start <= end:
                intervals.append((start, end))
            else:
                finding("contact_time_unresolved", path + f"/contacts/{j}", "Cannot assign contact to observations without a valid interval.")
        previous = None
        for j, b in enumerate(obbs):
            bp = path + f"/obb/{j}"
            if not isinstance(b, dict):
                finding("invalid_obb", bp, "OBB must be an object.")
                continue
            t, center, size, quat = b.get("t"), b.get("center"), b.get("size"), b.get("quat")
            valid_time = _vector([t], 1) and (previous is None or t > previous)
            if _vector([t], 1):
                previous = t
            valid = (valid_time and _vector(center, 3) and _vector(size, 3)
                     and min(size) > 0 and _vector(quat, 4)
                     and abs(math.hypot(*quat) - 1.) <= 1e-3)
            gap = None
            if not valid:
                finding("invalid_obb", bp, "Require ordered finite time, center, positive size and a unit quaternion; no implicit repair.")
            elif plane is not None:
                normal, offset = plane
                rot = rotation_matrix(quat)
                radius = sum(abs(sum(normal[r] * rot[r][c] for r in range(3))) * size[c] / 2
                             for c in range(3))
                gap = sum(normal[r] * center[r] for r in range(3)) + offset - radius
            claimed = any(a <= t <= z for a, z in intervals) if valid_time else None
            conflict = bool(claimed and gap is not None and abs(gap) > tolerance)
            if gap is not None and gap < -tolerance:
                finding("below_declared_ground", bp, "OBB extends below the declared plane; ground/geometry/transform cause is unresolved.")
            if conflict:
                finding("ground_contact_conflict", bp, "Declared ground contact disagrees with the OBB-plane gap; do not snap or guess a correction.")
            rows.append({"object_id": obj["id"], "path": bp, "t": t if _vector([t], 1) else None,
                         "gap": gap, "ground_contact_claimed": claimed,
                         "contact_conflict": conflict,
                         "legacy_tolerance": legacy_tol,
                         "legacy_on_ground": (abs(center[1] - size[1] / 2) < legacy_tol
                                              if valid and legacy_tol is not None else None)})

    candidate = None
    if program is not None:
        if not isinstance(program, dict) or not isinstance(program.get("objects"), list):
            raise ValueError("program must be the extracted Program object, not a validation wrapper")
        errors, warnings = geometry_findings(program)
        # Ground and support errors may describe the same physical pair. Keep
        # original diagnostics, but report unique affected object paths too.
        affected = sorted({d["path"] for d in errors
                           if d["code"] in {"ground_penetration", "support_penetration"}})
        nodes = {}
        for group in ("objects", "static"):
            for node in program.get(group, []):
                nodes.setdefault(node.get("id"), []).append(node)
        comparisons = []
        for obj in evidence["objects"]:
            matches = nodes.get(obj["id"], [])
            first = (obj.get("obb") or [None])[0]
            fields = {"center": None, "size": None, "quat": None}
            if len(matches) == 1 and isinstance(first, dict):
                node = matches[0]
                pairs = {"center": ((node.get("pose") or {}).get("pos"), first.get("center"), 3),
                         "size": ((node.get("geom") or {}).get("extent"), first.get("size"), 3),
                         "quat": ((node.get("pose") or {}).get("quat"), first.get("quat"), 4)}
                for key, (actual, observed, length) in pairs.items():
                    if _vector(actual, length) and _vector(observed, length):
                        error = max(abs(a - b) for a, b in zip(actual, observed))
                        if key == "quat":  # q and -q describe the same rotation.
                            error = min(error, max(abs(a + b) for a, b in zip(actual, observed)))
                        fields[key] = error <= .000500001  # Writer's three-decimal summary.
            comparisons.append({"object_id": obj["id"], "field_matches": fields,
                                "status": "changed" if False in fields.values() else
                                          "unavailable" if None in fields.values() else "matches_first_obb"})
        candidate = {"errors": errors, "warnings": warnings,
                     "penetration_pose_paths": affected,
                     "penetration_pose_count": len(affected),
                     "first_obb_comparison": comparisons,
                     "comparison_tolerance": .000500001,
                     "comparison_note": "Exact IDs only. A change is not proof of a Writer error; scale/representation changes require provenance."}
    incomplete = not rows or plane is None or any(r["gap"] is None for r in rows)
    incomplete = incomplete or any(d["code"] in {"observations_unavailable", "contact_time_unresolved"}
                                   for d in diagnostics)
    return {"status": "partial" if incomplete else "measured",
            "root_cause": "unresolved", "tolerance": tolerance,
            "units": (evidence.get("meta") or {}).get("scale", "unknown"),
            "alignment_record": deepcopy((evidence.get("meta") or {}).get("alignment")),
            "rows": rows, "diagnostics": diagnostics,
            "below_ground_objects": sorted({r["object_id"] for r in rows
                                            if r["gap"] is not None and r["gap"] < -tolerance}),
            "contact_conflict_objects": sorted({r["object_id"] for r in rows if r["contact_conflict"]}),
            "candidate": candidate,
            "limitations": ["No raw point-cloud/transform replay: root cause is not established.",
                            "Plane gaps are not finite-solid intersection depths or independently measured metres.",
                            "Legacy votes reproduce the supplied historical threshold, not physical contact confidence."]}
