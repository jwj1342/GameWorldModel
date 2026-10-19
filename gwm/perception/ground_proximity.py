"""Sample-local OBB/plane compatibility, never physical contact inference."""
from __future__ import annotations

import math

UNKNOWN = "unknown"


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(_number(v) for v in value)


def _settings(config):
    pc = (config or {}).get("perception", {})
    supplied = pc.get("ground_proximity", {})
    legacy = pc.get("contact_dist_m", .05)
    if not _number(legacy) or legacy < 0:
        raise ValueError("contact_dist_m must be finite and non-negative")
    # Preserve tighter legacy configs, but never inherit height multiplication.
    tolerance = supplied.get("max_gap_world", min(float(legacy), .05))
    minimum = supplied.get("min_ground_confidence", .8)
    if not _number(tolerance) or tolerance < 0:
        raise ValueError("ground_proximity.max_gap_world must be finite and non-negative")
    if not _number(minimum) or not 0 <= minimum <= 1:
        raise ValueError("ground_proximity.min_ground_confidence must be within [0, 1]")
    return float(tolerance), float(minimum)


def measure_ground_proximity(obbs, planes, config=None, *, frames=None,
                             scale=UNKNOWN, alignment=None):
    """Return sample-point geometry and unknown physical-contact confidence.

    No interpolated time intervals, height scaling, position edits or contact
    votes. ``frames`` is an optional expected source registry; missing OBBs remain
    unknown. Distances are in aligned-world units, metric only if independently
    established by the caller's scale provenance.
    """
    # compiler.__init__ exports Repair, which consumes the marker predicate in
    # this module. Defer the geometry import until this module is initialized.
    from ..compiler.geometry import rotation_matrix

    tolerance, minimum = _settings(config)
    issues = []
    grounds = [p for p in planes if isinstance(p, dict) and p.get("kind") == "ground"]
    plane = None
    if len(grounds) != 1:
        reason = "ambiguous_ground" if grounds else "missing_ground"
    elif alignment is not None and isinstance(alignment.get("ground_candidate_count"), int) and alignment["ground_candidate_count"] > 1:
        reason = "ambiguous_ground"
    else:
        ground = grounds[0]
        n, d, conf = ground.get("normal"), ground.get("offset"), ground.get("conf")
        reason = "invalid_ground"
        if _vector(n, 3) and _number(d):
            length = math.hypot(*n)
            if math.isfinite(length) and length > 1e-12:
                if (not _number(conf) or not 0 <= conf <= 1 or conf < minimum or
                        (alignment is not None and alignment.get("ground") in (None, "not_found"))):
                    reason = "unreliable_ground"
                else:
                    plane = ([v / length for v in n], float(d) / length)
                    reason = "sample_points_only"
    if plane is None:
        issues.append({"code": reason})

    expected = (frames if frames is not None else
                [{"frame_index": b.get("frame", UNKNOWN), "t": b.get("t")}
                 for b in obbs if isinstance(b, dict)])
    samples = []
    seen = set()
    previous = None
    for frame in expected:
        fi, t = frame.get("frame_index", UNKNOWN), frame.get("t")
        if (not _number(t) or t < 0 or (previous is not None and t <= previous)
                or (fi != UNKNOWN and (isinstance(fi, bool) or not isinstance(fi, int) or fi < 0))
                or (fi != UNKNOWN and fi in seen)):
            issues.append({"code": "invalid_frame_registry"})
            continue
        previous = t
        if fi != UNKNOWN:
            seen.add(fi)
        matches = [b for b in obbs if isinstance(b, dict) and _number(b.get("t"))
                   and abs(b["t"] - t) <= 1e-6
                   and (fi == UNKNOWN or b.get("frame") == fi)]
        sample = {"frame_index": fi, "t": float(t), "state": UNKNOWN, "gap": UNKNOWN}
        if len(matches) != 1:
            issues.append({"code": "missing_or_ambiguous_obb", "frame_index": fi})
        elif plane is not None:
            b = matches[0]
            center, size, quat = b.get("center"), b.get("size"), b.get("quat")
            valid = (_vector(center, 3) and _vector(size, 3) and min(size) > 0
                     and _vector(quat, 4) and abs(math.hypot(*quat) - 1.) <= 1e-3)
            if not valid:
                issues.append({"code": "invalid_obb", "frame_index": fi})
            else:
                normal, offset = plane
                rotation = rotation_matrix(quat)
                radius = sum(abs(sum(normal[r] * rotation[r][c] for r in range(3))) * size[c] / 2
                             for c in range(3))
                gap = sum(normal[r] * center[r] for r in range(3)) + offset - radius
                if math.isfinite(gap):
                    sample.update(gap=float(gap), state=("penetrating" if gap < -tolerance else
                                                        "separated" if gap > tolerance else "compatible"))
                else:
                    issues.append({"code": "non_finite_gap", "frame_index": fi})
        samples.append(sample)
    if not samples:
        issues.append({"code": "no_samples"})
    known = sum(s["state"] != UNKNOWN for s in samples)
    return {"basis": "obb_plane_distance", "coordinates": "aligned_world",
            "scale": scale if scale in {"metric", "relative", UNKNOWN} else UNKNOWN,
            "temporal_scope": "sample_points", "confidence": UNKNOWN,
            "physical_contact": UNKNOWN, "tolerance_world": tolerance,
            "status": "unknown" if not known else "partial" if known < len(samples) else "assessed",
            "reason": reason, "samples": samples, "issues": issues}


def is_geometric_proximity(contact):
    """Defense for explicitly marked records, not a reinterpretation of legacy contacts."""
    return isinstance(contact, dict) and contact.get("kind") == "geometric_proximity"
