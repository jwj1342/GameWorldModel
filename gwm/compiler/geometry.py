"""Model-independent geometric preflight checks for scene programs.

The checks in this module are deliberately conservative: they report only
properties that can be established from the program itself, without rendering
or invoking a learned model.  They never mutate the candidate program.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


_GROUND_CLASSES = {"floor", "ground", "terrain"}
_QUAT_TOLERANCE = 1e-3
_CONTACT_TOLERANCE_M = 0.05


def _finding(path: str, code: str, message: str, suggestion: str = "") -> dict:
    return {
        "path": path,
        "code": code,
        "message": message,
        "suggestion": suggestion,
    }


def _finite(values: Iterable[Any]) -> bool:
    try:
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
    except TypeError:
        return False


def _check_transform(transform: dict, path: str, errors: list[dict]) -> None:
    pos = transform.get("pos")
    if pos is not None and not _finite(pos):
        errors.append(_finding(
            f"{path}/pos", "non_finite_transform",
            "position must contain only finite numbers",
            "replace NaN or infinite coordinates with finite evidence-backed values",
        ))

    quat = transform.get("quat")
    if quat is None:
        return
    if not _finite(quat):
        errors.append(_finding(
            f"{path}/quat", "non_finite_quaternion",
            "quaternion must contain only finite numbers",
            "replace it with a finite normalized quaternion",
        ))
        return
    norm = math.sqrt(sum(float(v) * float(v) for v in quat))
    if norm <= 1e-12:
        errors.append(_finding(
            f"{path}/quat", "zero_quaternion",
            "quaternion norm must be non-zero",
            "use the identity quaternion [0, 0, 0, 1] when no rotation is intended",
        ))
    elif abs(norm - 1.0) > _QUAT_TOLERANCE:
        errors.append(_finding(
            f"{path}/quat", "non_unit_quaternion",
            f"quaternion must have unit norm; got {norm:.6g}",
            "normalize the quaternion without changing its rotation",
        ))


def _half_extents(geom: dict) -> tuple[float, float, float] | None:
    extent = geom.get("extent")
    if extent and len(extent) == 3 and _finite(extent):
        return tuple(float(v) / 2.0 for v in extent)
    shape = geom.get("shape")
    if shape == "sphere" and isinstance(geom.get("radius"), (int, float)):
        r = float(geom["radius"])
        return r, r, r
    if shape in {"cylinder", "cone"}:
        radii = [geom.get("radius"), geom.get("radius_top"), geom.get("radius_bottom")]
        radii = [float(v) for v in radii if isinstance(v, (int, float))]
        height = geom.get("height")
        if radii and isinstance(height, (int, float)):
            r = max(radii)
            return r, float(height) / 2.0, r
    if shape == "plane" and geom.get("size") and len(geom["size"]) == 2:
        return float(geom["size"][0]) / 2.0, 0.0, float(geom["size"][1]) / 2.0
    return None


def _aabb(node: dict, pose: dict) -> tuple[list[float], list[float]] | None:
    half = _half_extents(node.get("geom") or {})
    pos = pose.get("pos")
    if half is None or pos is None or not _finite(pos):
        return None

    # Convert local half extents to a conservative world-space AABB.  The
    # absolute rotation matrix is exact for an oriented box's AABB.
    quat = pose.get("quat", [0.0, 0.0, 0.0, 1.0])
    if not _finite(quat):
        return None
    qn = math.sqrt(sum(float(v) * float(v) for v in quat))
    if qn <= 1e-12:
        return None
    x, y, z, w = (float(v) / qn for v in quat)
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    world_half = [sum(abs(rotation[row][col]) * half[col] for col in range(3)) for row in range(3)]
    lo = [float(pos[i]) - world_half[i] for i in range(3)]
    hi = [float(pos[i]) + world_half[i] for i in range(3)]
    return lo, hi


def _overlaps_xz(a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]) -> bool:
    return a[0][0] <= b[1][0] and a[1][0] >= b[0][0] and a[0][2] <= b[1][2] and a[1][2] >= b[0][2]


def _iter_object_poses(program: dict):
    for i, obj in enumerate(program.get("objects", [])):
        if obj.get("pose"):
            yield obj, obj["pose"], f"/objects/{i}/pose"
        for j, pose in enumerate(obj.get("instances") or []):
            yield obj, pose, f"/objects/{i}/instances/{j}"


def geometry_findings(program: dict) -> tuple[list[dict], list[dict]]:
    """Return deterministic geometry errors and warnings without mutation."""
    errors: list[dict] = []
    warnings: list[dict] = []

    for group in ("static", "objects"):
        for i, node in enumerate(program.get(group, [])):
            if node.get("pose"):
                _check_transform(node["pose"], f"/{group}/{i}/pose", errors)
            for j, pose in enumerate(node.get("instances") or []):
                _check_transform(pose, f"/{group}/{i}/instances/{j}", errors)

    for i, keyframe in enumerate((program.get("camera") or {}).get("keyframes") or []):
        _check_transform(keyframe, f"/camera/keyframes/{i}", errors)

    for i, obj in enumerate(program.get("objects", [])):
        motion = obj.get("motion") or {}
        for j, keyframe in enumerate(motion.get("keyframes") or []):
            _check_transform(keyframe, f"/objects/{i}/motion/keyframes/{j}", errors)

    nodes_by_id = {
        node["id"]: node
        for group in ("static", "objects")
        for node in program.get(group, [])
        if node.get("id")
    }
    grounds: list[tuple[dict, tuple[list[float], list[float]]]] = []
    for node in program.get("static", []):
        if str(node.get("class", "")).lower() not in _GROUND_CLASSES:
            continue
        bounds = _aabb(node, node.get("pose") or {})
        if bounds is not None:
            grounds.append((node, bounds))

    for obj, pose, path in _iter_object_poses(program):
        bounds = _aabb(obj, pose)
        if bounds is None:
            continue
        for ground, ground_bounds in grounds:
            if _overlaps_xz(bounds, ground_bounds) and bounds[0][1] < ground_bounds[1][1] - _CONTACT_TOLERANCE_M:
                errors.append(_finding(
                    path, "ground_penetration",
                    f"object '{obj['id']}' penetrates ground '{ground['id']}' by "
                    f"{ground_bounds[1][1] - bounds[0][1]:.3f} m",
                    "lift the object to the ground surface or revise its evidence-backed extent",
                ))
                break

        support_id = obj.get("support")
        support = nodes_by_id.get(support_id) if support_id else None
        if support is None or not support.get("pose"):
            continue
        support_bounds = _aabb(support, support["pose"])
        if support_bounds is None:
            continue
        if not _overlaps_xz(bounds, support_bounds):
            errors.append(_finding(
                path, "support_no_overlap",
                f"object '{obj['id']}' has no horizontal overlap with support '{support_id}'",
                "move the object over its support or correct the support relation",
            ))
            continue
        gap = bounds[0][1] - support_bounds[1][1]
        if gap < -_CONTACT_TOLERANCE_M:
            errors.append(_finding(
                path, "support_penetration",
                f"object '{obj['id']}' penetrates support '{support_id}' by {-gap:.3f} m",
                "lift the object or revise the support geometry",
            ))
        elif gap > _CONTACT_TOLERANCE_M:
            warnings.append(_finding(
                path, "support_gap",
                f"object '{obj['id']}' is {gap:.3f} m above support '{support_id}'",
                "verify the support relation or move the object onto the support",
            ))

    return errors, warnings
