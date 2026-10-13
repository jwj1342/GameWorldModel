"""In-frame occlusion fractions from independently rendered instance silhouettes."""
import math
import numpy as np


def measure_visibility(visible, unoccluded):
    visible, unoccluded = np.asarray(visible), np.asarray(unoccluded)
    if visible.ndim != 2 or visible.shape != unoccluded.shape:
        return "unknown", "mask_size_mismatch"
    if visible.dtype != bool or unoccluded.dtype != bool:
        return "unknown", "non_boolean_masks"
    if np.any(visible & ~unoccluded):
        return "unknown", "visible_pixels_outside_reference"
    denominator = int(unoccluded.sum())
    if not denominator:
        return "unknown", "no_in_frame_reference"  # No guess about occlusion vs out of view.
    return float(visible.sum() / denominator), ""


def checked_visibility(record, object_id, frame_index, t, image_size, program_hash="", tolerance=1e-4):
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("visibility time tolerance must be finite and non-negative")
    if not isinstance(record, dict):
        return "unknown", "missing_reference"
    if not program_hash:
        return "unknown", "geometry_source_unverified"
    if record.get("same_state_verified") is not True:
        return "unknown", "reference_state_unverified"
    if record.get("object_id") != object_id or record.get("frame_index") != frame_index:
        return "unknown", "identity_or_frame_mismatch"
    if record.get("image_size") != list(image_size):
        return "unknown", "image_size_mismatch"
    if record.get("program_sha256") != program_hash:
        return "unknown", "program_source_mismatch"
    for field in ("requested_t", "actual_t"):
        value = record.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value - t) > tolerance:
            return "unknown", "time_mismatch"
    value = record.get("fraction")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        return "unknown", record.get("reason") or "invalid_fraction"
    if record.get("method") != "isolated_instance_id":
        return "unknown", "unsupported_reference_method"
    return float(value), ""
