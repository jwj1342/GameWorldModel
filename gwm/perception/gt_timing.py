"""Explicit work-frame timing for synthetic GT backends; never infer from FPS."""
import math
from numbers import Integral, Real


def _time(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def requested_frame_times(frame_files, frame_indices, frame_times):
    """Return exact clip-relative times, paired positionally with working frames."""
    if frame_times is None:
        raise ValueError("GT requires explicit frame_times; frame index / FPS is not a source timestamp")
    if not frame_files or len(frame_files) != len(frame_indices) or len(frame_times) != len(frame_indices):
        raise ValueError("GT frame files, indices and times must have the same nonzero length")
    if any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in frame_indices) or len(set(frame_indices)) != len(frame_indices):
        raise ValueError("GT frame indices must be unique non-negative integers")
    if any(not _time(t) for t in frame_times):
        raise ValueError("GT frame_times must be finite non-negative real numbers")
    times = tuple(float(t) for t in frame_times)
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("GT frame_times must be strictly increasing")
    # Current harness file names have millisecond labels. Reject aliasing instead
    # of reading a later frame through an overwritten path.
    if len({f"{t:.3f}" for t in times}) != len(times):
        raise ValueError("GT frame times collide in current millisecond render file labels")
    return times


def validate_render_times(index, times, cfg):
    """Refuse truncated, rounded or quantized-to-another-time GT renders."""
    tolerance = cfg.get("perception", {}).get("visibility_time_tolerance_s", 1e-4)
    if not _time(tolerance):
        raise ValueError("GT time tolerance must be finite and non-negative")
    frames = index.get("frames")
    if not isinstance(frames, list) or len(frames) != len(times):
        raise ValueError("GT render frame count differs from the requested work frames")
    for frame, expected in zip(frames, times):
        if not isinstance(frame, dict):
            raise ValueError("GT render frame must be an object")
        for field in ("requested_t", "actual_t", "t"):
            value = frame.get(field)
            if not _time(value) or abs(value - expected) > tolerance:
                raise ValueError(f"GT render {field} does not match source frame time {expected}")
        state_time = (frame.get("state") or {}).get("t")
        if not _time(state_time) or abs(state_time - expected) > tolerance:
            raise ValueError(f"GT rendered state does not match source frame time {expected}")
