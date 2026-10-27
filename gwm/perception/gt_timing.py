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


def validate_sampled_time(requested_t, actual_t, time_sampling=None, tolerance=1e-4):
    """Validate exact sampling, or the declared nearest physics frame, without relabeling."""
    if not _time(tolerance):
        raise ValueError("GT time tolerance must be finite and non-negative")
    if not _time(requested_t) or not _time(actual_t):
        raise ValueError("GT sample times must be finite and non-negative")
    # Old caches without sampling metadata retain strict exact-time semantics.
    if time_sampling is None:
        time_sampling = {"mode": "exact"}
    if not isinstance(time_sampling, dict):
        raise ValueError("GT time_sampling must be an object")
    mode = time_sampling.get("mode")
    expected = requested_t
    if mode == "nearest_physics_frame":
        dt = time_sampling.get("dt")
        if not _time(dt) or dt == 0:
            raise ValueError("GT physics dt must be finite and positive")
        frame = requested_t / dt
        if not math.isfinite(frame) or frame > 2**53 - 1:
            raise ValueError("GT physics frame is outside the supported range")
        # JavaScript Math.round on nonnegative values: ties go toward +infinity.
        # Do not use Python round (ties-to-even) or floor(frame + .5), which can
        # round a representable value just below .5 up before floor sees it.
        lower = math.floor(frame)
        expected = (lower + int(frame - lower >= .5)) * dt
        if abs(actual_t - requested_t) > dt / 2 + tolerance:
            raise ValueError("GT actual time is outside the nearest physics step")
    elif mode != "exact":
        raise ValueError("GT time_sampling mode is unsupported")
    if abs(actual_t - expected) > tolerance:
        raise ValueError("GT actual time does not match the declared sampled frame")


def validate_render_times(index, times, cfg):
    """Keep request identity strict; validate actual/state times under declared sampling."""
    tolerance = cfg.get("perception", {}).get("visibility_time_tolerance_s", 1e-4)
    if not _time(tolerance):
        raise ValueError("GT time tolerance must be finite and non-negative")
    frames = index.get("frames")
    if not isinstance(frames, list) or len(frames) != len(times):
        raise ValueError("GT render frame count differs from the requested work frames")
    for frame, expected in zip(frames, times):
        if not isinstance(frame, dict):
            raise ValueError("GT render frame must be an object")
        requested = frame.get("requested_t")
        if not _time(requested) or abs(requested - expected) > tolerance:
            raise ValueError(f"GT render requested_t does not match source frame time {expected}")
        actual = frame.get("actual_t")
        try:
            validate_sampled_time(expected, actual, index.get("time_sampling"), tolerance)
        except ValueError as exc:
            raise ValueError(f"GT render actual_t does not match source frame time {expected}: {exc}") from exc
        value = frame.get("t")
        if not _time(value) or abs(value - actual) > tolerance:
            raise ValueError(f"GT render t does not match source frame time {expected}'s actual sample")
        state_time = (frame.get("state") or {}).get("t")
        if not _time(state_time) or abs(state_time - actual) > tolerance:
            raise ValueError(f"GT rendered state does not match source frame time {expected}")
