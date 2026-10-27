# GT backends use actual selected-frame times

`run_perception` passes the selected work frames' clip-relative `t` values to
GT geometry, GT segmentation and the optional GT-camera ablation. Their ordering
matches `frame_files` and `frame_indices`, including nonconsecutive subsets.
Neural backend method calls retain their existing arguments.

For direct GT callers, supply `frame_times=[...]` as a keyword to `estimate` or
`segment`. Omitting it now raises explicitly: frame index / sampling FPS cannot
identify the actual decoded frame. This is a deliberate safety change to the GT
API, not a fallback to inferred timing. It does not rewrite old cache timestamps.

Times must be finite, nonnegative and strictly increasing; all three lists must
have equal nonzero lengths and indices must be valid/unique. The render transport
preserves float precision instead of rounding to milliseconds. Current image file
names still use millisecond labels, so colliding labels are rejected for GT input
rather than allowing a frame's files to overwrite another frame's files.

Returned frame count and requested time must match the source within the existing
`perception.visibility_time_tolerance_s` (default 1e-4). Missing sampling metadata
retains strict exact-time behavior. In `exact` mode, actual/state times also match
the source. Only an explicit `nearest_physics_frame` mode with finite positive
`dt` allows a different actual time: it must equal the nearest physical step
(JavaScript positive-tie rounding), and remain within `dt/2 + tolerance` of the
request. A merely nearby off-grid time or wrong adjacent frame is rejected.
Frame `t` and `state.t` must agree with `actual_t` within the original tolerance.

Visibility records retain `time_sampling` along with unchanged `requested_t` and
`actual_t`, and use the same policy when rebuilt from cache. Physical quantization
is never relabeled as an exact measurement at the source PTS. This is bounded GT
sampling, not a correction of video/model timing or independently measured truth.
The reference Program must already share the input clip's time origin; this does
not infer crop offsets. Even 30/60 fps sources can require quantization if the
selected source times have an offset from the physics frame grid.

Run 9478 completed but had 224 rejected references: source times began at 0.1,
0.366667, while GT used 0, 0.25. Its zero change in `static->object` is not evidence
of visibility-gate efficacy. This patch fixes that timing path; it does not prove
motion accuracy, physical contact or real-video improvement. Future A/B runs must
apply the same timing correction to both arms and keep those runs distinct.
