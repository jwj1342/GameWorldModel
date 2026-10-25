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

Returned frame count, requested time and actual/state time must match the source
frame within the existing `perception.visibility_time_tolerance_s` (default 1e-4).
A physics step that cannot represent the source time is not relabeled as exact.
Visibility checks and their tolerance are not relaxed. The reference Program must
already share the input clip's time origin; this does not infer crop offsets.

Run 9478 completed but had 224 rejected references: source times began at 0.1,
0.366667, while GT used 0, 0.25. Its zero change in `static->object` is not evidence
of visibility-gate efficacy. This patch fixes that timing path; it does not prove
motion accuracy, physical contact or real-video improvement. Future A/B runs must
apply the same timing correction to both arms and keep those runs distinct.
