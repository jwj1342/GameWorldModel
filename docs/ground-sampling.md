# Source-bound ground sampling roles

Candidate sampling used to exclude every tracked mask, including a detected
floor surface. This optional policy lets reviewed support masks participate
without changing the object union used for plane qualification. It is not a
new detector, physical ground proof, contact label, or repair by object lifting.

## Defaults and configuration

`perception.ground_sampling.mode` defaults to `legacy`: all object masks are
excluded as before; alignment metadata, numerical operations and RNG use stay
compatible. No role is inferred from a class name, substring, color, size or ID.
The existing taxonomy and downstream motion/repair logic are unchanged.

Opt in with `mode: reviewed_roles` and a `roles` list. Each declaration contains:

- `object_id`: a current Track reference, supplied as data, never a code rule.
- `role`: `ground_support`, `exclude`, or `unknown`.
- `frame_indices`: explicit working frame indices in the current Geometry.
  These are not decoded source frame numbers, times or intervals. They are not
  expanded across gaps, unreviewed frames or the rest of a trajectory.
- For `ground_support`, `source.kind` must be `human_review`, `assistant_review`, or `provider`,
  `source.reference` a nonempty review/observation reference, and
  `source.video_sha256` the known SHA256 of the current video.
- For `ground_support`, `mask_sha256` maps each declared frame's string index to
  `gwm.perception.ground_sampling.mask_sha256(native_mask)`. This fingerprints
  dtype, shape and pixels before resizing. It deliberately differs from a
  digest of bare pixel bytes. A new segmentation or lossy cache cannot silently
  reuse approval of a different mask with the same Track name.

Example structure (placeholders must be replaced by actual reviewed inputs):

```yaml
perception:
  ground_sampling:
    mode: reviewed_roles
    roles:
      - object_id: <current-reviewed-track>
        role: ground_support
        frame_indices: [0]
        source:
          kind: human_review
          reference: <review-record-reference>
          video_sha256: <actual-64-hex-video-sha256>
        mask_sha256:
          '0': <native-mask-fingerprint>
```

Configuration expresses the caller's reviewed sampling role. This module checks
identity/references, not whether a human/provider's semantic judgment is true.
An assistant's visual judgment must be labeled `assistant_review`, never passed
off as human approval or independent ground truth.
It never promotes a provider's ordinary detection score into ground confidence.
Absent and `unknown` roles stay excluded like ordinary foreground. Conflicting
declarations, malformed/unknown references, unknown/mismatched video identity,
or mismatched mask fingerprints fail explicitly, without legacy fallback.

If a declared frame has no observed mask, no mask is invented: it stays excluded
and a `ground_support_missing_mask` warning is recorded. A usable declaration on
one frame provides no support on an unlisted frame. Source timestamps should
be audited against the frame registry before preparing declarations.

## Sampling and uncertainty

`align_world(..., source_video_sha256=known_sha)` accepts role input via config.
`build_evidence` also cross-checks known frame/video hashes; the production and
legacy-rebuild entrypoints pass their existing source identity. Loading an old
lossy cache does not restore its precision or invalidate the fingerprint check.

The policy builds the same union of all observed masks for object points O.
Only static candidate exclusion changes: a support role exempts its own mask,
while other-object exclusion wins overlaps, regardless of Track order. Existing
near-depth, confidence and dynamic-mask filters still apply. Actual sampling
stride/cap, adaptive RANSAC threshold, eligibility, score, normal orientation,
scale derivation and RNG algorithm are unchanged. Do not copy a numeric threshold
from a frozen diagnostic experiment into production configuration.

When opt-in mode has no observed support surviving the filters on its sampled
frames, alignment returns `ground: not_found`, identity T and scale 1, with a
`ground_sampling_no_support` warning. It does not select a background-only plane
as if support had been observed. Otherwise the existing candidate selection is
used; several eligible planes remain ambiguous in existing quality logic.

`meta.alignment.ground_sampling` and the optional replay alignment record retain
declarations/source references, per-frame exclusion/union/support counts and
structured warnings. Counts labeled `before_cap` do not claim final retained
RANSAC support. `physical_ground_correctness` remains `unknown` even if a plane
is selected. Schema/Program validators and their tolerances are not relaxed.

This first change is scoped to candidate sampling. The separate legacy ground
extent hint estimator and contact-center/smoothing operations remain unchanged;
they are not certified by these sampling roles. End-to-end Evidence geometry
and ground coverage must still be checked on new outputs before claiming a
production geometry fix. In opt-in mode, selecting a different T/scale can
legitimately change newly computed world poses; original masks/arrays and
historical outputs are not edited, and individual objects are not auto-snapped.

## Tests and next validation

Synthetic CPU tests cover default compatibility, a detected surface excluded by
legacy policy, explicit frame scope, overlap priority, unknown roles, source and
mask mismatches, missing observations, filtered/empty support, original O and
input immutability, the production adaptive threshold and replay round trips.
No synthetic result is an independent physical-ground accuracy measurement.

After local regression, separately authorize cache validation with the production
adaptive threshold. A previous frozen-threshold intervention is not that test.
Retain the baseline, label role provenance and any manual intervention; never
select declarations or a plane by whether Program penetration errors disappear.
No model rerun, API, GT input, rendering or automatic historical Evidence update
is required or authorized by this implementation.
