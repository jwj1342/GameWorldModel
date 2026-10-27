# Ground proximity is not physical contact

## Producer and contract

The previous producer multiplied `contact_dist_m` by object height and, after a
majority vote, emitted one `contacts` interval over the entire track. The vote
fraction was stored as `conf`. Neither the enlarged distance nor a majority of
sampled frames establishes continuous physical contact or a measured probability.

`measure_ground_proximity()` now records only OBB/plane compatibility in the
optional `object.ground_proximity` field. The production caller passes the source
frame registry, so a missing OBB is an unknown sample, not an interpolated contact.
Even two compatible endpoints 100 seconds apart remain two points. No intervals
are inferred, so no maximum-gap heuristic can silently bridge an unobserved gap.

Each sample has `frame_index`, `t`, `gap`, and `state` (`compatible`, `separated`,
`penetrating`, or `unknown`). The gap uses the oriented box's support radius along
the normalized plane normal, including its offset. A missing or ambiguous plane,
low/unknown plane confidence, failed alignment or invalid OBB yields unknown.
Alignment now records how many planes passed its existing ground-candidate
filter. More than one plausible candidate also yields unknown proximity; this
metadata addition does not change the selected plane, transform or geometry.
`confidence` and `physical_contact` are always `unknown`. New automatic records
leave physical `contacts` empty, rather than pretending geometric proximity is
physical contact. No object geometry, pose, motion or identity is modified.

The Schema version stays unchanged. The new field is optional and explicitly
typed; the semantic validator checks unique source frame/time/OBB references,
finite gaps, states, source geometry and scale consistency. Legacy Evidence is
still adapted/read without inventing the new field or rewriting old contacts.
Historical contacts retain their legacy consumer behavior for compatibility;
this change does not retroactively certify or correct their meaning.

## Configuration and units

```yaml
perception:
  ground_proximity:
    max_gap_world: 0.05
    min_ground_confidence: 0.80
```

`max_gap_world` is a fixed tolerance in the aligned-world coordinate units, not a
multiple of object height. A `relative` geometry backend remains relative even
after calibration using an assumed camera height: this tolerance is not proof of
measured metre accuracy. `metric` is recorded only when the input declares it.
No video, category or object-specific rules are used.

When the new tolerance is absent, `contact_dist_m` is accepted as a legacy alias,
capped at the conservative default 0.05; a smaller legacy value is preserved.
An explicitly supplied `ground_proximity.max_gap_world` takes precedence. A loaded
new default config contains that new setting, so old overrides should migrate to
the new key. No height multiplier is retained.

## Consumers

- Quality reports expose separate sample-state counts; unavailable observations
  and below-plane geometry generate warnings, not fabricated successes or an
  automatic fix. Structural/reference errors still block through the contract.
- Writer summaries show counts and a bounded list of sampled states, explicitly
  state that physical contact is unknown, and forbid inferring support/moving
  geometry from this field. This is a prompt constraint, not a guarantee about
  an untested model response.
- Direct translation does not manufacture `support` from the new field or from
  records explicitly marked `kind: geometric_proximity`. Empty contacts do not
  mean that an object must be floating, and cause no position edit.
- Conservative Repair still requires its original support/contact/metric/quality
  conditions; an explicitly marked geometric-proximity record is ineligible even
  if a caller supplies a numeric `conf`. Missing physical contact is not evidence
  for snapping. Existing legacy Repair tests and behavior remain supported.

The separate `settle_on_ground` assembly prior in `rebuild_with_assets` can move
objects independently of contacts; it is NOT disabled or changed here. Likewise,
the existing contact-centre projection and OBB smoothing are unchanged. This patch
does not promise that the entire existing reconstruction pipeline makes no edits.
It only makes the new contact evidence conservative and avoids promoting it into
an automatic support correction.

## Limits

This label fix does not solve incorrect ground, depth, masks or geometry. It does
not repair penetrations or turn historical run 8902 into a success. Historical
Evidence and failed-run records remain unchanged. Real-cache and model behavior
require a separate server validation after local tests; no such result is claimed.
