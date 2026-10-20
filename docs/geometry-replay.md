# Geometry replay records (Issue #3)

This change adds diagnostic recording, not a numerical geometry repair. Run 8902
remains a geometry-gate failure (11 errors at six poses). Run 9038 established a
conflict between cached object points and an approximately recovered historical
plane before OBB fitting/smoothing; it did not establish which observation was
wrong. The 9099b5b label correction validated by 9039 is unchanged.

## Recording

Set `perception.geometry_replay: true` in the effective config for a separately
authorized future run. The default is false. This option does not invoke a model,
change plane ranking, add a geometry correction, change tolerances, or bypass a
gate. No new real-video run is part of this implementation.

`build_evidence` writes `geometry_replay/inputs.npz` and `manifest.json` beside
the ordinary Evidence. Existing sidecars are never overwritten; use a fresh
output directory. A write failure fails the recording operation rather than
claiming a complete record. Legacy cache writers/loaders remain unchanged.
The semantic directory name is `geometry_replay`; the format version stays in
the manifest. Older `geometry_replay_v1` archives remain readable when their
directory is passed explicitly, and neither directory may be overwritten.
When recording is disabled, the original three-argument alignment call is kept,
including compatibility with existing test doubles and integrations.

The record contains:

- All geometry frames' original in-memory depth, confidence, dynamic masks,
  intrinsics and camera transforms, preserving dtype and shape; original object
  masks, frame IDs/times and track IDs. NPZ compression is lossless.
- Static sample pixel addresses `(frame, row, column)`, including the actual
  random subsample ordering; RANSAC RNG state, distance threshold, candidate
  planes and support indices into that ordered sample list.
- Candidate eligibility, ranking scores, normal orientation used for selection,
  selected index (or null), exact transform and scale, and assumed-height basis.
  Existing rounded Evidence diagnostics stay unchanged.
- Every attempted object/frame fit's raw OBB (or null), point count, contact
  replacement branch and candidate, and OBB before temporal smoothing.
- Redacted effective config, relevant computation-source hashes, input archive
  hash and Python/NumPy/SciPy versions. This identifies the recorded computation,
  not model weight provenance or independent physical ground truth.
- `source_video_sha256`, cross-checked against known per-frame video hashes.
  Available decoded source PTS, source frame indices and image hashes are retained
  without inventing missing values or persisting personal source-file paths.

Extra storage is proportional to the original arrays, rather than the old
float16 depth and width-480 masks. No fixed size estimate is claimed. Records
are ignored by Git; retain them with the corresponding run artifacts.

## CPU inspection and compatibility

`gwm.perception.geometry_replay.load_replay(path)` checks the version and input
archive hash, then returns `(geometry, tracks, manifest)`. It does no fitting or
inference and writes nothing. A missing record or unsupported version raises an
explicit error; it never silently falls back to lossy caches.

For source-bound use, call `load_replay(path, expected_video_sha256=video_hash)`.
A mismatch or unknown recorded source is rejected; conflicting frame and archive
video hashes are rejected even without an expected hash. For backward-compatible
inspection only, omitting the expected hash permits old archives, with missing
video identity explicitly returned as `unknown` (not verified). Production passes
the actual source-file SHA; legacy rebuilds reuse existing recorded provenance,
and cannot recover missing source PTS or prove physical ground correctness.
Hash consistency is not a cryptographic signature or proof against deliberate
metadata tampering. The video hash identifies the encoded file, not its semantic
contents or the correctness of independently supplied timestamps.

For an explicitly authorized CPU replay, pass the returned inputs and manifest
config to `align_world(..., replay={})`, using the recorded RNG state if the
original caller supplied a non-default RNG. Compare candidate planes/support,
selection, T and scale to `manifest['alignment']`. Reproject each object mask
with stride 2 and fit `obb_from_points` before replacement/smoothing. Match
source hashes and runtime first; across runtimes use a declared numeric tolerance,
not bitwise equivalence. Tests use 1e-12 for the synthetic plane residual.

`run_perception` labels records `production_in_memory`; `rebuild_evidence` labels
them `legacy_lossy_cache` with `historical_exactness: unavailable`. Direct callers
default to `in_memory_unspecified`. Even a production record only captures its
inputs exactly: `physical_ground_correctness` remains `unknown` in every case.
It does not certify metric scale or a real ground plane. Old runs do not acquire
missing precision by being rebuilt. Evidence schema and downstream validation
semantics are unchanged.

## Acceptance and stopping point

CPU synthetic tests verify lossless round trips, original support pixel mapping,
subsampling, exact plane-selection replay, raw OBB replay, failed alignment,
legacy/missing/corrupt record handling, non-overwrite behavior and unchanged
Evidence with recording on/off. Rendering is stubbed; no model, API or GPU runs.

This recording work is complete when these tests and relevant regressions pass
and the commit is pushed. It is not completion of the 8902 geometry repair.
Historical exact inputs cannot be recovered from the current cache. At minimum,
a future authorized observation must capture the original depth/confidence and
mask arrays plus this alignment/OBB record in the same execution. Do not choose
a new ground by which one makes objects pass validation or by ground truth.
If exact replay still cannot distinguish ground ambiguity from object-depth or
mask errors, stop at an unresolved evidence conflict and identify the independent
ground constraint needed. A completion date for that numerical repair cannot be
promised until such evidence supports a specific fix and acceptance test.
