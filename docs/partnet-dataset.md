# PartNet imported-asset dataset entry

This additive entry consumes a previously verified private output of
`scripts/partnet/import_asset.py`. It reuses the existing flat fixed-root
body/lid revolute subset. It does not implement the administrator's hierarchical
DSL, language template or training work, and does not change generic SceneSpec
generation or its asset fallback rules.

```sh
python -m scripts.partnet.generate_dataset PRIVATE_IMPORTED NEW_PRIVATE_DATASET --style-seeds 1 2
```

The input must contain program.json, asset_manifest.json, source_mapping.json,
body.glb and lid.glb. Geometry and source joint semantics must already have been
verified by the importer; this entry does not re-read the original URDF or prove
an arbitrary edited import is faithful to it. It records source input hashes.

`generate_partnet_scene(source, clip=..., style_seed=...)` deep-copies the known
Program, changes the clip ID, removes source notes, samples the existing style
subtree and opaque PBR colors/roughness. IDs, geometry, poses, camera and motion
are retained. Identical seeds reproduce identical output. Source identity and
seeds remain in manifest.json, not new Program supervision notes.

The CLI checks two explicit asset refs and licence fields, copies unchanged GLB
bytes into each sample, rewrites only manifest file locations and records hashes.
It does not call the CC0 catalog, fetch assets, repair geometry or fall back to
primitives. A pre-existing output directory is rejected. Missing files, mismatched
refs and unsupported moving-parent/joint inputs fail explicitly.

The manifest records base-to-style-variant pairs. No behavior/camera/layout
sampling, semantic open/closed labels, source textures, language or new joint
fields are generated. Mechanical limits retain their source meaning.

Program validation is not rendered acceptance. This step produces data files
only; each new sample still needs production compilation, actual mesh-load
checks, endpoint visibility and rendering before being used in an experiment.
It is one source asset, not a train/test asset split or a complete benchmark.

All source and derived dataset/GLB outputs must stay private and Git-ignored
unless redistribution is separately permitted. The source licence is retained;
copying it is not a new legal approval or a complete licence audit.
