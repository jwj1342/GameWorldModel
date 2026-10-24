# Read-only Evidence geometry audit

This audit locates observable inconsistencies without fixing Evidence or relaxing
Program validation. It is not a replacement quality gate or a reconstruction run.

```bash
python -m scripts.audit_evidence_geometry --evidence evidence.json --out new_audit.json
```

Optional `--program` reads an existing Candidate. If the Candidate is inside a
validation record, first inspect its keys, then supply the exact JSON pointer with
`--program-pointer`; the tool does not guess which nested object is the Program.
Pass `--legacy-contact-dist` only from that run's actual configuration. No default
historical value is assumed. All computation on SOAR must run under Slurm.

The module entry point `audit_evidence_geometry(evidence, program=None, ...)`
returns:

- Per-OBB signed gap to one declared ground plane, including offset and box
  rotation. Missing/ambiguous ground, invalid geometry or quaternion is unknown,
  not a made-up plane, zero gap or repaired observation.
- Contact conflicts only inside the declared contact interval. Historical
  height-scaled contact votes can explain a permissive label, but are not measured
  physical contact confidence. The default audit tolerance reuses the compiler's
  contact tolerance; it does not modify it.
- Original Candidate geometry findings, with unique affected pose paths so ground
  and support findings are not counted as independent penetrations.
- Exact-ID comparison of Candidate pose/extent against the first OBB, allowing
  three-decimal summary rounding and quaternion sign equivalence. Missing fields
  remain unknown; changed geometry is not automatically a Writer error.
- An untouched copy of recorded alignment metadata; absent transformation
  provenance stays absent. The CLI hashes only its actual JSON inputs and refuses
  to overwrite an existing output.

`status=measured` means the gaps were measurable, not that the Evidence passed.
The CLI's zero exit status likewise means the audit executed, not that geometry
was accepted. `root_cause` remains `unresolved`: a contradiction does not tell us
whether the ground, object geometry or transform is incorrect. Raw cached depth,
masks and camera/alignment transforms are needed for that next isolation step.

Gaps are in the Evidence coordinate system. Relative scale is not independently
measured metres; plane penetration is not finite-solid intersection depth.
This tool does not use a video, GT Program, model, API, browser, or GPU, does not
snap objects to ground, and is not connected to the production pipeline by default.
