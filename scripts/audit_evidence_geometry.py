"""Read cached Evidence and optional Program; write a new read-only audit.

python -m scripts.audit_evidence_geometry --evidence evidence.json --out new_audit.json
Pass --legacy-contact-dist only from the run's actual configuration.
This tool needs no video, model, GPU, API, or ground-truth Program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gwm.perception.geometry_audit import audit_evidence_geometry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--program-pointer", default="", help="Explicit JSON pointer to Program inside a validation record")
    parser.add_argument("--legacy-contact-dist", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; refusing to overwrite")
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode("utf-8-sig"))

    evidence = read(args.evidence)
    program = read(args.program) if args.program else None
    if args.program_pointer:
        if program is None or not args.program_pointer.startswith("/"):
            parser.error("--program-pointer requires --program and a JSON pointer beginning with /")
        for token in args.program_pointer[1:].split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            program = program[int(key)] if isinstance(program, list) else program[key]
    report = audit_evidence_geometry(evidence, program, legacy_contact_dist=args.legacy_contact_dist)
    report["input_sha256"] = inputs
    report["program_pointer"] = args.program_pointer
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(content)
    # This indicates the audit ran, not that the Evidence passed a quality gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
