#!/usr/bin/env python3
"""Create a pending source association from explicit files; no model or decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwm.feedback.ground_truth_source import build_pending_association


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--program', required=True, help='Independent reference, never a current prediction')
    ap.add_argument('--record', required=True, help='Existing generation/reference provenance record')
    ap.add_argument('--out', required=True, help='Fresh manifest path above all source files')
    ap.add_argument('--association-id', required=True)
    ap.add_argument('--duration', required=True, type=float)
    ap.add_argument('--kind', required=True, choices=['synthetic_generation', 'manual_reference'])
    ap.add_argument('--review-note', required=True, help='Why this proposed association requires human review')
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    try:
        if out.exists():
            raise ValueError('Refusing to overwrite an existing manifest')
        data = build_pending_association(args.video, args.program, args.record, out,
            association_id=args.association_id, duration_s=args.duration,
            provenance_kind=args.kind, review_note=args.review_note)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write('\n')
    except (OSError, ValueError, TypeError) as error:
        ap.error(str(error))
    print(f'Pending association: {out}; human source/time review required before confirmation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
