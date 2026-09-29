"""Content fingerprints used to bind Evidence and requests to a source video."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def declared_video_sha256(evidence: dict) -> str | None:
    """Read either Evidence source-hash layout, rejecting contradictory declarations."""
    meta = evidence.get("meta") or {}
    legacy = meta.get("source_video_sha256")
    nested = (meta.get("source_video") or {}).get("sha256")
    values = [value for value in (legacy, nested) if value is not None]
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in values):
        raise ValueError("Evidence source video SHA-256 is invalid")
    if len(set(values)) > 1:
        raise ValueError("Evidence source video SHA-256 declarations conflict")
    return values[0] if values else None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_sha256(evidence: dict) -> str:
    """Hash normalized Evidence content, independent of JSON whitespace and key order."""
    content = json.dumps(evidence, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
