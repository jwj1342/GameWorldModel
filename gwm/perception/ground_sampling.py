"""Source-bound, frame-local mask roles for candidate sampling, not ground truth."""
from __future__ import annotations

import copy
import hashlib
import re

import numpy as np


def uses_sampling_roles(config):
    settings = config.get("perception", {}).get("ground_sampling", {})
    if not isinstance(settings, dict):
        raise ValueError("ground_sampling must be an object")
    mode = settings.get("mode", "legacy")
    if mode not in ("legacy", "reviewed_roles"):
        raise ValueError("ground_sampling.mode must be legacy or reviewed_roles")
    return mode == "reviewed_roles"


def mask_sha256(mask):
    """Fingerprint the native mask, including dtype and shape (before resizing)."""
    array = np.ascontiguousarray(mask)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}:".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-fA-F0-9]{64}", value) is None:
        raise ValueError("sampling role requires a known SHA256")
    return value.lower()


class GroundSamplingPolicy:
    """Validate declarations once, then build exclusion and union masks per frame.

    Unassigned/unknown/exclude masks stay excluded; source-bound ground_support
    only exempts its own mask. Exclusion wins overlaps. The object union is never
    changed. Frames are working indices, not an interpolated time interval.
    """

    def __init__(self, tracks, frame_indices, config, *, source_video_sha256=None):
        self.enabled = uses_sampling_roles(config)
        self.declarations = {}
        self.audit = {"mode": "reviewed_roles", "physical_ground_correctness": "unknown",
                      "declarations": [], "frames": [], "diagnostics": []}
        if not self.enabled:
            return
        self.audit["source_video_sha256"] = _sha(source_video_sha256)
        objects = {o.id: o for o in tracks.objects}
        if len(objects) != len(tracks.objects):
            raise ValueError("sampling roles require unique object IDs")
        records = config["perception"]["ground_sampling"].get("roles", [])
        if not isinstance(records, list):
            raise ValueError("ground_sampling.roles must be a list")
        available = set(frame_indices)
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("sampling role declaration must be an object")
            oid, role, frames = record.get("object_id"), record.get("role"), record.get("frame_indices")
            if not isinstance(oid, str) or oid not in objects:
                raise ValueError("sampling role references an unknown object")
            if role not in ("ground_support", "exclude", "unknown"):
                raise ValueError("invalid sampling role")
            if (not isinstance(frames, list) or not frames or
                    any(isinstance(fi, bool) or not isinstance(fi, int) or fi not in available for fi in frames) or
                    len(set(frames)) != len(frames)):
                raise ValueError("sampling role requires unique existing working frame indices")
            if role == "ground_support":
                source = record.get("source", {})
                if (not isinstance(source, dict) or source.get("kind") not in ("human_review", "assistant_review", "provider") or
                        not isinstance(source.get("reference"), str) or not source["reference"].strip()):
                    raise ValueError("ground_support requires explicit review/provider provenance")
                if _sha(source.get("video_sha256")) != self.audit["source_video_sha256"]:
                    raise ValueError("sampling role source video mismatch")
                hashes = record.get("mask_sha256")
                if not isinstance(hashes, dict) or set(hashes) != {str(fi) for fi in frames}:
                    raise ValueError("ground_support requires native mask hashes for its exact frames")
                for fi in frames:
                    expected = _sha(hashes[str(fi)])
                    mask = objects[oid].masks.get(fi)
                    if mask is not None and mask_sha256(mask) != expected:
                        raise ValueError("sampling role mask hash mismatch")
            for fi in frames:
                key = (oid, fi)
                if key in self.declarations:
                    raise ValueError("duplicate or conflicting sampling role declaration")
                self.declarations[key] = role
                if fi not in objects[oid].masks:
                    self.warn("ground_support_missing_mask", f"/objects/{oid}/masks/{fi}",
                              "No mask observation; do not interpolate or promote this frame.")
            self.audit["declarations"].append(copy.deepcopy(record))

    def warn(self, code, path, hint):
        self.audit["diagnostics"].append({"stage": "perception", "severity": "warning",
                                          "code": code, "path": path, "hint": hint})

    def masks(self, tracks, fi, size, resize):
        width, height = size
        union = np.zeros((height, width), bool)
        excluded = np.zeros_like(union)
        support = np.zeros_like(union)
        for obj in tracks.objects:
            raw = obj.masks.get(fi)
            if raw is None:
                continue
            mask = resize(raw, size)
            union |= mask
            if self.declarations.get((obj.id, fi)) == "ground_support":
                support |= mask
            else:
                excluded |= mask
        return union, excluded, support

    def record_frame(self, fi, union, excluded, support, static_mask):
        if self.enabled:
            eligible = static_mask & support
            self.audit["frames"].append({"frame_index": int(fi),
                "object_union_pixels": int(union.sum()), "excluded_pixels": int(excluded.sum()),
                "declared_support_pixels": int(support.sum()),
                "eligible_support_pixels": int(eligible.sum()),
                "eligible_support_stride4_before_cap": int(eligible[::4, ::4].sum()),
                "candidate_pixels_stride4_before_cap": int(static_mask[::4, ::4].sum())})
