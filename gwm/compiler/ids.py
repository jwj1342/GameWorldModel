"""Object ids, ID-pass colours and registry order: the contract shared by the kernel (scene.js) and the Python side."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

def id_to_color(i: int) -> tuple[int, int, int]:
    """Same mapping as kernel/scene.js idToColor: registry index -> 24-bit colour."""
    v = ((i * 2654435) % 0xFFFFFF) or 1
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)

def registry_order(program: dict) -> list[dict]:
    """Replicates kernel/scene.js: statics first, then objects with instances expanded; idIndex starts at 1."""
    reg, idx = [], 1
    for s in program.get("static", []): reg.append({"id": s["id"], "kind": "static", "idx": idx, "class": s.get("class", "static"), "instance": 0}); idx += 1
    for o in program.get("objects", []):
        n = len(o["instances"]) if o.get("instances") else 1
        for k in range(n): reg.append({"id": o["id"], "kind": "object", "idx": idx, "class": o.get("class", ""), "instance": k}); idx += 1
    return reg

def packed_ids(path: str | Path) -> np.ndarray:
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
    return (im[..., 0] << 16) | (im[..., 1] << 8) | im[..., 2]

def decode_id_png(path: str | Path, reg: list[dict]) -> dict[str, np.ndarray]:
    """Per program id (instances merged) boolean masks from an ID pass image."""
    packed = packed_ids(path); masks: dict[str, np.ndarray] = {}
    for r in reg:
        c = id_to_color(r["idx"]); m = packed == ((c[0] << 16) | (c[1] << 8) | c[2])
        masks[r["id"]] = masks.get(r["id"], np.zeros(m.shape, bool)) | m
    return masks

def decode_id_png_by_entry(path: str | Path, reg: list[dict]) -> dict[int, np.ndarray]:
    """Per registry entry (instance-level) boolean masks, keyed by idx."""
    packed = packed_ids(path); out = {}
    for r in reg:
        c = id_to_color(r["idx"]); out[r["idx"]] = packed == ((c[0] << 16) | (c[1] << 8) | c[2])
    return out

def decode_depth_png(path: str | Path, far: float) -> np.ndarray:
    """Depth pass: linear view depth encoded as R*256+G over 65535, scaled by far."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    return (im[..., 0] * 256 + im[..., 1]) / 65535.0 * far
