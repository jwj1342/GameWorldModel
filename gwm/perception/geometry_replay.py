"""Optional lossless diagnostic sidecar; never supplies geometry decisions."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from ..config import redact
from .base import Geometry, Tracks, TrackedObject


VERSION = 1


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_replay(out, geom, tracks, frames, cfg, alignment, objects, source):
    """Write once into a dedicated directory, without replacing prior evidence.

    Preserve each array's actual dtype/shape (including optional dynamic masks).
    Numeric keys avoid interpreting object IDs as archive member paths.
    """
    out = Path(out)
    out.mkdir(exist_ok=False)
    arrays = {}
    for name in ("intrinsics", "cam_to_world", "depth", "depth_conf", "dynamic_mask"):
        values = getattr(geom, name)
        if values is not None:
            for i, value in enumerate(values):
                arrays[f"{name}_{i}"] = np.asarray(value)
    registry = []
    for i, obj in enumerate(tracks.objects):
        masks = []
        for j, (fi, mask) in enumerate(sorted(obj.masks.items())):
            key = f"mask_{i}_{j}"
            arrays[key] = np.asarray(mask)
            masks.append({"frame": int(fi), "array": key})
        registry.append({"id": obj.id, "phrase": obj.phrase, "score": float(obj.score),
                         "masks": masks})
    np.savez_compressed(out / "inputs.npz", **arrays)
    # Hash the whole computation chain used by alignment and OBB extraction.
    sources = [Path(__file__), *[Path(__file__).with_name(n) for n in
               ("evidence.py", "base.py", "masks.py", "run.py")],
               Path(__file__).parents[1] / "config.py"]
    manifest = {
        "version": VERSION, "input_source": source,
        "historical_exactness": "unavailable" if source == "legacy_lossy_cache" else "recorded_inputs_only",
        "physical_ground_correctness": "unknown",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__},
        "frame_indices": [int(fi) for fi in geom.frame_indices],
        "frames": [{"index": int(f["index"]), "t": float(f["t"])} for f in frames],
        "geometry": {"scale": geom.scale, "backend": geom.backend,
                     "has_conf": geom.depth_conf is not None,
                     "has_dynamic_mask": geom.dynamic_mask is not None},
        "tracks": {"frame_size": list(tracks.frame_size), "backend": tracks.backend,
                   "objects": registry},
        "config": redact(cfg), "source_sha256": {p.name: _sha(p) for p in sources},
        "inputs_sha256": _sha(out / "inputs.npz"),
        "alignment": alignment, "objects": objects,
    }
    def numpy_json(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(f"unsupported replay value: {type(value).__name__}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=numpy_json))


def load_replay(out):
    """Return inputs for CPU inspection; missing/legacy records are explicit.

    Does not invoke fitting, inference, evidence generation or a validator.
    """
    out = Path(out)
    if not (out / "manifest.json").exists():
        raise ValueError("geometry replay unavailable: historical record missing")
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported geometry replay version")
    if _sha(out / "inputs.npz") != manifest["inputs_sha256"]:
        raise ValueError("geometry replay inputs hash mismatch")
    count = len(manifest["frame_indices"])
    with np.load(out / "inputs.npz", allow_pickle=False) as z:
        def series(name):
            return [z[f"{name}_{i}"].copy() for i in range(count)]
        g = manifest["geometry"]
        geom = Geometry(manifest["frame_indices"], series("intrinsics"), series("cam_to_world"),
                        series("depth"), series("depth_conf") if g["has_conf"] else None,
                        series("dynamic_mask") if g["has_dynamic_mask"] else None,
                        scale=g["scale"], backend=g["backend"])
        tr = manifest["tracks"]
        tracks = Tracks([TrackedObject(o["id"], o["phrase"], o["score"],
                         {m["frame"]: z[m["array"]].copy() for m in o["masks"]})
                         for o in tr["objects"]], tuple(tr["frame_size"]), tr["backend"])
    return geom, tracks, manifest
