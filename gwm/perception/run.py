"""Perception stage driver: frames -> working set -> keyframes -> phrases -> geometry (backend chain) -> segmentation -> evidence.json + masks.npz"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from PIL import Image
from .frames import extract_frames, pick_uniform, select_keyframes, motion_profile, resize_copy
from .evidence import build_evidence
from ..errors import ErrorLog

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PHRASES = ["toy train", "train track", "cardboard box", "conveyor belt", "ball", "toy car", "door", "table", "cup", "bottle", "chair", "robot"]

def prepare_geometry_frames(work: list[dict], out_dir: Path, width: int) -> list[str]:
    """Resize working frames to width x (multiple of 14) so VGGT does not crop; masks are resized to the same size later."""
    out_dir.mkdir(parents=True, exist_ok=True); files = []
    for f in work:
        with Image.open(f["file"]) as im:
            h = int(round(im.height * width / im.width / 14) * 14); h = max(14, min(h, 518))
            p = out_dir / f"g_{f['index']:05d}.jpg"; im.convert("RGB").resize((width, h), Image.LANCZOS).save(p, quality=92)
        files.append(str(p))
    return files

def name_phrases(client, keyframe_file: str, log_dir: Path) -> list[str]:
    schema = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "maxItems": 10, "items": {"type": "object", "required": ["phrase", "size_m", "moving"], "properties": {"phrase": {"type": "string"}, "size_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "moving": {"type": "boolean"}}}}}}
    r = client.chat((REPO / "prompts" / "namer.md").read_text(), "Frame attached.", images=[keyframe_file], json_schema=schema, temperature=0.2)
    (log_dir / "namer.json").write_text(json.dumps(r["json"], indent=1, ensure_ascii=False))
    out = r["json"]
    items = out.get("objects") if isinstance(out, dict) else out   # tolerate a bare list or different key names
    phrases = []
    for o in items or []:
        if isinstance(o, str): phrases.append(o.strip().lower()); continue
        if not isinstance(o, dict): continue
        name = o.get("phrase") or o.get("object") or o.get("name") or o.get("label")
        if name: phrases.append(str(name).strip().lower())
    return phrases[:10] or DEFAULT_PHRASES

def run_perception(video: str, clip: str, out_dir: Path, cfg: dict, phrases: list[str] | None = None, client=None, log: ErrorLog | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True); log = log or ErrorLog(out_dir / "errors.jsonl")
    fc, pc = cfg["frames"], cfg["perception"]; t0 = time.time(); timing = {}
    frames = extract_frames(video, out_dir / "frames", fc["sample_fps"], fc["max_seconds"], max_side=960)
    work = pick_uniform(frames, fc["geometry_frames"]); work_idx = [f["index"] for f in work]
    motion = motion_profile(work)
    keyframes = select_keyframes(work, fc["keyframes"], motion)
    kf_dir = out_dir / "keyframes"; kf_files = []
    for i, k in enumerate(keyframes):
        dst = kf_dir / f"kf_{i:02d}_t{k['t']:.2f}.jpg"; resize_copy(k["file"], dst, fc["keyframe_max_side"]); kf_files.append(str(dst)); k["file_small"] = str(dst)
    (out_dir / "keyframes.json").write_text(json.dumps(keyframes, indent=1))
    timing["frames"] = round(time.time() - t0, 1)
    # phrases
    phrases_source = "cli"
    if not phrases:
        if client is not None:
            try: phrases = name_phrases(client, kf_files[0], out_dir); phrases_source = "vlm_namer"
            except Exception as e: log.record("perception", "namer_failed", repr(e), action_taken="default phrases"); phrases = DEFAULT_PHRASES; phrases_source = "default"
        else: phrases = DEFAULT_PHRASES; phrases_source = "default"
    (out_dir / "phrases.json").write_text(json.dumps({"phrases": phrases, "source": phrases_source}))
    # geometry backend chain
    gfiles = prepare_geometry_frames(work, out_dir / "geom_frames", fc["geometry_width"])
    fallbacks, geom = [], None
    for name in [pc["geometry_backend"], *pc["geometry_fallbacks"]]:
        t1 = time.time()
        try:
            geom = make_geometry_backend(name, cfg).estimate(gfiles, work_idx, cfg); timing[f"geometry_{name}"] = round(time.time() - t1, 1); break
        except Exception as e:
            log.record("perception", f"geometry_{name}_failed", repr(e), action_taken="next backend", exc=e); fallbacks.append(f"geometry:{name}")
    if geom is None: raise RuntimeError("all geometry backends failed")
    save_geometry(geom, out_dir / "geometry.npz")
    # segmentation backend chain
    tracks = None
    for name in [pc["segmentation_backend"], *pc["segmentation_fallbacks"]]:
        t1 = time.time()
        try:
            tracks = make_segmentation_backend(name, cfg).segment([f["file"] for f in work], work_idx, phrases, cfg); timing[f"segmentation_{name}"] = round(time.time() - t1, 1); break
        except Exception as e:
            log.record("perception", f"segmentation_{name}_failed", repr(e), action_taken="next backend", exc=e); fallbacks.append(f"segmentation:{name}")
    if tracks is None: raise RuntimeError("all segmentation backends failed")
    # save masks (frame-resolution, downscaled to width 480 to keep the file small)
    masks_index = {}
    with_arrays = {}
    for o in tracks.objects:
        for fi, m in o.masks.items():
            small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((480, int(round(480 * m.shape[0] / m.shape[1]))), Image.NEAREST)) > 127
            with_arrays[f"{o.id}__{fi}"] = small; masks_index.setdefault(o.id, []).append(fi)
    np.savez_compressed(out_dir / "masks.npz", **with_arrays)
    (out_dir / "masks_index.json").write_text(json.dumps({"objects": {o.id: {"phrase": o.phrase, "score": o.score, "frames": sorted(masks_index.get(o.id, [])),
                                                                                  "identity_hypotheses": o.identity_hypotheses,
                                                                                  "source_detection_ids": o.source_detection_ids} for o in tracks.objects},
                                                          "frame_size": tracks.frame_size, "backend": tracks.backend,
                                                          "association_diagnostics": tracks.association_diagnostics}))
    t1 = time.time()
    ev = build_evidence(clip, frames, geom, tracks, cfg, out_dir, keyframes, fallbacks, phrases_source)
    timing["evidence"] = round(time.time() - t1, 1); timing["total"] = round(time.time() - t0, 1)
    ev["meta"]["timing_s"] = timing; (out_dir / "evidence.json").write_text(json.dumps(ev, indent=1))
    return ev

def save_geometry(geom, path: Path) -> None:
    np.savez_compressed(path, frame_indices=np.asarray(geom.frame_indices), intrinsics=np.stack(geom.intrinsics), cam_to_world=np.stack(geom.cam_to_world), depth=np.stack(geom.depth).astype(np.float16),
                        depth_conf=np.stack(geom.depth_conf).astype(np.float16) if geom.depth_conf is not None else np.zeros(1), has_conf=np.asarray(geom.depth_conf is not None), scale=np.asarray(geom.scale), backend=np.asarray(geom.backend), notes=np.asarray(geom.notes))

def load_geometry(path: Path):
    from .base import Geometry
    z = np.load(path)
    return Geometry(frame_indices=[int(i) for i in z["frame_indices"]], intrinsics=[k for k in z["intrinsics"]], cam_to_world=[m for m in z["cam_to_world"]], depth=[d.astype(np.float32) for d in z["depth"]],
                    depth_conf=[c.astype(np.float32) for c in z["depth_conf"]] if bool(z["has_conf"]) else None, scale=str(z["scale"]), backend=str(z["backend"]), notes=str(z["notes"]))

def rebuild_evidence(out_dir: Path, cfg: dict) -> dict:
    """Recompute evidence.json from cached geometry.npz + masks.npz (CPU only)."""
    from .base import Tracks, TrackedObject
    out_dir = Path(out_dir)
    frames = json.loads((out_dir / "frames" / "frames.json").read_text())["frames"]
    keyframes = json.loads((out_dir / "keyframes.json").read_text())
    geom = load_geometry(out_dir / "geometry.npz")
    mi = json.loads((out_dir / "masks_index.json").read_text()); masks = load_masks(out_dir)
    objs = [TrackedObject(id=oid, phrase=m["phrase"], score=m["score"], masks=masks.get(oid, {}),
                          identity_hypotheses=m.get("identity_hypotheses", []), source_detection_ids=m.get("source_detection_ids", []))
            for oid, m in mi["objects"].items()]
    tracks = Tracks(objects=objs, frame_size=tuple(mi["frame_size"]), backend=mi["backend"],
                    association_diagnostics=mi.get("association_diagnostics", []))
    old = json.loads((out_dir / "evidence.json").read_text()) if (out_dir / "evidence.json").exists() else None
    return build_evidence(old["meta"]["clip"] if old else out_dir.parent.name, frames, geom, tracks, cfg, out_dir, keyframes, old["meta"]["fallbacks"] if old else [], old["meta"].get("phrases_source", "") if old else "")

def load_masks(out_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    z = np.load(out_dir / "masks.npz"); res: dict[str, dict[int, np.ndarray]] = {}
    for k in z.files:
        oid, fi = k.rsplit("__", 1); res.setdefault(oid, {})[int(fi)] = z[k]
    return res

def make_geometry_backend(name: str, cfg: dict):
    w = Path(cfg["paths"]["weights"])
    if name == "vggt":
        from .vggt_backend import VGGTBackend; return VGGTBackend(w / cfg["weights"]["vggt"])
    if name == "static":
        from .static_backend import StaticDepthBackend; return StaticDepthBackend(w / cfg["weights"]["depth_anything"], cfg["frames"]["geometry_width"])
    if name == "gt":
        from .gt_backend import GTGeometryBackend; return GTGeometryBackend(cfg["perception"]["gt_program"], cfg["frames"]["geometry_width"])
    raise ValueError(f"unknown backend {name!r} (see configs/default.yaml perception.*_backend)")

def make_segmentation_backend(name: str, cfg: dict):
    w = Path(cfg["paths"]["weights"])
    if name == "grounded_sam2":
        from .grounded_sam2_backend import GroundedSAM2Backend; return GroundedSAM2Backend(w / cfg["weights"]["grounding_dino"], w / cfg["weights"]["sam2"])
    if name == "gt":
        from .gt_backend import GTSegmentationBackend; return GTSegmentationBackend(cfg["perception"]["gt_program"], cfg["frames"]["geometry_width"])
    raise ValueError(f"unknown backend {name!r} (see configs/default.yaml perception.*_backend)")

if __name__ == "__main__":
    import argparse
    from ..config import load_config
    ap = argparse.ArgumentParser(); ap.add_argument("--video"); ap.add_argument("--clip"); ap.add_argument("--out", required=True); ap.add_argument("--phrases", default=None, help="comma-separated noun phrases"); ap.add_argument("--rebuild", action="store_true", help="recompute evidence from cached geometry/masks in --out (CPU)")
    a = ap.parse_args(); cfg = load_config()
    if a.rebuild: ev = rebuild_evidence(Path(a.out), cfg)
    else: ev = run_perception(a.video, a.clip, Path(a.out), cfg, phrases=[p.strip() for p in a.phrases.split(",")] if a.phrases else None)
    print(json.dumps({"objects": [(o["id"], o["class_guess"], o["motion_guess"]["type"], len(o["obb"])) for o in ev["objects"]], "backend": ev["meta"]["geometry_backend"], "fallbacks": ev["meta"]["fallbacks"], "timing": ev["meta"]["timing_s"]}, indent=1))
