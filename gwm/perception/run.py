"""Perception stage driver: frames -> working set -> keyframes -> phrases -> geometry (backend chain) -> segmentation -> evidence.json + masks.npz"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from PIL import Image
from .frames import extract_frames, pick_uniform, select_keyframes, motion_profile, resize_copy
from .evidence import build_evidence
from .provenance import file_sha256
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
    work_times = [f["t"] for f in work]
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
            timing_args = {"frame_times": work_times} if name == "gt" else {}
            geom = make_geometry_backend(name, cfg).estimate(gfiles, work_idx, cfg, **timing_args); timing[f"geometry_{name}"] = round(time.time() - t1, 1); break
        except Exception as e:
            log.record("perception", f"geometry_{name}_failed", repr(e), action_taken="next backend", exc=e); fallbacks.append(f"geometry:{name}")
    if geom is None: raise RuntimeError("all geometry backends failed")
    # 消融档位：只把相机换成真值，深度和掩码仍走神经感知。
    #
    # 为什么需要它：物体的世界坐标 = 相机位姿 x 相机坐标下的观测。#7 下面十条排除全在
    # 相机**下游**，而实测重建相机的位移误差占它自己行程的 64%（尺度对齐之后），行程比
    # 本身在 0.13 到 6.9 倍之间乱跳。如果相机是主因，这一档的运动类型准确率应该明显
    # 高于全神经；如果不是，说明还有别的东西。纯替换，不实现任何新方法。
    gt_camera = pc.get("gt_camera_program")
    if gt_camera:
        from .gt_backend import GTGeometryBackend
        truth = GTGeometryBackend(gt_camera, fc["geometry_width"]).estimate(gfiles, work_idx, cfg, frame_times=work_times)
        if len(truth.cam_to_world) == len(geom.cam_to_world):
            geom.cam_to_world = truth.cam_to_world
            geom.intrinsics = truth.intrinsics
            geom.notes = (geom.notes or "") + f" | 相机位姿换成真值（{gt_camera}），深度保持神经"
        else:
            log.record("perception", "gt_camera_frame_mismatch",
                       f"{len(truth.cam_to_world)} != {len(geom.cam_to_world)}",
                       action_taken="保留重建的相机")
    save_geometry(geom, out_dir / "geometry.npz")
    # segmentation backend chain
    tracks = None
    for name in [pc["segmentation_backend"], *pc["segmentation_fallbacks"]]:
        t1 = time.time()
        try:
            timing_args = {"frame_times": work_times} if name == "gt" else {}
            tracks = make_segmentation_backend(name, cfg).segment([f["file"] for f in work], work_idx, phrases, cfg, **timing_args); timing[f"segmentation_{name}"] = round(time.time() - t1, 1); break
        except Exception as e:
            log.record("perception", f"segmentation_{name}_failed", repr(e), action_taken="next backend", exc=e); fallbacks.append(f"segmentation:{name}")
    if tracks is None: raise RuntimeError("all segmentation backends failed")
    # save masks (frame-resolution, downscaled to width 480 to keep the file small)
    save_visibility_references(tracks, out_dir)
    masks_index = {}
    with_arrays = {}
    for o in tracks.objects:
        for fi, m in o.masks.items():
            small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((480, int(round(480 * m.shape[0] / m.shape[1]))), Image.NEAREST)) > 127
            with_arrays[f"{o.id}__{fi}"] = small; masks_index.setdefault(o.id, []).append(fi)
    np.savez_compressed(out_dir / "masks.npz", **with_arrays)
    (out_dir / "masks_index.json").write_text(json.dumps({"objects": {o.id: {"phrase": o.phrase, "score": o.score, "frames": sorted(masks_index.get(o.id, [])),
                                                                                  "identity_hypotheses": o.identity_hypotheses,
                                                                                  "source_detection_ids": o.source_detection_ids,
                                                                                  "visibility_measurements": o.visibility_measurements} for o in tracks.objects},
                                                          "frame_size": tracks.frame_size, "backend": tracks.backend,
                                                          "association_diagnostics": tracks.association_diagnostics}))
    t1 = time.time()
    ev = build_evidence(clip, frames, geom, tracks, cfg, out_dir, keyframes, fallbacks, phrases_source)
    timing["evidence"] = round(time.time() - t1, 1); timing["total"] = round(time.time() - t0, 1)
    ev["meta"]["timing_s"] = timing
    ev["meta"]["source_video_sha256"] = file_sha256(video)
    ev["meta"]["source_video_path"] = str(Path(video).resolve())
    (out_dir / "evidence.json").write_text(json.dumps(ev, indent=1))
    return ev

def save_visibility_references(tracks, out_dir: Path) -> None:
    """Keep byte-exact PNG members in one NPZ; verify cached members before reuse.

    Legacy relative PNG references remain readable, but no new PNG tree is made.
    Hash the container and original PNG bytes independently; never trust an NPZ
    member merely because it has the expected name. No pickle is allowed.
    """
    import hashlib
    import io
    import os
    import tempfile
    import zipfile

    out_dir = Path(out_dir)
    root = out_dir.resolve()
    archive_path = out_dir / "visibility_refs.npz"
    archives = {}
    members, retained = {}, []
    write_archive = False

    def digest(payload):
        return hashlib.sha256(payload).hexdigest()

    def cached_bytes(record, key):
        reference = record.get("reference", "unknown")
        if not isinstance(reference, str):
            raise ValueError("invalid reference")
        relative, sep, member = reference.partition("#")
        path = (out_dir / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("reference outside cache")
        if not sep:  # Read old caches without deleting or rewriting their PNGs.
            return path.read_bytes()
        if relative != "visibility_refs.npz" or member != key:
            raise ValueError("reference member identity mismatch")
        if path not in archives:
            payload = path.read_bytes()
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                if len(data.files) != len(set(data.files)):
                    raise ValueError("duplicate archive members")
                archives[path] = (digest(payload), {name: data[name] for name in data.files})
        sha, data = archives[path]
        if sha != record.get("reference_archive_sha256"):
            raise ValueError("archive changed")
        array = data[member]
        if array.dtype != np.uint8 or array.ndim != 1:
            raise ValueError("invalid PNG member encoding")
        return array.tobytes()

    for obj in tracks.objects:
        for fi, record in obj.visibility_measurements.items():
            source = record.pop("reference_path", None)
            if source == "unknown":
                continue
            try:
                key = f"entry_{int(record['registry_index'])}_frame_{int(fi)}"
                if record.get("object_id") != obj.id or record.get("frame_index") != fi:
                    raise ValueError("record identity mismatch")
                payload = Path(source).read_bytes() if source is not None else cached_bytes(record, key)
                if digest(payload) != record.get("reference_sha256"):
                    raise ValueError("PNG bytes changed")
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
                record.update(fraction="unknown", reason=("reference_changed_or_missing" if source is not None
                                                         else "cached_reference_changed_or_missing"))
                continue
            if key in members:
                raise ValueError(f"duplicate visibility reference identity: {key}")
            members[key] = np.frombuffer(payload, dtype=np.uint8)
            retained.append((record, key))
            write_archive |= source is not None

    if not write_archive:
        return  # Cache rebuild is validation only; do not overwrite history.
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **members)
    payload = buffer.getvalue()
    archive_hash = digest(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=out_dir, prefix="visibility_refs_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        if file_sha256(temporary) != archive_hash:
            raise OSError("visibility archive write verification failed")
        os.replace(temporary, archive_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    for record, key in retained:
        record.update(reference=f"visibility_refs.npz#{key}", reference_archive_sha256=archive_hash)


def save_geometry(geom, path: Path) -> None:
    np.savez_compressed(path, frame_indices=np.asarray(geom.frame_indices), intrinsics=np.stack(geom.intrinsics), cam_to_world=np.stack(geom.cam_to_world), depth=np.stack(geom.depth).astype(np.float16),
                        depth_conf=np.stack(geom.depth_conf).astype(np.float16) if geom.depth_conf is not None else np.zeros(1), has_conf=np.asarray(geom.depth_conf is not None), scale=np.asarray(geom.scale), backend=np.asarray(geom.backend), notes=np.asarray(geom.notes), source_program_sha256=np.asarray(geom.source_program_sha256))

def load_geometry(path: Path):
    from .base import Geometry
    z = np.load(path)
    return Geometry(frame_indices=[int(i) for i in z["frame_indices"]], intrinsics=[k for k in z["intrinsics"]], cam_to_world=[m for m in z["cam_to_world"]], depth=[d.astype(np.float32) for d in z["depth"]],
                    depth_conf=[c.astype(np.float32) for c in z["depth_conf"]] if bool(z["has_conf"]) else None, scale=str(z["scale"]), backend=str(z["backend"]), notes=str(z["notes"]), source_program_sha256=str(z["source_program_sha256"]) if "source_program_sha256" in z.files else "")

def rebuild_evidence(out_dir: Path, cfg: dict) -> dict:
    """Recompute evidence.json from cached geometry.npz + masks.npz (CPU only)."""
    from .base import Tracks, TrackedObject
    out_dir = Path(out_dir)
    frames = json.loads((out_dir / "frames" / "frames.json").read_text())["frames"]
    keyframes = json.loads((out_dir / "keyframes.json").read_text())
    geom = load_geometry(out_dir / "geometry.npz")
    mi = json.loads((out_dir / "masks_index.json").read_text()); masks = load_masks(out_dir)
    objs = [TrackedObject(id=oid, phrase=m["phrase"], score=m["score"], masks=masks.get(oid, {}),
                          identity_hypotheses=m.get("identity_hypotheses", []), source_detection_ids=m.get("source_detection_ids", []),
                          visibility_measurements={int(k): v for k, v in m.get("visibility_measurements", {}).items()})
            for oid, m in mi["objects"].items()]
    tracks = Tracks(objects=objs, frame_size=tuple(mi["frame_size"]), backend=mi["backend"],
                    association_diagnostics=mi.get("association_diagnostics", []))
    save_visibility_references(tracks, out_dir)
    old = json.loads((out_dir / "evidence.json").read_text()) if (out_dir / "evidence.json").exists() else None
    rebuilt = build_evidence(old["meta"]["clip"] if old else out_dir.parent.name, frames, geom, tracks, cfg, out_dir,
                             keyframes, old["meta"]["fallbacks"] if old else [], old["meta"].get("phrases_source", "") if old else "")
    if old:
        for key in ("source_video_sha256", "source_video_path"):
            if key in old["meta"]:
                rebuilt["meta"][key] = old["meta"][key]
    return rebuilt

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
    if name == "none":
        from .base import NullSegmentationBackend; return NullSegmentationBackend(cfg)
    raise ValueError(f"unknown backend {name!r} (see configs/default.yaml perception.*_backend)")

if __name__ == "__main__":
    import argparse
    from ..config import load_config
    ap = argparse.ArgumentParser(); ap.add_argument("--video"); ap.add_argument("--clip"); ap.add_argument("--out", required=True); ap.add_argument("--phrases", default=None, help="comma-separated noun phrases"); ap.add_argument("--rebuild", action="store_true", help="recompute evidence from cached geometry/masks in --out (CPU)")
    a = ap.parse_args(); cfg = load_config()
    if a.rebuild: ev = rebuild_evidence(Path(a.out), cfg)
    else: ev = run_perception(a.video, a.clip, Path(a.out), cfg, phrases=[p.strip() for p in a.phrases.split(",")] if a.phrases else None)
    print(json.dumps({"objects": [(o["id"], o["class_guess"], o["motion_guess"]["type"], len(o["obb"])) for o in ev["objects"]], "backend": ev["meta"]["geometry_backend"], "fallbacks": ev["meta"]["fallbacks"], "timing": ev["meta"]["timing_s"]}, indent=1))
