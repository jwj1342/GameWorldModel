"""Ground-truth perception backends for synthetic clips: render depth + ID passes of the *source program* with the harness
(CPU only) and expose them through the GeometryBackend / SegmentationBackend protocols. Lets the rest of the pipeline
(evidence, direct translation, feedback, binding, playtest) be tested without GPU models, with perfect perception."""
from __future__ import annotations
import json, tempfile
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from .base import Geometry, Tracks, TrackedObject
from ..compiler.compile import compile_program
from ..feedback.render import render
from ..compiler.ids import registry_order, decode_depth_png, decode_id_png_by_entry

CV2THREE = np.diag([1.0, -1.0, -1.0])

@lru_cache(maxsize=4)
def _render_gt(program_path: str, times: tuple, width: int, height: int):
    program = json.loads(Path(program_path).read_text())
    tmp = Path(tempfile.mkdtemp(prefix="gwm_gt_"))
    comp = compile_program(program, tmp / "game")
    if not comp["ok"]: raise RuntimeError("gt program does not compile")
    idx = render(tmp / "game", list(times), tmp / "render", width, height, ("rgb", "depth", "id"))
    return program, idx, tmp / "render"

def _size_for(frame_file: str, width: int) -> tuple[int, int]:
    with Image.open(frame_file) as im: h = int(round(im.height * width / im.width / 14) * 14)
    return width, max(14, min(h, 518))

class GTGeometryBackend:
    name = "gt"
    def __init__(self, program_path: str | Path, width: int = 518):
        self.program_path = str(program_path); self.width = width
    def estimate(self, frame_files, frame_indices, cfg) -> Geometry:
        fps = cfg["frames"]["sample_fps"]; times = tuple(round(i / fps, 3) for i in frame_indices)
        W, H = _size_for(frame_files[0], self.width)
        program, idx, rdir = _render_gt(self.program_path, times, W, H)
        far = float(idx.get("camera_far") or 100)
        intr, c2w, depth = [], [], []
        for fr in idx["frames"]:
            cam = fr["camera"]; fy = (H / 2) / np.tan(np.radians(cam["fov"]) / 2); K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], float)
            Rt = R.from_quat(cam["quat"]).as_matrix(); Rcv = Rt @ CV2THREE
            M = np.eye(4); M[:3, :3] = Rcv; M[:3, 3] = cam["pos"]
            F = np.eye(4); F[:3, :3] = CV2THREE
            c2w.append(F @ M)  # expressed in the VGGT-like (y-down) world so evidence._fix restores y-up
            d = decode_depth_png(rdir / fr["files"]["depth"], far); d[d >= far * 0.999] = 0.0
            intr.append(K); depth.append(d.astype(np.float32))
        return Geometry(frame_indices=list(frame_indices), intrinsics=intr, cam_to_world=c2w, depth=depth, depth_conf=[np.ones_like(d) for d in depth], scale="metric", backend=self.name, notes=f"ground truth from {self.program_path}")

class GTSegmentationBackend:
    name = "gt"
    def __init__(self, program_path: str | Path, width: int = 518):
        self.program_path = str(program_path); self.width = width
    def segment(self, frame_files, frame_indices, phrases, cfg) -> Tracks:
        fps = cfg["frames"]["sample_fps"]; times = tuple(round(i / fps, 3) for i in frame_indices)
        W, H = _size_for(frame_files[0], self.width)
        program, idx, rdir = _render_gt(self.program_path, times, W, H)
        reg = registry_order(program); classes = {o["id"]: o.get("class", o["id"]) for o in program["objects"] + program.get("static", [])}
        include_static = bool(cfg["perception"].get("gt_include_static", True))
        multi = {r["id"] for r in reg if r["kind"] == "object" and r.get("instance", 0) > 0}
        objs: dict[str, TrackedObject] = {}
        for fr, fi in zip(idx["frames"], frame_indices):
            by_entry = decode_id_png_by_entry(rdir / fr["files"]["id"], reg)
            for r in reg:
                if r["kind"] != "object" and not (include_static and classes.get(r["id"], "") not in ("ground",)): continue
                m = by_entry[r["idx"]]
                if m.sum() < 4: continue
                oid = f"{r['id']}_{r.get('instance', 0) + 1}" if r["id"] in multi else r["id"]   # one track per instance, like a detector would
                o = objs.setdefault(oid, TrackedObject(id=oid, phrase=classes.get(r["id"], r["id"]), score=1.0))
                o.masks[fi] = m; ys, xs = np.where(m); o.boxes[fi] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        return Tracks(objects=list(objs.values()), frame_size=(W, H), backend=self.name)
