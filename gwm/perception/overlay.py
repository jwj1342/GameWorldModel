"""把感知结果画回视频帧上，拼成一段检查视频。纯诊断用途，和证据提取本身无关，所以单独放一个文件。"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from .base import CV2THREE, Geometry, Tracks
from .masks import resize_mask as _resize_mask
from .evidence import _fix

def project(pts_w: np.ndarray, K: np.ndarray, c2w_aligned: np.ndarray, scale: float):
    """World (aligned, scaled) -> pixels using aligned cam pose."""
    w2c = np.linalg.inv(c2w_aligned); pc = (w2c[:3, :3] @ (pts_w / scale).T).T + w2c[:3, 3]  # OpenCV camera coords (c2w_aligned maps OpenCV-camera points)
    z = pc[:, 2]; ok = z > 1e-4
    u = K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]; v = K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]
    return np.stack([u, v], 1), ok

def render_overlay(frames, geom: Geometry, tracks: Tracks, evidence: dict, T, scale, out_dir: Path, fps: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = geom.size()
    colors = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 200, 60), (230, 120, 255), (255, 140, 40), (80, 255, 220), (200, 200, 200)]
    for k, fi in enumerate(geom.frame_indices):
        im = Image.open(frames[fi]["file"]).convert("RGB").resize((W, H)); dr = ImageDraw.Draw(im, "RGBA")
        c2w = T @ _fix(geom.cam_to_world[k])
        for j, o in enumerate(evidence["objects"]):
            col = colors[j % len(colors)]
            ob = next((b for b in o["obb"] if b.get("frame") == fi), None)
            m = next((t for t in tracks.objects if t.id == o["id"]), None)
            if m is not None and fi in m.masks:
                mm = _resize_mask(m.masks[fi], (W, H)); ov = Image.new("RGBA", (W, H), col + (0,)); ov.putalpha(Image.fromarray((mm * 70).astype(np.uint8))); im.paste(ov, (0, 0), ov); dr = ImageDraw.Draw(im, "RGBA")
            if ob is None: continue
            c = np.array(ob["center"]); s = np.array(ob["size"]) / 2; Rm = R.from_quat(ob["quat"]).as_matrix()
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
            cw = (Rm @ corners.T).T + c; uv, ok = project(cw, geom.intrinsics[k], c2w, scale)
            edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            for a, b in edges:
                if ok[a] and ok[b]: dr.line([tuple(uv[a]), tuple(uv[b])], fill=col + (255,), width=2)
            if ok.any(): dr.text((float(uv[ok][:, 0].min()), float(uv[ok][:, 1].min()) - 12), f"{o['id']} {o['motion_guess']['type']}", fill=col + (255,))
        dr.text((6, 6), f"t={frames[fi]['t']:.2f}s  {geom.backend}/{tracks.backend}", fill=(255, 255, 255, 255))
        im.save(out_dir / f"ov_{k:05d}.jpg", quality=85)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(out_dir / "ov_%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_dir.parent / "overlay.mp4")], check=False)
