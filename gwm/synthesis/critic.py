"""VLM critic: only FAIL clauses; crops + overlay + numbers -> structured JSON Patch suggestions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .vlm import VLMClient
from ..compiler.ids import registry_order, decode_id_png
from ..perception.masks import resize_mask as _resize_bool

REPO = Path(__file__).resolve().parents[2]
CRITIC_SCHEMA = {"type": "object", "required": ["items", "summary"], "additionalProperties": False, "properties": {
    "items": {"type": "array", "items": {"type": "object", "required": ["object_id", "issue", "evidence", "suggested_edit"], "additionalProperties": False, "properties": {
        "object_id": {"type": "string"}, "issue": {"type": "string", "enum": ["missing", "extra", "pose", "scale", "timing", "class", "camera", "static"]}, "evidence": {"type": "string"},
        "suggested_edit": {"type": "array", "maxItems": 3, "items": {"type": "object", "required": ["op", "path"], "properties": {"op": {"type": "string", "enum": ["replace", "add", "remove"]}, "path": {"type": "string"}, "value": {}}}}}}},
    "summary": {"type": "string"}}}

def make_crops(program: dict, render_index: dict, render_dir: Path, frames: list[dict], sam_masks: dict, clause_ids: set[str], out_dir: Path, pad: float = 0.25) -> dict[str, list[str]]:
    """For each failing program object: [video_crop, render_crop, overlay] at the frame where the video mask is largest."""
    out_dir.mkdir(parents=True, exist_ok=True); reg = registry_order(program); W, H = render_index["width"], render_index["height"]
    inv = {}
    for ev_id, mbf in sam_masks.items():
        from ..feedback.metrics import _match_program_id
        pid = _match_program_id(ev_id, program)
        if pid: inv.setdefault(pid, ev_id)
    crops = {}
    for pid in clause_ids:
        ev_id = inv.get(pid)
        best = None
        for fr in render_index["frames"]:
            f = min(frames, key=lambda x: abs(x["t"] - fr["t"]))
            mv = sam_masks.get(ev_id, {}).get(f["index"]) if ev_id else None
            mv = _resize_bool(mv, (W, H)) if mv is not None else np.zeros((H, W), bool)
            mr = decode_id_png(render_dir / fr["files"]["id"], reg).get(pid, np.zeros((H, W), bool))
            area = mv.sum() + mr.sum()
            if best is None or area > best[0]: best = (area, fr, f, mv, mr)
        if best is None or best[0] == 0: continue
        _, fr, f, mv, mr = best
        ys, xs = np.where(mv | mr); x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x2 - x1 + 1, y2 - y1 + 1; x1 = max(0, int(x1 - pad * bw)); x2 = min(W - 1, int(x2 + pad * bw)); y1 = max(0, int(y1 - pad * bh)); y2 = min(H - 1, int(y2 + pad * bh))
        video = Image.open(f["file"]).convert("RGB").resize((W, H)); render = Image.open(render_dir / fr["files"]["rgb"]).convert("RGB")
        ov = np.asarray(video).copy(); ov[mv] = (0.5 * ov[mv] + [128, 0, 0]).astype(np.uint8); ov[mr] = (0.5 * ov[mr] + [0, 128, 0]).astype(np.uint8)
        files = []
        for name, im in (("video", video), ("render", render), ("overlay", Image.fromarray(ov))):
            p = out_dir / f"{pid}_{name}.jpg"; im.crop((x1, y1, x2, y2)).resize((max(64, (x2 - x1) * 2), max(64, (y2 - y1) * 2))).save(p, quality=90); files.append(str(p))
        crops[pid] = files
    return crops

def critique(client: VLMClient, program: dict, clauses: list[dict], crops: dict[str, list[str]], out_path: Path, evidence: dict | None = None) -> dict:
    fails = [c for c in clauses if c["status"] == "FAIL" and c["object_id"] in crops]
    if not fails:
        res = {"items": [], "summary": "no failing objects with crops"}; out_path.write_text(json.dumps(res, indent=1)); return res
    by_obj: dict[str, list[dict]] = {}
    for c in fails: by_obj.setdefault(c["object_id"], []).append(c)
    images, parts = [], []
    for pid, cl in list(by_obj.items())[:4]:
        node = next((o for o in program["objects"] if o["id"] == pid), None); idx = next((i for i, o in enumerate(program["objects"]) if o["id"] == pid), None)
        images.extend(crops[pid])
        ev_obj = next((o for o in (evidence or {}).get("objects", []) if o["id"] == pid or o["id"].rsplit("_", 1)[0] == pid), None)
        ev_txt = ""
        if ev_obj and ev_obj.get("obb"):
            ob = ev_obj["obb"]; step = max(1, len(ob) // 6)
            ev_txt = "\nevidence centres over time (metres, world frame): " + "; ".join(f"t={b['t']:.2f} c={[round(v, 3) for v in b['center']]}" for b in ob[::step]) + f"\nevidence size: {[round(v, 3) for v in ob[0]['size']]}"
        parts.append(f"OBJECT {pid} (program path /objects/{idx}): images = video crop, render crop, overlay (red = video mask, green = render mask).\nfailing checks: {json.dumps(cl)}\ncurrent node: {json.dumps(node)}{ev_txt}\nIf the pose is wrong, replace /objects/{idx}/pose/pos with the evidence centre at the first time (absolute metres). Never output zero vectors or placeholder values.")
    system = (REPO / "prompts" / "critic.md").read_text()
    r = client.chat(system, "\n\n".join(parts), images=images, json_schema=CRITIC_SCHEMA, temperature=0.3)
    res = r["json"] or {"items": [], "summary": "no output"}
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False)); return res
