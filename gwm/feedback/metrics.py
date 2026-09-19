"""Numeric feedback: per-object mask IoU / centroid / trajectory error / size ratio against evidence, plus DINOv2 global similarity. Produces clauses and a scalar score."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from PIL import Image

from ..compiler.ids import id_to_color, registry_order, decode_id_png, decode_depth_png
from ..perception.masks import resize_mask as _resize_bool, iou, centroid

class Dino:
    def __init__(self, weights_dir: str | Path):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(str(weights_dir)); self.model = AutoModel.from_pretrained(str(weights_dir)).to(self.device).eval()
    def embed(self, path) -> np.ndarray:
        import torch
        with torch.no_grad():
            out = self.model(**self.proc(images=Image.open(path).convert("RGB"), return_tensors="pt").to(self.device))
        v = out.pooler_output[0] if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[0, 0]
        v = v.float().cpu().numpy(); return v / (np.linalg.norm(v) + 1e-9)

def compute_metrics(program: dict, render_index: dict, render_dir: Path, evidence: dict, frames: list[dict], sam_masks: dict, cfg: dict, dino: Dino | None = None) -> dict:
    """sam_masks: {object_id: {frame_index: bool mask (frame-res)}} from the perception stage (saved as npz)."""
    th = cfg["feedback"]["thresholds"]; W, H = render_index["width"], render_index["height"]
    reg = registry_order(program); ev_objs = {o["id"]: o for o in evidence["objects"]}
    per_obj: dict[str, dict] = {}; clauses = []; dinos = []
    frame_by_t = {round(f["t"], 3): f for f in frames}
    for fr in render_index["frames"]:
        t = fr["t"]; f = frame_by_t.get(round(t, 3)) or min(frames, key=lambda x: abs(x["t"] - t))
        id_masks = decode_id_png(render_dir / fr["files"]["id"], reg)
        if dino is not None and "rgb" in fr["files"]:
            try: dinos.append(float(dino.embed(render_dir / fr["files"]["rgb"]) @ dino.embed(f["file"])))
            except Exception: pass
        for oid, mask_by_frame in sam_masks.items():
            m_video = mask_by_frame.get(f["index"])
            if m_video is None: continue
            m_video = _resize_bool(m_video, (W, H))
            pid = _match_program_id(oid, program)
            m_render = id_masks.get(pid, np.zeros((H, W), bool)) if pid else np.zeros((H, W), bool)
            d = per_obj.setdefault(oid, {"program_id": pid, "ious": [], "cent_px": [], "ate": [], "times": []})
            d["ious"].append(iou(m_video, m_render)); d["times"].append(t)
            cv, cr = centroid(m_video), centroid(m_render)
            if cv and cr: d["cent_px"].append(math.hypot(cv[0] - cr[0], cv[1] - cr[1]) / math.hypot(W, H))
            elif cv and not cr: d["cent_px"].append(1.0)
            # 3D trajectory error: kernel state position vs evidence OBB centre at the nearest time
            st_obj = next((s for s in fr["state"]["objects"] if s["id"] == pid), None) if pid else None
            ob = min(ev_objs[oid]["obb"], key=lambda b: abs(b["t"] - t)) if oid in ev_objs and ev_objs[oid]["obb"] else None
            if st_obj and ob and abs(ob["t"] - t) < 0.6: d["ate"].append(float(np.linalg.norm(np.asarray(st_obj["pos"]) - np.asarray(ob["center"]))))
    scores = []
    for oid, d in per_obj.items():
        miou = float(np.mean(d["ious"])) if d["ious"] else 0.0; mcent = float(np.mean(d["cent_px"])) if d["cent_px"] else 1.0; mate = float(np.mean(d["ate"])) if d["ate"] else None
        d.update(mean_iou=miou, mean_centroid_frac=mcent, mean_ate=mate)
        if d["program_id"] is None:
            clauses.append({"object_id": oid, "check": "presence", "status": "FAIL", "evidence": {"note": "no program object matches this evidence object"}, "hint": "missing"})
            scores.append(0.0); continue
        clauses.append({"object_id": d["program_id"], "check": "presence", "status": "PASS", "evidence": {}, "hint": ""})
        st = "FAIL" if miou < th["iou_fail"] else "PASS"
        hint = "" if st == "PASS" else ("suspect scale" if mcent < th["centroid_px_frac_fail"] else "suspect pose")
        if st == "FAIL" and mate is not None and mate > th["ate_m_fail"] and ev_objs.get(oid, {}).get("is_dynamic"): hint = "suspect timing or motion parameters"
        clauses.append({"object_id": d["program_id"], "check": "mask_iou", "status": st, "evidence": {"iou": round(miou, 3), "centroid_frac": round(mcent, 4), "ate_m": None if mate is None else round(mate, 3), "t_range": [min(d["times"]), max(d["times"])]}, "hint": hint})
        if mate is not None: clauses.append({"object_id": d["program_id"], "check": "trajectory_ate", "status": "FAIL" if mate > th["ate_m_fail"] else "PASS", "evidence": {"ate_m": round(mate, 3)}, "hint": "check motion type/period/phase" if mate > th["ate_m_fail"] else ""})
        w = cfg["feedback"]["weights"]
        s = w["iou"] * miou + w["centroid"] * (1 - min(mcent / 0.2, 1)) + w["ate"] * (1 - min((mate or 0) / 1.0, 1)) + w["dino"] * (float(np.mean(dinos)) if dinos else 0.5)
        scores.append(s)
    # program objects with no evidence counterpart
    ev_pids = {d["program_id"] for d in per_obj.values()}
    for o in program.get("objects", []):
        if o["id"] not in ev_pids: clauses.append({"object_id": o["id"], "check": "presence", "status": "WARN", "evidence": {"note": "program object has no evidence counterpart"}, "hint": "extra?"})
    summary = {"score": float(np.mean(scores)) if scores else 0.0, "dino_mean": float(np.mean(dinos)) if dinos else None, "n_fail": sum(c["status"] == "FAIL" for c in clauses), "per_object": {k: {kk: vv for kk, vv in v.items() if kk in ("program_id", "mean_iou", "mean_centroid_frac", "mean_ate")} for k, v in per_obj.items()}}
    return {"clauses": clauses, "summary": summary}

def _match_program_id(ev_id: str, program: dict) -> str | None:
    """Evidence id -> program object id. Exact match first (direct translation and the writer keep evidence ids);
    otherwise a program object whose id equals the evidence id without its instance suffix (coin_3 -> coin)."""
    ids = [o["id"] for o in program.get("objects", [])] + [s["id"] for s in program.get("static", [])]
    if ev_id in ids: return ev_id
    base = ev_id.rsplit("_", 1)[0]
    if base in ids: return base
    return None
