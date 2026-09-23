"""Evidence -> program without any VLM (baseline, ablation, and fallback when generation fails)."""
from __future__ import annotations
import math
import numpy as np
from ..perception.contract import format_evidence_errors, validate_evidence

def _slug(s: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_"); s = s if s and s[0].isalpha() else "o_" + s
    return s[:40]

def guess_material(cls: str) -> str:
    c = cls.lower()
    for key, mat in [("coin", "gold"), ("gold", "gold"), ("box", "cardboard"), ("carton", "cardboard"), ("train", "red"), ("car", "red"), ("door", "wood"), ("table", "wood"),
                     ("track", "plastic"), ("rail", "plastic"), ("belt", "rubber"), ("conveyor", "metal"), ("ball", "blue"), ("wall", "stone"), ("floor", "concrete")]:
        if key in c: return mat
    return "default"

def _confidence(value):
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else None

def motion_from_guess(mg: dict, obbs: list[dict]) -> dict | None:
    t = mg.get("type", "unknown")
    if t in ("static", "unknown") or len(obbs) < 2: return None
    ts = [o["t"] for o in obbs]; cs = np.asarray([o["center"] for o in obbs])
    if t == "periodic_translate":
        # kernel: c(t) = base + axis*amp*sin(2pi t/T + phase); base must be the mean centre; pick the phase branch matching the initial velocity
        ax = np.asarray(mg["axis"]); mean = cs.mean(0); proj = (cs - mean) @ ax; amp = max(mg.get("amp") or 0.1, 1e-3); T = float(mg["period"])
        phi = math.asin(max(-1, min(1, proj[0] / amp)))
        v0 = float(proj[min(2, len(proj) - 1)] - proj[0])
        if math.cos(phi) * v0 < 0: phi = math.pi - phi
        phase = phi - 2 * math.pi * ts[0] / T
        return {"type": "periodic_translate", "axis": [float(v) for v in ax], "amp": float(amp), "period": T, "phase": float(phase), "_base": [float(v) for v in mean]}
    if t == "periodic_rotate":
        return {"type": "periodic_rotate", "axis": [0, 1, 0], "amp_deg": float(mg.get("amp_deg") or 30), "period": float(mg.get("period") or max(ts[-1] - ts[0], 1.0) * 2), "phase": float(mg.get("phase") or 0.0)}
    if t == "spin":
        return {"type": "spin", "axis": [0, 1, 0], "rate_dps": float(mg.get("rate") or 90)}
    if t == "revolute" and mg.get("pivot"):
        rng = mg.get("range") or [0, 90]
        return {"type": "revolute", "axis": [0, 1, 0], "pivot": [float(v) for v in mg["pivot"]], "range_deg": [0.0, float(max(rng[1], 5))],
                "schedule": [{"t": float(ts[0]), "to_deg": float(max(rng[1], 5)), "duration": float(max(ts[-1] - ts[0], 0.5))}]}
    # prismatic / trajectory: keyframes (subsample to <= 12)
    idx = np.unique(np.round(np.linspace(0, len(obbs) - 1, min(12, len(obbs)))).astype(int))
    return {"type": "trajectory", "interp": "linear", "keyframes": [{"t": float(ts[i]), "pos": [float(v) for v in cs[i]]} for i in idx]}

def evidence_to_program(ev: dict, clip: str | None = None, cfg: dict | None = None) -> dict:
    evidence_report = validate_evidence(ev)
    if not evidence_report["ok"]:
        raise ValueError("invalid evidence:\n" + format_evidence_errors(evidence_report))
    ev = evidence_report["evidence"]
    meta = ev["meta"]; cam = ev["camera"]; g = ev["static"]["planes"][0] if ev["static"]["planes"] else None
    program = {"meta": {"clip": clip or meta["clip"], "fps": 30, "duration": float(max(meta["duration"], 1.0)), "units": "m", "up": "y", "notes": "direct translation of evidence (no VLM)"},
               "style": {"background": "#8fb3d9"},
               "camera": {"intrinsics": {"fov_deg": float(cam["intrinsics"].get("fov_deg", 60)), "aspect": cam["intrinsics"]["width"] / cam["intrinsics"]["height"], "far": 100},
                          "keyframes": [{"t": p["t"], "pos": p["pos"], "quat": p["quat"]} for p in cam["poses"][::max(1, len(cam["poses"]) // 24)]], "interp": "catmull_rom"},
               "static": [], "objects": [], "binding": {"template": "platformer_3p", "slots": {}}, "residual": None}
    if g:
        ext = g.get("extent_hint") or [10, 0.2, 10]; c = g.get("center_hint") or [0, 0, 0]
        ground_node = {"id": "ground", "class": "ground", "geom": {"kind": "primitive", "shape": "box", "extent": [float(max(ext[0], 2)) * 1.4, 0.3, float(max(ext[2], 2)) * 1.4]}, "pose": {"pos": [float(c[0]), -0.15, float(c[2])]}, "material": "grass"}
        if _confidence(g.get("conf")) is not None: ground_node["confidence"] = _confidence(g["conf"])
        program["static"].append(ground_node)
    static_classes = [k for k in (cfg or {}).get("perception", {}).get("static_classes", [])] if cfg else ["wall", "floor", "ground", "table", "track", "rail", "belt", "conveyor", "road", "lawn", "grass"]
    for o in ev["objects"]:
        if not o["obb"]: continue
        cls = o["class_guess"].lower()
        if o["motion_guess"].get("type") == "static" and any(k in cls for k in static_classes):
            # fixed structure: median OBB as a static box
            cs = np.median(np.asarray([b["center"] for b in o["obb"]]), 0); sz = np.median(np.asarray([b["size"] for b in o["obb"]]), 0)
            sz = [float(max(v, 0.05)) for v in sz]
            if any(k in cls for k in ("floor", "ground", "lawn", "grass", "carpet", "mat", "road")): sz[1] = min(sz[1], 0.3); cs[1] = -sz[1] / 2 + 0.0  # thin slab flush with the ground plane
            static_node = {"id": _slug(o["id"]), "class": cls[:40], "geom": {"kind": "primitive", "shape": "box", "extent": sz}, "pose": {"pos": [float(v) for v in cs], "quat": [float(v) for v in o["obb"][0]["quat"]]},
                           "material": guess_material(cls), "notes": "static structure from evidence"}
            if _confidence(o.get("confidence")) is not None: static_node["confidence"] = _confidence(o["confidence"])
            program["static"].append(static_node)
            continue
        first = o["obb"][0]; size = [float(max(s, 0.05)) for s in first["size"]]
        node = {"id": _slug(o["id"]), "class": o["class_guess"][:40],
                "geom": {"kind": "asset", "query": o["class_guess"][:80], "extent": size}, "pose": {"pos": [float(v) for v in first["center"]], "quat": [float(v) for v in first["quat"]]},
                "material": guess_material(o["class_guess"]), "notes": o["motion_guess"].get("notes", "")[:200]}
        if _confidence(o.get("confidence")) is not None: node["confidence"] = _confidence(o["confidence"])
        m = motion_from_guess(o["motion_guess"], o["obb"])
        if m:
            base = m.pop("_base", None)
            if base: node["pose"]["pos"] = base
            node["motion"] = m
        if any(c["with_id"] == "ground" for c in o.get("contacts", [])) and g: node["support"] = "ground"
        program["objects"].append(node)
    return program
