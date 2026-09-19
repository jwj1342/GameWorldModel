"""Rule-based gameplay binding for platformer_3p: spawn, goal, collectibles, hazards; add ground if the scene has none."""
from __future__ import annotations
import math
import numpy as np

def _top_surfaces(program: dict) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(min, max, id) world AABBs of static boxes/planes and static objects (approximate, ignores rotation except for extent swap)."""
    out = []
    for s in program.get("static", []):
        g = s["geom"]; p = np.asarray(s["pose"]["pos"], float)
        if g.get("kind") == "primitive" and g.get("shape") == "box": e = np.asarray(g["extent"], float) / 2
        elif g.get("kind") == "primitive" and g.get("shape") == "plane": e = np.asarray([g["size"][0] / 2, 0.05, g["size"][1] / 2])
        elif g.get("kind") == "heightfield": e = np.asarray(g["scale"], float) / 2
        else: e = np.asarray([0.5, 0.5, 0.5])
        out.append((p - e, p + e, s["id"]))
    return out

def scale_program(program: dict, s: float) -> dict:
    """Uniformly scale every length in the program (positions, extents, radii, motion amplitudes, keyframes, camera)."""
    import copy
    p = copy.deepcopy(program)
    def v(x): return [float(a) * s for a in x]
    def geom(g):
        for k in ("extent", "size", "scale"):
            if k in g: g[k] = v(g[k])
        for k in ("radius", "radius_top", "radius_bottom", "height", "tube"):
            if k in g and g[k] is not None: g[k] = float(g[k]) * s
    for n in p.get("static", []) + p.get("objects", []):
        geom(n["geom"])
        if n.get("pose"): n["pose"]["pos"] = v(n["pose"]["pos"])
        for inst in n.get("instances", []) or []: inst["pos"] = v(inst["pos"])
        m = n.get("motion")
        if m:
            if m.get("pivot"): m["pivot"] = v(m["pivot"])
            if m.get("amp") is not None: m["amp"] = float(m["amp"]) * s
            if m.get("rate") is not None and m.get("type") == "prismatic": m["rate"] = float(m["rate"]) * s
            if m.get("range") and m.get("type") == "prismatic": m["range"] = v(m["range"])
            if m.get("from") is not None and m.get("type") == "prismatic": m["from"] = float(m["from"]) * s
            for kf in m.get("keyframes", []) or []: kf["pos"] = v(kf["pos"])
            for sc in m.get("schedule", []) or []:
                if "to" in sc and m.get("type") == "prismatic": sc["to"] = float(sc["to"]) * s
        for ev in n.get("events", []) or []:
            if ev.get("volume"):
                if "pos" in ev["volume"]: ev["volume"]["pos"] = v(ev["volume"]["pos"])
                if "extent" in ev["volume"]: ev["volume"]["extent"] = v(ev["volume"]["extent"])
    for kf in (p.get("camera") or {}).get("keyframes", []) or []: kf["pos"] = v(kf["pos"])
    if p.get("camera", {}).get("intrinsics", {}).get("far"): p["camera"]["intrinsics"]["far"] = float(p["camera"]["intrinsics"]["far"]) * s
    slots = (p.get("binding") or {}).get("slots") or {}
    if slots.get("player_spawn"): slots["player_spawn"] = v(slots["player_spawn"])
    if slots.get("goal_volume"):
        slots["goal_volume"]["pos"] = v(slots["goal_volume"]["pos"])
        if slots["goal_volume"].get("extent"): slots["goal_volume"]["extent"] = v(slots["goal_volume"]["extent"])
    p.setdefault("meta", {})["game_scale"] = float(program.get("meta", {}).get("game_scale", 1.0)) * s
    return p

def normalise_game_scale(program: dict, cfg: dict) -> tuple[dict, float]:
    """Scale so that the largest static footprint's long side equals binding.target_ground_extent_m."""
    target = cfg["binding"].get("target_ground_extent_m")
    surfaces = _top_surfaces(program)
    if not target or not surfaces: return program, 1.0
    gmin, gmax, _ = max(surfaces, key=lambda s: (s[1][0] - s[0][0]) * (s[1][2] - s[0][2]))
    long_side = float(max(gmax[0] - gmin[0], gmax[2] - gmin[2]))
    if long_side < 1e-3: return program, 1.0
    s = target / long_side
    if abs(s - 1.0) < 0.05: return program, 1.0
    return scale_program(program, s), s

def bind(program: dict, evidence: dict | None, cfg: dict, prompt: str | None = None) -> dict:
    b = cfg["binding"]; program = dict(program); program.setdefault("binding", {"template": b["template"], "slots": {}}); slots = program["binding"].setdefault("slots", {})
    notes = []
    # ground
    surfaces = _top_surfaces(program)
    if not surfaces and b.get("add_ground_if_missing", True):
        objs = [np.asarray(o["pose"]["pos"]) for o in program.get("objects", []) if o.get("pose")]
        c = np.mean(objs, 0) if objs else np.zeros(3); ext = (np.ptp(np.asarray(objs), 0) if len(objs) > 1 else np.zeros(3)) + 2 * b["ground_margin_m"]
        program["static"] = list(program.get("static", [])) + [{"id": "ground_auto", "class": "ground", "geom": {"kind": "primitive", "shape": "box", "extent": [float(max(ext[0], 8)), 0.3, float(max(ext[2], 8))]}, "pose": {"pos": [float(c[0]), -0.15, float(c[2])]}, "material": "grass", "notes": "added by binder"}]
        notes.append("added ground_auto"); surfaces = _top_surfaces(program)
    program, gs = normalise_game_scale(program, cfg)
    slots = program.setdefault("binding", {"template": b["template"], "slots": {}}).setdefault("slots", {})  # re-bind: normalisation deep-copies the program
    if gs != 1.0: notes.append(f"game scale x{gs:.2f}")
    surfaces = _top_surfaces(program)
    # objects whose bottom is below the ground top are lifted to rest on it
    if b.get("lift_objects_to_ground", True):
        gtop = max(surfaces, key=lambda s: (s[1][0] - s[0][0]) * (s[1][2] - s[0][2]))[1][1]
        for o in program.get("objects", []):
            if not o.get("pose"): continue
            h = (o["geom"].get("extent") or [1, o["geom"].get("height", 1) or 1, 1])[1]
            bottom = o["pose"]["pos"][1] - h / 2
            if bottom < gtop - 0.02:
                o["pose"]["pos"][1] = float(gtop + h / 2 + 0.01); o["notes"] = (o.get("notes", "") + " lifted to ground").strip()
                for kf in (o.get("motion") or {}).get("keyframes", []) or []: kf["pos"][1] = max(kf["pos"][1], gtop + h / 2 + 0.01)
    # walkable = largest top face near the lowest level
    ground = max(surfaces, key=lambda s: (s[1][0] - s[0][0]) * (s[1][2] - s[0][2]))
    gmin, gmax, gid = ground; top = float(gmax[1])
    cam0 = np.asarray(evidence["camera"]["poses"][0]["pos"]) * gs if evidence and evidence.get("camera", {}).get("poses") else np.asarray([gmin[0], top, gmax[2]])
    # spawn: point on the ground nearest the first camera (clamped inside with margin), goal: farthest ground corner
    m = 0.15
    def clamp(pt): return np.array([np.clip(pt[0], gmin[0] + m * (gmax[0] - gmin[0]), gmax[0] - m * (gmax[0] - gmin[0])), top, np.clip(pt[2], gmin[2] + m * (gmax[2] - gmin[2]), gmax[2] - m * (gmax[2] - gmin[2]))])
    spawn = clamp(cam0)
    corners = [np.array([x, top, z]) for x in (gmin[0] + m * (gmax[0] - gmin[0]), gmax[0] - m * (gmax[0] - gmin[0])) for z in (gmin[2] + m * (gmax[2] - gmin[2]), gmax[2] - m * (gmax[2] - gmin[2]))]
    # avoid placing goal inside a static obstacle: prefer corners not overlapping other static boxes
    def blocked(pt):
        return any((s[2] != gid) and (s[0][0] - 0.5 <= pt[0] <= s[1][0] + 0.5) and (s[0][2] - 0.5 <= pt[2] <= s[1][2] + 0.5) and s[1][1] > top + 0.3 for s in surfaces)
    cands = [c for c in corners if not blocked(c)] or corners
    goal = max(cands, key=lambda c: np.linalg.norm(c - spawn))
    slots.setdefault("player_spawn", [float(spawn[0]), float(top + 1.2), float(spawn[2])])
    slots.setdefault("goal_volume", {"pos": [float(goal[0]), float(top + 1.0), float(goal[2])], "extent": [1.5, 2.0, 1.5]})
    slots.setdefault("walkable", "auto")
    coll = [o["id"] for o in program.get("objects", []) if any(k in (o.get("class") or "").lower() for k in b["collectible_classes"])]
    haz = [n["id"] for n in program.get("static", []) + program.get("objects", []) if any(k in ((n.get("class") or "") + " " + (n.get("material") or "")).lower() for k in b["hazard_classes"])]
    slots.setdefault("collectibles", coll); slots.setdefault("hazards", haz)
    # collectibles get the despawn event if missing
    for o in program.get("objects", []):
        if o["id"] in coll and not any(e.get("type") == "despawn_on_contact" for e in o.get("events", [])):
            o.setdefault("events", []).append({"type": "despawn_on_contact", "with": "player"})
    if prompt: notes.append(f"prompt ignored in MVP binder: {prompt[:80]}")
    program["binding"]["notes"] = "; ".join(notes) if notes else ""
    program["binding"] = {k: v for k, v in program["binding"].items() if k in ("template", "slots")} | ({} if not notes else {})
    program.setdefault("meta", {}).setdefault("notes", "")
    if notes: program["meta"]["notes"] = (program["meta"]["notes"] + " | binder: " + "; ".join(notes)).strip(" |")
    return program
