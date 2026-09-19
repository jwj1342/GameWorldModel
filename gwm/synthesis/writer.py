"""Staged program generation with schema-constrained decoding and a validate->repair loop; falls back to evidence direct translation."""
from __future__ import annotations
import copy, json, re
from pathlib import Path
from typing import Any
from ..compiler.validate import validate, format_errors, schema as full_schema
from .direct import evidence_to_program
from .vlm import VLMClient

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "prompts"

def _p(name: str) -> str: return (PROMPTS / name).read_text()

def _defs(): return full_schema()["$defs"]

def stage_schema(stage: str) -> dict:
    d = _defs(); base = {"$defs": d, "type": "object", "additionalProperties": False}
    if stage == "camera_static":
        return {**base, "required": ["camera", "static"], "properties": {"style": {"type": "object", "properties": {"background": {"type": "string"}}}, "camera": full_schema()["properties"]["camera"], "static": {"type": "array", "items": {"$ref": "#/$defs/staticNode"}}}}
    if stage == "objects":
        obj = copy.deepcopy(d["objectNode"]); obj["properties"].pop("motion", None); obj["properties"].pop("events", None)
        return {**base, "required": ["objects"], "properties": {"objects": {"type": "array", "items": obj}}}
    if stage == "motion":
        return {**base, "required": ["motions"], "properties": {"motions": {"type": "object", "additionalProperties": {"type": "object", "required": ["motion"], "properties": {"motion": {"$ref": "#/$defs/motion"}, "events": {"type": "array", "items": {"$ref": "#/$defs/event"}}, "notes": {"type": "string"}}}}}}
    if stage == "single_stage":
        s = copy.deepcopy(full_schema()); s.pop("$schema", None); s.pop("$id", None); s["required"] = ["camera", "static", "objects"]; s["properties"].pop("binding", None); s["properties"].pop("residual", None); s["properties"].pop("meta", None)
        return s
    raise ValueError(stage)

def evidence_summary(ev: dict, max_obb: int = 12) -> str:
    cam = ev["camera"]; lines = [f"camera: fov_deg={cam['intrinsics'].get('fov_deg', 60):.1f}, aspect={cam['intrinsics']['width']/cam['intrinsics']['height']:.3f}, {len(cam['poses'])} poses; first pose pos={_r(cam['poses'][0]['pos'])} quat={_r(cam['poses'][0]['quat'])}, last pose pos={_r(cam['poses'][-1]['pos'])} quat={_r(cam['poses'][-1]['quat'])} (t={cam['poses'][-1]['t']:.2f}); scale={ev['meta']['scale']}"]
    g = ev["static"]["planes"][0] if ev["static"]["planes"] else None
    if g: lines.append(f"ground plane: y=0, centre_hint={_r(g['center_hint'])}, extent_hint={_r(g['extent_hint'])}, conf={g['conf']:.2f}")
    lines.append(f"objects ({len(ev['objects'])}):")
    for o in ev["objects"]:
        mg = o["motion_guess"]; ob = o["obb"]; step = max(1, len(ob) // max_obb)
        traj = "; ".join(f"t={b['t']:.2f} c={_r(b['center'])}" for b in ob[::step])
        mgs = ", ".join(f"{k}={_r(v) if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v)}" for k, v in mg.items() if k not in ("notes",) and v is not None)
        lines.append(f"- id={o['id']} class_guess='{o['class_guess']}' conf={o['confidence']:.2f} dynamic={o['is_dynamic']} size_first={_r(ob[0]['size'])} quat_first={_r(ob[0]['quat'])} contacts={[c['with_id'] for c in o.get('contacts', [])]}\n  motion_guess: {mgs} ({mg.get('notes','')})\n  centres: {traj}")
    return "\n".join(lines)

def camera_keyframes_from_evidence(ev: dict, n: int = 16) -> list[dict]:
    poses = ev["camera"]["poses"]; step = max(1, len(poses) // n)
    return [{"t": round(p["t"], 3), "pos": _r(p["pos"]), "quat": _r(p["quat"])} for p in poses[::step]]

def _r(v, nd=3):
    if isinstance(v, (list, tuple)): return [round(float(x), nd) for x in v]
    return round(float(v), nd)

class Writer:
    def __init__(self, client: VLMClient, cfg: dict, log_dir: Path):
        self.client, self.cfg, self.log_dir = client, cfg, Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.w = cfg["writer"]

    def generate(self, ev: dict, keyframe_files: list[str], clip: str, temperature: float | None = None, tag: str = "r0") -> tuple[dict, dict]:
        """Returns (program, info). Uses staged generation; each stage validated and repaired; falls back to evidence on failure."""
        use_ev = self.w.get("use_evidence", True) and not self.cfg["ablations"].get("no_evidence", False)
        stages = ["single_stage"] if self.cfg["ablations"].get("single_stage") else list(self.w["stages"])
        program = {"meta": {"clip": clip, "fps": 30, "duration": float(max(ev["meta"]["duration"], 1.0)), "units": "m", "up": "y", "notes": "generated"}, "style": {"background": "#8fb3d9"}, "camera": {}, "static": [], "objects": [], "binding": {"template": "platformer_3p", "slots": {}}, "residual": None}
        info = {"stages": {}, "fallbacks": []}
        ev_text = evidence_summary(ev) if use_ev else "(evidence withheld in this ablation; rely on the frames)"
        common = _p("common_dsl.md") + "\n" + _p("fewshot/handwritten_summary.md")
        for st in stages:
            ok, tries = False, 0
            errors_text = ""
            while tries < self.w["repair_attempts"] and not ok:
                tries += 1
                user = self._user_text(st, ev_text, program, errors_text, ev)
                try:
                    r = self.client.chat(common + "\n\n" + _p(f"writer_{st}.md"), user, images=keyframe_files, json_schema=stage_schema(st), temperature=temperature)
                    out = r["json"]
                except Exception as e:
                    errors_text = f"previous attempt failed: {e!r}"; (self.log_dir / f"{tag}_{st}_try{tries}_error.txt").write_text(errors_text); continue
                (self.log_dir / f"{tag}_{st}_try{tries}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
                cand = self._merge(program, st, out)
                rep = validate(cand)
                if rep["ok"]: program = cand; ok = True
                else: errors_text = "Your previous output had these errors; fix them and output the complete JSON for this stage again:\n" + format_errors(rep)
            info["stages"][st] = {"ok": ok, "tries": tries}
            if not ok:
                info["fallbacks"].append(st)
                program = self._fallback(program, st, ev, clip)
        # final safety
        rep = validate(program)
        if not rep["ok"]:
            info["fallbacks"].append("full_direct"); program = evidence_to_program(ev, clip, self.cfg); rep = validate(program)
        info["final_validation"] = {"ok": rep["ok"], "n_errors": len(rep["errors"]), "warnings": [w["code"] for w in rep["warnings"]]}
        return program, info

    def _user_text(self, st: str, ev_text: str, program: dict, errors_text: str, ev: dict) -> str:
        parts = [f"Clip duration {program['meta']['duration']:.2f} s. Key frames are attached in time order at t = " + ", ".join(f"{k['t']:.2f}" for k in ev["keyframes"]) + " s.", "EVIDENCE:\n" + ev_text]
        if st in ("objects", "motion"): parts.append("CURRENT PROGRAM (camera + static" + (" + objects" if st == "motion" else "") + "):\n" + json.dumps({k: program[k] for k in ("camera", "static") + (("objects",) if st == "motion" else ())}, ensure_ascii=False))
        if st == "camera_static":
            parts.append("Suggested camera keyframes from evidence (you may copy them):\n" + json.dumps(camera_keyframes_from_evidence(ev)))
            reserved = [o["id"] for o in ev["objects"] if o["motion_guess"].get("type") != "static" or not any(k in o["class_guess"].lower() for k in self.cfg["perception"].get("static_classes", []))]
            parts.append("Reserved object ids (these are handled in stage 2 as objects; do NOT put them in static): " + json.dumps(reserved) + "\nstatic is only for fixed structure: ground, floors, walls, tables, tracks, belts, ledges, steps.")
        if errors_text: parts.append(errors_text)
        return "\n\n".join(parts)

    def _merge(self, program: dict, st: str, out: dict) -> dict:
        p = copy.deepcopy(program)
        if st == "camera_static":
            p["camera"] = out.get("camera", {}); p["static"] = out.get("static", []); p["style"] = out.get("style", p["style"])
        elif st == "objects":
            objs = [{k: v for k, v in o.items() if k not in ("motion", "events")} for o in out.get("objects", [])]
            # normalise ids the model suffixed to avoid clashes (lift_1 -> lift) and drop static duplicates of the same thing
            static_ids = {n["id"] for n in p["static"]}
            for o in objs:
                base = re.sub(r"_\d+$", "", o["id"])
                if o["id"] not in static_ids and base != o["id"] and base in static_ids and not any(x["id"] == base for x in objs): o["id"] = base
            obj_ids = {o["id"] for o in objs}
            p["static"] = [n for n in p["static"] if n["id"] not in obj_ids]
            p["objects"] = objs
        elif st == "motion":
            motions = out.get("motions", {})
            for o in p["objects"]:
                base = re.sub(r"_\d+$", "", o["id"])
                m = motions.get(o["id"]) or motions.get(base) or next((v for k, v in motions.items() if o["id"].startswith(k) or k.startswith(o["id"])), None)
                if m:
                    if m.get("motion") and m["motion"].get("type") != "static": o["motion"] = m["motion"]
                    if m.get("events"): o["events"] = m["events"]
                    if m.get("notes"): o["notes"] = (o.get("notes", "") + " " + m["notes"]).strip()[:300]
        elif st == "single_stage":
            for k in ("style", "camera", "static", "objects"):
                if k in out: p[k] = out[k]
        return p

    def _fallback(self, program: dict, st: str, ev: dict, clip: str) -> dict:
        d = evidence_to_program(ev, clip, self.cfg); p = copy.deepcopy(program)
        if st in ("camera_static", "single_stage"): p["camera"], p["static"], p["style"] = d["camera"], d["static"], d["style"]
        if st in ("objects", "single_stage"): p["objects"] = [{k: v for k, v in o.items() if k != "motion"} for o in d["objects"]]
        if st in ("motion", "single_stage"):
            dm = {o["id"]: o.get("motion") for o in d["objects"]}
            for o in p["objects"]:
                if dm.get(o["id"]): o["motion"] = dm[o["id"]]
        return p
