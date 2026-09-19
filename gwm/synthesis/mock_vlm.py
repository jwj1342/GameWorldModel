"""A deterministic stand-in for the VLM: parses the evidence summary in the prompt and returns schema-valid stage outputs.
Used to exercise writer / critic / loop plumbing on CPU without a model (configs/vlm/mock.yaml)."""
from __future__ import annotations
import json, re

class MockVLMClient:
    model = "mock"
    def __init__(self, cfg: dict, log_path=None):
        self.calls = 0; self.prompt_tokens = 0; self.completion_tokens = 0
    def chat(self, system, user_text, images=None, json_schema=None, temperature=None, max_tokens=None, seed=None) -> dict:
        self.calls += 1
        if "Stage 1 of 3" in system: out = self._stage1(user_text)
        elif "Stage 2 of 3" in system: out = self._stage2(user_text)
        elif "Stage 3 of 3" in system: out = self._stage3(user_text)
        elif "whole scene program in one shot" in system: out = {**self._stage1(user_text), **self._stage2(user_text)}
        elif "compare a rendered 3D scene" in system: out = self._critic(user_text)
        elif "noun phrases" in system: out = {"objects": [{"phrase": "toy train", "size_m": [0.2, 0.1, 0.1], "moving": True}, {"phrase": "floor", "size_m": [3, 0.1, 3], "moving": False}]}
        elif "automated playthrough" in system: out = {"related_to_video": True, "playable": True, "defect": "mock review"}
        else: out = {}
        return {"text": json.dumps(out), "json": out, "usage": None, "finish": "stop"}

    @staticmethod
    def _objects(user_text: str) -> list[dict]:
        objs = []
        for m in re.finditer(r"- id=(\S+) class_guess='([^']*)' conf=([\d.]+) dynamic=(True|False) size_first=(\[[^\]]*\]) quat_first=(\[[^\]]*\]).*?\n\s*motion_guess: (.*?)\n\s*centres: (.*?)(?=\n- id=|\n*$)", user_text, flags=re.S):
            oid, cls, conf, dyn, size, quat, mg, centres = m.groups()
            cs = [(float(t), json.loads(c)) for t, c in re.findall(r"t=([\d.]+) c=(\[[^\]]*\])", centres)]
            mgd = dict(re.findall(r"(\w+)=([^,()]+)", mg))
            objs.append({"id": oid, "class": cls, "conf": float(conf), "dynamic": dyn == "True", "size": json.loads(size), "quat": json.loads(quat), "centres": cs, "mg": mgd})
        return objs

    def _stage1(self, user_text: str) -> dict:
        kf = re.search(r"Suggested camera keyframes from evidence \(you may copy them\):\n(\[.*?\])\n", user_text, flags=re.S)
        g = re.search(r"ground plane: y=0, centre_hint=(\[[^\]]*\]), extent_hint=(\[[^\]]*\])", user_text)
        fov = re.search(r"fov_deg=([\d.]+), aspect=([\d.]+)", user_text)
        centre = json.loads(g.group(1)) if g else [0, 0, 0]; ext = json.loads(g.group(2)) if g else [10, 0.2, 10]
        return {"style": {"background": "#8fb3d9"},
                "camera": {"intrinsics": {"fov_deg": float(fov.group(1)) if fov else 60, "aspect": float(fov.group(2)) if fov else 1.777, "far": 100}, "keyframes": json.loads(kf.group(1)) if kf else [{"t": 0, "pos": [0, 2, 6], "quat": [0, 0, 0, 1]}], "interp": "catmull_rom"},
                "static": [{"id": "ground", "class": "ground", "geom": {"kind": "primitive", "shape": "box", "extent": [max(ext[0], 2) * 1.4, 0.3, max(ext[2], 2) * 1.4]}, "pose": {"pos": [centre[0], -0.15, centre[2]]}, "material": "grass"}]}

    def _stage2(self, user_text: str) -> dict:
        objs = []
        for o in self._objects(user_text):
            pos = o["centres"][0][1] if o["centres"] else [0, 0.5, 0]
            objs.append({"id": o["id"], "class": o["class"][:40], "confidence": o["conf"], "geom": {"kind": "asset", "query": o["class"][:80], "extent": [max(v, 0.05) for v in o["size"]]}, "pose": {"pos": pos, "quat": o["quat"]}})
        return {"objects": objs}

    def _stage3(self, user_text: str) -> dict:
        motions = {}
        for o in self._objects(user_text):
            t = o["mg"].get("type", "static")
            if t == "periodic_translate" and "period" in o["mg"]:
                motions[o["id"]] = {"motion": {"type": "periodic_translate", "axis": [0, 1, 0], "amp": float(o["mg"].get("amp", 0.5)), "period": float(o["mg"]["period"]), "phase": 0.0}, "events": []}
            elif t == "spin": motions[o["id"]] = {"motion": {"type": "spin", "axis": [0, 1, 0], "rate_dps": 90}, "events": [{"type": "despawn_on_contact", "with": "player"}] if "coin" in o["class"] else []}
            elif t in ("trajectory", "prismatic") and len(o["centres"]) >= 2:
                motions[o["id"]] = {"motion": {"type": "trajectory", "interp": "linear", "keyframes": [{"t": t_, "pos": c} for t_, c in o["centres"][:12]]}, "events": []}
            else: motions[o["id"]] = {"motion": {"type": "static"}, "events": [], "notes": "mock: static"}
        return {"motions": motions}

    def _critic(self, user_text: str) -> dict:
        items = []
        for m in re.finditer(r"OBJECT (\S+) \(program path (/objects/\d+)\)", user_text):
            oid, path = m.groups()
            items.append({"object_id": oid, "issue": "pose", "evidence": "mock: nudge position slightly", "suggested_edit": [{"op": "replace", "path": f"{path}/confidence", "value": 0.5}]})
        return {"items": items[:4], "summary": "mock critic"}
