"""铰接探针的任务定义：首末两帧 + 一句话 + 已知的几何 → 每个部件的关节。

和 common.py 的预实验任务并列。区别是物体的几何和 t=0 位姿给定，模型只写 motion：
哪个部件动、什么关节、轴和铰链在哪、开多大。像素到世界坐标的放置那一半不挡路，
测的正好是 #90 的主张所在的那一半。样本行里 task 为 "art" 的走这里。
"""
from __future__ import annotations

import copy
import json

from gwm.compiler.validate import schema as full_schema

TASK = """You write the motion half of a 4D scene program: JSON that a fixed three.js runtime executes. Units are metres, y is up, the ground is the plane y=0, time t is in seconds.

You see two frames of the same piece of furniture from a fixed camera: frame 1 at t=0 with every part closed, frame 2 at t=T (T = meta.duration) after some parts have opened. A one-sentence description says which kinds of parts move, but not which ones, where their hinges are, or how far they open. The program's meta, camera, static parts, and every object's geometry, colour and t=0 pose are given and must not be changed.

For every object id, write its motion:
- {"type":"static"} if it does not move.
- {"type":"revolute","axis":[x,y,z],"pivot":[x,y,z],"range_deg":[0,A],"from_deg":0,"schedule":[{"t":t0,"to_deg":A,"duration":d}]} for a part that swings about a hinge. axis is a unit vector in world coordinates and pivot is any world point on the hinge line. A point p of the part at angle a is at pivot + Rot(axis, a)(p - pivot), counter-clockwise about axis by the right-hand rule, so the sign of axis decides which way it swings. The part must have reached angle A by t=T.
- {"type":"prismatic","axis":[x,y,z],"range":[0,s],"from":0,"schedule":[{"t":t0,"to":s,"duration":d}]} for a part that slides. A point p is at p + axis*distance, axis a unit vector in world coordinates, s in metres, reached by t=T.
Quaternions are [x,y,z,w]. A box's extent is its size along its own x, y, z axes, before the rotation pose.quat is applied; pose.pos is its centre.
Output {"motions":{"<id>":{...}, ...}} with one entry per object id, as JSON only."""


def system_prompt() -> str:
    return TASK


def context_of(program: dict) -> dict:
    ctx = {k: copy.deepcopy(program[k]) for k in ("meta", "camera", "static") if k in program}
    ctx["objects"] = [{k: v for k, v in o.items() if k != "motion"} for o in program.get("objects") or []]
    return ctx


def user_text(context: dict, sentence: str) -> str:
    duration = float((context.get("meta") or {}).get("duration", 4.0))
    return ("Known parts of the program:\n" + json.dumps(context, separators=(",", ":"), ensure_ascii=False)
            + f"\n\nDescription: {sentence}\nFrame 1 is t=0 and frame 2 is t={duration:g}. "
            + 'Output {"motions":{...}} only.')


def target_text(program: dict) -> str:
    return json.dumps({"motions": {o["id"]: o.get("motion") or {"type": "static"} for o in program.get("objects") or []}},
                      separators=(",", ":"), ensure_ascii=False)


def motions_schema() -> dict:
    s = full_schema()
    return {"$defs": s["$defs"], "type": "object", "additionalProperties": False, "required": ["motions"],
            "properties": {"motions": {"type": "object", "additionalProperties": {"$ref": "#/$defs/motion"}}}}


def extract_motions(parsed) -> dict | None:
    if isinstance(parsed, dict) and isinstance(parsed.get("motions"), dict):
        m = parsed["motions"]
        return m if all(isinstance(v, dict) for v in m.values()) else None
    return None


def merge_motions(program: dict, motions: dict) -> tuple[dict, int]:
    """真值程序换上预测的 motion。没写到的部件按不动处理，返回漏写的个数。"""
    out = copy.deepcopy(program)
    missing = 0
    for o in out["objects"]:
        if o["id"] in motions:
            o["motion"] = copy.deepcopy(motions[o["id"]])
        else:
            o["motion"] = {"type": "static"}
            missing += 1
    return out, missing
