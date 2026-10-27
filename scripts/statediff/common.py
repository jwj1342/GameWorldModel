"""状态差分预实验（#90 的可行性）共用的零件。

任务：给同一场景的首帧与末帧（rgb）、一句话、程序里已知的部分（meta/style/camera/static），
写出 objects 数组：每个可动物体的几何、t=0 位姿和运动。material 不写，样式半边是给定的。
评测：把 objects 合回已知部分，编译、渲染两端的 id 通道，逐物体和真值掩码比 IoU。

语言描述是**欠定**的：说有哪些东西、各自怎么动，不说数字、轴向、行程。
不然句子把答案漏给模型，基准就退化成句法解析。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.compiler.compile import compile_program            # noqa: E402
from gwm.compiler.ids import decode_id_png, registry_order  # noqa: E402
from gwm.compiler.validate import schema as full_schema     # noqa: E402
from gwm.feedback.render import render as render_passes     # noqa: E402

WIDTH, HEIGHT = 1280, 720
CONTEXT_KEYS = ("meta", "style", "camera", "static")
STRIP_OBJECT_KEYS = ("material", "confidence", "notes")
PROMPTS = REPO / "prompts"

# ---------------------------------------------------------------- 语言模板器
_NAMED = {"red": (200, 40, 40), "orange": (230, 130, 30), "yellow": (220, 210, 50),
          "green": (60, 170, 70), "cyan": (60, 200, 200), "blue": (50, 80, 210),
          "purple": (140, 60, 180), "pink": (230, 120, 180), "brown": (120, 80, 40),
          "white": (235, 235, 235), "gray": (128, 128, 128), "black": (25, 25, 25)}
_PALETTE_COLOR = {"red", "blue", "green", "yellow", "white", "black", "gold"}
_PALETTE_ADJ = {"grass": "grassy", "stone": "stone", "wood": "wooden", "metal": "metal",
                "concrete": "concrete", "plastic": "plastic", "rubber": "rubber",
                "lava": "glowing", "water": "watery", "glass": "glass", "cardboard": "cardboard"}
_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
VERB_S = {"static": "stays put", "trajectory": "travels along a winding path",
          "revolute": "swings around a hinge", "prismatic": "slides straight along a line",
          "periodic_translate": "bobs back and forth", "periodic_rotate": "rocks back and forth about a pivot",
          "spin": "spins in place", "dynamic": "is let go and falls under gravity"}
VERB_P = {"static": "stay put", "trajectory": "travel along winding paths",
          "revolute": "swing around hinges", "prismatic": "slide straight along lines",
          "periodic_translate": "bob back and forth", "periodic_rotate": "rock back and forth about pivots",
          "spin": "spin in place", "dynamic": "are let go and fall under gravity"}


def _hex_to_name(h) -> str:
    h = str(h or "")
    if not h.startswith("#") or len(h) != 7:
        return ""
    try:
        rgb = (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    except ValueError:
        return ""
    return min(_NAMED, key=lambda k: sum((a - b) ** 2 for a, b in zip(_NAMED[k], rgb)))


def colour_word(material) -> str:
    """材质 → 一个颜色或材料形容词；认不出就空串。"""
    if material is None:
        return ""
    if isinstance(material, dict):
        return _hex_to_name(material.get("base_color"))
    m = str(material)
    if m.startswith("#"):
        return _hex_to_name(m)
    if m in _PALETTE_COLOR:
        return m
    return _PALETTE_ADJ.get(m, "")


def _plural(name: str) -> str:
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    if name.endswith("y") and name[-2:-1] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def _article(name: str) -> str:
    return ("an " if name[:1] in "aeiou" else "a ") + name


def _num(n: int) -> str:
    return _NUM[n] if 0 <= n < len(_NUM) else str(n)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def describe_scene(program: dict) -> str:
    """一句欠定的描述：有哪些物体、各自怎么动。同名同色的物体合成一组，不排序、不指位置。"""
    objs = program.get("objects") or []
    if not objs:
        return "The scene contains no movable objects."
    groups: dict[str, list[str]] = {}
    for o in objs:
        name = (colour_word(o.get("material")) + " " + str(o.get("class") or "object")).strip()
        groups.setdefault(name, []).append(((o.get("motion") or {}).get("type") or "static"))
    inventory = [_article(n) if len(k) == 1 else f"{_num(len(k))} {_plural(n)}" for n, k in groups.items()]
    head = f"The scene contains {_num(len(objs))} object{'s' if len(objs) != 1 else ''}: {_join(inventory)}."
    parts = []
    for name, kinds in groups.items():
        if len(kinds) == 1:
            parts.append(f"The {name} {VERB_S.get(kinds[0], 'moves')}.")
        elif len(set(kinds)) == 1:
            who = "Both" if len(kinds) == 2 else "All " + _num(len(kinds))
            parts.append(f"{who} {_plural(name)} {VERB_P.get(kinds[0], 'move')}.")
        else:
            counts: dict[str, int] = {}
            for k in kinds:
                counts[k] = counts.get(k, 0) + 1
            clauses = [f"one {VERB_S.get(k, 'moves')}" if c == 1 else f"{_num(c)} {VERB_P.get(k, 'move')}"
                       for k, c in counts.items()]
            parts.append(f"Of the {_num(len(kinds))} {_plural(name)}, {_join(clauses)}.")
    return head + " " + " ".join(parts)


# ---------------------------------------------------------------- 上下文 / 目标 / 提示词
TASK = """
Task. You see two rendered frames of the same scene: frame 1 at t=0 and frame 2 at t=T, where T is meta.duration in seconds. A one-sentence description says what happens in between; it names the objects and how each one moves, but gives no numbers. The program's meta, style, camera and static parts are given below and must not be changed. Write only the "objects" array: one entry per movable object, with a primitive geom (sizes in metres), the pose at t=0 (for periodic_translate / periodic_rotate, pose.pos is the oscillation centre and phase aligns t=0), and the motion, so that the scene matches frame 1 at t=0 and frame 2 at t=T. Do not write material (the style half is given), confidence or notes. Output {"objects":[...]} as JSON only.
"""


def system_prompt() -> str:
    return (PROMPTS / "common_dsl.md").read_text(encoding="utf-8") + "\n" + TASK.strip()


def user_text(context: dict, sentence: str) -> str:
    duration = float((context.get("meta") or {}).get("duration", 8.0))
    return ("Known parts of the program:\n" + json.dumps(context, separators=(",", ":"), ensure_ascii=False)
            + f"\n\nDescription: {sentence}\nFrame 1 is t=0 and frame 2 is t={duration:g}. "
            + 'Output {"objects":[...]} only.')


def context_of(program: dict) -> dict:
    return {k: copy.deepcopy(program[k]) for k in CONTEXT_KEYS if k in program}


def target_objects(program: dict) -> list[dict]:
    return [{k: v for k, v in o.items() if k not in STRIP_OBJECT_KEYS} for o in (program.get("objects") or [])]


def target_text(program: dict) -> str:
    return json.dumps({"objects": target_objects(program)}, separators=(",", ":"), ensure_ascii=False)


def objects_schema() -> dict:
    """给提示基线用的 response_format：objects 数组，带 motion，不带 material。"""
    s = full_schema()
    d = s["$defs"]
    obj = copy.deepcopy(d["objectNode"])
    for k in STRIP_OBJECT_KEYS:
        obj.get("properties", {}).pop(k, None)
    return {"$defs": d, "type": "object", "additionalProperties": False, "required": ["objects"],
            "properties": {"objects": {"type": "array", "items": obj}}}


def extract_objects(parsed):
    if isinstance(parsed, dict) and isinstance(parsed.get("objects"), list):
        return [o for o in parsed["objects"] if isinstance(o, dict)]
    if isinstance(parsed, list) and parsed and all(isinstance(o, dict) for o in parsed):
        return parsed
    return None


def merge_prediction(context: dict, objects: list[dict]) -> dict:
    program = copy.deepcopy(context)
    program.setdefault("meta", {"clip": "pred"})
    program.setdefault("static", [])
    program["objects"] = copy.deepcopy(objects)
    return program


# ---------------------------------------------------------------- 渲染 / 掩码 / 配对
def render_endpoints(program: dict, work: str | Path, times: list[float], cfg: dict | None,
                     width: int = WIDTH, height: int = HEIGHT) -> dict:
    """编译 + 渲染给定时刻的 rgb 与 id 通道。失败时说明死在哪一步。"""
    work = Path(work)
    res = compile_program(program, work / "game", cfg)
    if not res["ok"]:
        return {"ok": False, "stage": "compile", "validation": res["validation"]}
    try:
        index = render_passes(res["game_dir"], times, work / "render", width, height, passes=("rgb", "id"))
    except (RuntimeError, OSError) as e:
        return {"ok": False, "stage": "render", "error": str(e)[-800:]}
    files = {round(float(fr["requested_t"]), 3): {p: str(work / "render" / f) for p, f in fr["files"].items()}
             for fr in index["frames"]}
    return {"ok": True, "files": files, "time_sampling": index.get("time_sampling")}


def object_masks(program: dict, id_png: str | Path) -> dict[str, np.ndarray]:
    reg = registry_order(program)
    masks = decode_id_png(id_png, reg)
    return {r["id"]: masks[r["id"]] for r in reg if r["kind"] == "object"}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / union if union else 0.0


def match_masks(gt: dict[str, np.ndarray], pred: dict[str, np.ndarray]) -> dict[str, tuple[str | None, float]]:
    """匈牙利配对，最大化 IoU 之和。完全不重叠的真值物体配 None。"""
    from scipy.optimize import linear_sum_assignment
    g_ids, p_ids = list(gt), list(pred)
    out: dict[str, tuple[str | None, float]] = {g: (None, 0.0) for g in g_ids}
    if not g_ids or not p_ids:
        return out
    m = np.zeros((len(g_ids), len(p_ids)))
    for i, g in enumerate(g_ids):
        for j, p in enumerate(p_ids):
            m[i, j] = iou(gt[g], pred[p])
    rows, cols = linear_sum_assignment(-m)
    for i, j in zip(rows, cols):
        if m[i, j] > 0:
            out[g_ids[i]] = (p_ids[j], float(m[i, j]))
    return out


def axis_of(motion) -> np.ndarray | None:
    if not isinstance(motion, dict):
        return None
    a = motion.get("axis")
    if not (isinstance(a, list) and len(a) == 3):
        return None
    try:
        v = np.array([float(x) for x in a])
    except (TypeError, ValueError):
        return None
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = abs(float(np.clip(np.dot(a, b), -1.0, 1.0)))     # 轴不分正负
    return float(np.degrees(np.arccos(c)))


def load_samples(path: str | Path, sets: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not sets or r["set"] in sets]


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
