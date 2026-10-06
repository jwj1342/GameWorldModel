#!/usr/bin/env python3
"""把一句自然语言的修改要求，翻译成对场景程序的 JSON Patch 并应用。

这是 issue #13 可编辑性实验的执行入口。论文的核心主张之一是我们的产物
可以被读、被改、被重新组合，而三维重建和视频世界模型做不到这件事。
这个脚本负责把那个主张变成可以量化的实验。
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

import jsonpatch

from gwm.config import REPO, load_config
from gwm.compiler.validate import validate

EDIT_SCHEMA = {
    "type": "object", "required": ["ops", "note"], "additionalProperties": False,
    "properties": {
        "ops": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "required": ["op", "path"], "additionalProperties": False,
            "properties": {"op": {"type": "string", "enum": ["replace", "add", "remove"]},
                           "path": {"type": "string"}, "value": {}}}},
        "note": {"type": "string"},
    },
}


def outline(program: dict) -> str:
    """给模型一份紧凑的程序结构，而不是整份 JSON，省 token 也少干扰。"""
    lines = []
    for i, s in enumerate(program.get("static", [])):
        lines.append(f"/static/{i}  id={s['id']}  class={s.get('class')}")
    for i, o in enumerate(program.get("objects", [])):
        m = o.get("motion") or {}
        params = {k: v for k, v in m.items() if k != "type"}
        lines.append(f"/objects/{i}  id={o['id']}  class={o.get('class')}  "
                     f"motion={m.get('type', 'static')} {json.dumps(params, ensure_ascii=False)[:120]}")
    return "\n".join(lines)


def changed_paths(before: dict, after: dict) -> list[str]:
    """枚举两份程序之间所有实际不同的叶子路径，用来量化副作用。"""
    out = []

    def walk(a, b, path):
        if type(a) is not type(b):
            out.append(path); return
        if isinstance(a, dict):
            for k in set(a) | set(b):
                if k not in a or k not in b: out.append(f"{path}/{k}")
                else: walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            if len(a) != len(b): out.append(path); return
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}/{i}")
        elif a != b:
            out.append(path)

    walk(before, after, "")
    return sorted(out)


def apply_edit(program: dict, instruction: str, client) -> dict:
    """返回 {ops, note, program, ok, error, changed}。失败时 program 保持原样。"""
    system = (REPO / "prompts" / "editor.md").read_text() + "\n\n" + (REPO / "prompts" / "common_dsl.md").read_text()
    user = f"程序结构：\n{outline(program)}\n\n修改要求：{instruction}"
    r = client.chat(system, user, json_schema=EDIT_SCHEMA, temperature=0.0)
    got = r.get("json") or {}
    ops, note = got.get("ops") or [], got.get("note", "")
    if not ops:
        return {"ops": [], "note": note, "program": program, "ok": False, "error": "模型没有给出任何操作", "changed": []}
    try:
        after = jsonpatch.JsonPatch(ops).apply(copy.deepcopy(program))
    except Exception as e:
        return {"ops": ops, "note": note, "program": program, "ok": False, "error": f"补丁应用失败: {e}", "changed": []}
    v = validate(after)
    if not v["ok"]:
        return {"ops": ops, "note": note, "program": program, "ok": False,
                "error": "改完之后不符合 schema: " + "; ".join(e.get("message", "") for e in v["errors"][:3]), "changed": []}
    return {"ops": ops, "note": note, "program": after, "ok": True, "error": "", "changed": changed_paths(program, after)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--out", default=None, help="改完的程序写到哪；不给就只打印")
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args(argv)

    from gwm.config import site_name
    cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml",
                       *[REPO / c for c in a.config]])
    from gwm.synthesis.vlm import make_client
    client = make_client(cfg)

    program = json.loads(Path(a.program).read_text())
    r = apply_edit(program, a.instruction, client)
    print(json.dumps({k: v for k, v in r.items() if k != "program"}, ensure_ascii=False, indent=1))
    if r["ok"] and a.out:
        Path(a.out).write_text(json.dumps(r["program"], ensure_ascii=False, indent=1) + "\n")
        print(f"写到 {a.out}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
