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
import jsonschema

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
        "refused": {"type": "boolean"},
    },
}


def outline(program: dict) -> str:
    """保留编辑所需的值和数组顺序；不截断相对编辑的输入。"""
    return json.dumps(program, ensure_ascii=False, separators=(",", ":"))


def changed_paths(before: dict, after: dict) -> list[str]:
    """枚举两份程序之间所有实际不同的叶子路径，用来量化副作用。"""
    out = []

    def walk(a, b, path):
        if type(a) is not type(b):
            out.append(path); return
        if isinstance(a, dict):
            for k in set(a) | set(b):
                child = f"{path}/{str(k).replace('~', '~0').replace('/', '~1')}"
                if k not in a or k not in b: out.append(child)
                else: walk(a[k], b[k], child)
        elif isinstance(a, list):
            if len(a) != len(b): out.append(path); return
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}/{i}")
        elif a != b:
            out.append(path)

    walk(before, after, "")
    return sorted(out)


def apply_edit(program: dict, instruction: str, client) -> dict:
    """保留旧返回字段，增加 status 和 before；失败不修改输入，不中断批次。"""
    before = copy.deepcopy(program)
    def result(status, *, ops=None, note="", error="", after=None):
        output = copy.deepcopy(before) if after is None else after
        return {"ops": ops or [], "note": note, "program": output,
                "before": before, "status": status, "ok": status == "applied",
                "error": error, "changed": changed_paths(before, output)}

    system = (REPO / "prompts" / "editor.md").read_text(encoding="utf-8") + "\n\n" + (REPO / "prompts" / "common_dsl.md").read_text(encoding="utf-8")
    user = f"程序结构：\n{outline(program)}\n\n修改要求：{instruction}"
    try:
        r = client.chat(system, user, json_schema=EDIT_SCHEMA, temperature=0.0)
    except Exception as e:
        return result("execution_failed", error=f"编辑调用失败 ({type(e).__name__})")
    got = r.get("json") if isinstance(r, dict) else None
    errors = list(jsonschema.Draft202012Validator(EDIT_SCHEMA).iter_errors(got))
    if errors:
        return result("invalid_response", error="编辑响应不符合 EDIT_SCHEMA")
    ops, note = got.get("ops") or [], got.get("note", "")
    if got.get("refused"):
        if ops or not note.strip():
            return result("invalid_response", ops=ops, note=note, error="拒绝必须无操作且给出原因")
        return result("refused", note=note, error="明确拒绝编辑")
    if not ops:
        return result("no_change", note=note, error="模型没有给出任何操作（不是明确拒绝）")
    try:
        after = jsonpatch.JsonPatch(ops).apply(copy.deepcopy(before))
    except Exception as e:
        return result("invalid_patch", ops=ops, note=note, error=f"补丁应用失败 ({type(e).__name__})")
    v = validate(after)
    if not v["ok"]:
        return result("invalid_program", ops=ops, note=note,
                      error="改完之后不符合 schema: " + "; ".join(e.get("message", "") for e in v["errors"][:3]))
    return result("applied", ops=ops, note=note, after=after)


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

    program = json.loads(Path(a.program).read_text(encoding="utf-8"))
    r = apply_edit(program, a.instruction, client)
    print(json.dumps({k: v for k, v in r.items() if k not in ("program", "before")}, ensure_ascii=False, indent=1))
    if r["ok"] and a.out:
        Path(a.out).write_text(json.dumps(r["program"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"写到 {a.out}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
