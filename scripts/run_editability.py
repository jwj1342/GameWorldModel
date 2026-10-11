#!/usr/bin/env python3
"""跑一遍编辑指令集，统计指令执行成功率和副作用率。

issue #13。论文主张产物可编辑，这个脚本把主张变成两个数字：
  执行成功率 —— 指令要改的地方确实改了
  副作用率   —— 指令没提到的地方有没有被动过
另外单独统计「应当拒绝」那几条，看模型会不会为了交差而编造改动。
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
import jsonpatch
import jsonpointer

from gwm.config import REPO, load_config, site_name
sys.path.insert(0, str(REPO / "scripts"))
from edit_program import apply_edit, changed_paths


def program_digest(program: dict) -> str:
    """对解析后的 JSON 排序键再哈希；空白不影响，数组顺序和初值影响。"""
    payload = json.dumps(program, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def baseline_error(program: dict, expected_hash: str | None) -> str:
    if not expected_hash:
        return "无法评测：参考编辑缺少 program_sha256 基线"
    try:
        actual = program_digest(program)
    except (TypeError, ValueError):
        return "无法评测：原始 Program 不能生成有限 JSON 基线"
    return "" if actual == expected_hash else "无法评测：原始 Program 与参考编辑的 program_sha256 不匹配"


def equivalent(actual, expected, tolerances=None, path="") -> bool:
    """完整结构比较；仅指定 JSON Pointer 数值字段可覆盖默认绝对容差。"""
    numeric = (int, float)
    if type(actual) in numeric and type(expected) in numeric:
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= (tolerances or {}).get(path, 1e-6)
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equivalent(actual[k], v, tolerances, path + "/" + k.replace("~", "~0").replace("/", "~1"))
            for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equivalent(a, b, tolerances, path + "/" + str(i)) for i, (a, b) in enumerate(zip(actual, expected)))
    return actual == expected


def judge(case: dict, r: dict) -> dict:
    """把一次编辑的结果判成通过或失败，并说明理由。"""
    if "expect_ops" in case or case.get("program_sha256"):
        error = baseline_error(r.get("before"), case.get("program_sha256"))
        if error:
            return {"ok": False, "why": error}
    if case.get("expect_refuse"):
        ok = r.get("status") == "refused" and not r["ops"] and bool(r.get("note", "").strip())
        if ok: return {"ok": True, "why": "正确拒绝"}
        # 判据用 #39 那版（光是没产出操作不够，得明确拒绝并给理由），
        # 理由沿用 #28 那版：编造了什么路径要写出来，不然查不动。
        paths = [o.get("path") for o in r["ops"]]
        return {"ok": False, "why": f"应当拒绝却编造了 {len(r['ops'])} 个操作：{paths}"
                if r["ops"] else "未明确拒绝，或响应/执行失败"}

    if "expect_ops" in case:
        if "before" not in r:
            return {"ok": False, "why": "无法评测：缺少编辑前 Program"}
        try:
            expected = jsonpatch.JsonPatch(case["expect_ops"]).apply(copy.deepcopy(r["before"]))
        except (jsonpatch.JsonPatchException, jsonpointer.JsonPointerException, TypeError, KeyError, IndexError):
            return {"ok": False, "why": "无法评测：参考补丁与输入不匹配"}
        tolerances = case.get("tolerance", {})
        if not isinstance(tolerances, dict):
            return {"ok": False, "why": "无法评测：tolerance 必须是字段路径到绝对容差的映射"}
        editable = {op.get("path") for op in case["expect_ops"] if op.get("op") in ("add", "replace")}
        for path, value in tolerances.items():
            try:
                target = jsonpointer.resolve_pointer(expected, path)
                valid = (path in editable and type(value) in (int, float) and math.isfinite(value)
                         and value >= 0 and type(target) in (int, float) and math.isfinite(target))
            except (jsonpointer.JsonPointerException, TypeError):
                valid = False
            if not valid:
                return {"ok": False, "why": "无法评测：容差必须有限、非负且只针对参考编辑的数值字段"}
        if r.get("status") not in ("applied", "no_change"):
            return {"ok": False, "why": r.get("error") or "编辑没有成功执行"}
        if not equivalent(r["program"], expected, tolerances):
            return {"ok": False, "why": f"目标值/对象错误或有副作用：{changed_paths(expected, r['program'])}"}
        return {"ok": True, "why": "原输入已经满足要求" if equivalent(r["before"], expected, tolerances) else "完整结果符合参考编辑，无额外副作用"}

    if not r["ok"]:
        return {"ok": False, "why": r["error"]}

    changed = r["changed"]
    wanted = case["expect_paths"]
    hit = [p for p in changed if any(p == w or p.startswith(w + "/") for w in wanted)]
    extra = [p for p in changed if p not in hit]

    if not hit:
        return {"ok": False, "why": f"该改的没改。期望命中 {wanted}，实际改了 {changed}"}
    if extra:
        return {"ok": False, "why": f"有副作用，额外改了 {extra}"}

    if not case.get("expect_value"):
        return {"ok": False, "why": "无法评测：只有路径约束，没有目标值或参考补丁"}
    if "before" not in r:
        return {"ok": False, "why": "无法评测：缺少编辑前 Program，不能核对副作用"}
    expected = copy.deepcopy(r["before"])
    for path, want in case["expect_value"].items():
        try:
            expected = jsonpatch.JsonPatch([{"op": "replace", "path": path, "value": want}]).apply(expected)
        except (jsonpatch.JsonPatchException, jsonpointer.JsonPointerException, TypeError, KeyError, IndexError):
            return {"ok": False, "why": f"无法评测：原始 Program 缺少或不支持 {path}"}
    if not equivalent(r["program"], expected):
        return {"ok": False, "why": f"目标值错误或有副作用：{changed_paths(expected, r['program'])}"}

    return {"ok": True, "why": "改对了，且没有副作用"}


def evaluate_cases(program: dict, cases: list[dict], client, *, program_sha256: str | None = None) -> list[dict]:
    """每条从同一输入出发；调用失败也保留一行，不会被记为正确拒绝。"""
    rows = []
    for case in cases:
        case = {**case, **({"program_sha256": program_sha256} if program_sha256 is not None else {})}
        if "expect_ops" in case or case.get("program_sha256"):
            error = baseline_error(program, case.get("program_sha256"))
            if error:
                rows.append({**{k: case[k] for k in ("id", "kind", "instruction")},
                             "ok": False, "why": error, "ops": [], "changed": [], "note": "",
                             "status": "input_mismatch", "error": error})
                continue
        r = apply_edit(program, case["instruction"], client)
        v = judge(case, r)
        rows.append({**{k: case[k] for k in ("id", "kind", "instruction")},
                     "ok": v["ok"], "why": v["why"], "ops": r["ops"],
                     "changed": r["changed"], "note": r["note"], "status": r["status"],
                     "error": r["error"]})
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", default="experiments/editability/instructions.json")
    ap.add_argument("--out", default="docs/results/editability.json", help="结果写到哪，默认和其它实验产物放一起")
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args(argv)

    spec = json.loads((REPO / a.instructions).read_text(encoding="utf-8"))
    program = json.loads((REPO / spec["program"]).read_text(encoding="utf-8"))
    expected_hash = spec.get("program_sha256")
    if expected_hash or any("expect_ops" in c for c in spec["cases"]):
        error = baseline_error(program, expected_hash)
        if error:
            print(error, file=sys.stderr)
            return 2  # 在配置或客户端初始化前拒绝失效的参考，避免无意义调用。
    cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml",
                       *[REPO / c for c in a.config]])
    from gwm.synthesis.vlm import make_client
    client = make_client(cfg)

    t0 = time.time()
    rows = evaluate_cases(program, spec["cases"], client, program_sha256=expected_hash)
    for row in rows:
        print(f"  [{'通过' if row['ok'] else '失败'}] {row['id']:28s} {row['why'][:90]}")

    normal = [x for x in rows if not any(c.get("expect_refuse") and c["id"] == x["id"] for c in spec["cases"])]
    refuse = [x for x in rows if x not in normal]
    print()
    print(f"  可执行指令 {sum(x['ok'] for x in normal)}/{len(normal)} 通过")
    print(f"  应当拒绝的 {sum(x['ok'] for x in refuse)}/{len(refuse)} 正确拒绝")
    print(f"  总计 {sum(x['ok'] for x in rows)}/{len(rows)}，耗时 {time.time()-t0:.0f}s，模型调用 {client.calls} 次")

    if a.out:
        out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "model": client.model, "judgment": "reference_program",
                                           "program_sha256": program_digest(program),
                                           "passed": sum(x["ok"] for x in rows), "total": len(rows)},
                                          ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"  明细写到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
