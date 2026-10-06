#!/usr/bin/env python3
"""跑一遍编辑指令集，统计指令执行成功率和副作用率。

issue #13。论文主张产物可编辑，这个脚本把主张变成两个数字：
  执行成功率 —— 指令要改的地方确实改了
  副作用率   —— 指令没提到的地方有没有被动过
另外单独统计「应当拒绝」那几条，看模型会不会为了交差而编造改动。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

from gwm.config import REPO, load_config, site_name
sys.path.insert(0, str(REPO / "scripts"))
from edit_program import apply_edit


def judge(case: dict, r: dict) -> dict:
    """把一次编辑的结果判成通过或失败，并说明理由。"""
    if case.get("expect_refuse"):
        ok = not r["ops"]
        if ok: return {"ok": True, "why": "正确拒绝"}
        paths = [o.get("path") for o in r["ops"]]
        return {"ok": False, "why": f"应当拒绝却编造了 {len(r['ops'])} 个操作：{paths}"}

    if not r["ok"]:
        return {"ok": False, "why": r["error"]}

    changed = r["changed"]
    wanted = case["expect_paths"]
    hit = [p for p in changed if any(p.startswith(w) for w in wanted)]
    extra = [p for p in changed if p not in hit]

    if not hit:
        return {"ok": False, "why": f"该改的没改。期望命中 {wanted}，实际改了 {changed}"}
    if extra:
        return {"ok": False, "why": f"有副作用，额外改了 {extra}"}

    for path, want in (case.get("expect_value") or {}).items():
        cur = r["program"]
        for seg in path.strip("/").split("/"):
            cur = cur[int(seg)] if seg.isdigit() else cur[seg]
        if isinstance(want, (int, float)) and isinstance(cur, (int, float)):
            if abs(cur - want) > 1e-6: return {"ok": False, "why": f"{path} 期望 {want}，实际 {cur}"}
        elif cur != want:
            return {"ok": False, "why": f"{path} 期望 {want}，实际 {cur}"}

    return {"ok": True, "why": "改对了，且没有副作用"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", default="experiments/editability/instructions.json")
    ap.add_argument("--out", default="docs/results/editability.json", help="结果写到哪，默认和其它实验产物放一起")
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args(argv)

    spec = json.loads((REPO / a.instructions).read_text())
    program = json.loads((REPO / spec["program"]).read_text())
    cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml",
                       *[REPO / c for c in a.config]])
    from gwm.synthesis.vlm import make_client
    client = make_client(cfg)

    rows, t0 = [], time.time()
    for case in spec["cases"]:
        r = apply_edit(program, case["instruction"], client)       # 每条都从原始程序出发，互不影响
        v = judge(case, r)
        rows.append({**{k: case[k] for k in ("id", "kind", "instruction")},
                     "ok": v["ok"], "why": v["why"], "ops": r["ops"], "changed": r["changed"], "note": r["note"]})
        print(f"  [{'通过' if v['ok'] else '失败'}] {case['id']:28s} {v['why'][:90]}")

    normal = [x for x in rows if not any(c.get("expect_refuse") and c["id"] == x["id"] for c in spec["cases"])]
    refuse = [x for x in rows if x not in normal]
    print()
    print(f"  可执行指令 {sum(x['ok'] for x in normal)}/{len(normal)} 通过")
    print(f"  应当拒绝的 {sum(x['ok'] for x in refuse)}/{len(refuse)} 正确拒绝")
    print(f"  总计 {sum(x['ok'] for x in rows)}/{len(rows)}，耗时 {time.time()-t0:.0f}s，模型调用 {client.calls} 次")

    if a.out:
        out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "model": client.model,
                                           "passed": sum(x["ok"] for x in rows), "total": len(rows)},
                                          ensure_ascii=False, indent=1) + "\n")
        print(f"  明细写到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
