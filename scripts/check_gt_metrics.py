#!/usr/bin/env python3
"""敏感性检查：把一份正确的程序程序化改坏，看指标掉不掉。

issue #4 要求的第一步。指标写出来了但其实测不到东西，是最尴尬的情况，
这个脚本就是用来挡住它的。对每种扰动检查相关指标实际恶化；
不保证末段误差总比整段更大，也不声称检查了全部不相关指标。

轨迹不在 Python 里重算，两份程序都经 harness 跑出逐帧状态再比，
所以这个脚本需要一个能跑无头浏览器的环境。
"""
from __future__ import annotations
import argparse, copy, json, math, subprocess, sys
from pathlib import Path

from gwm.config import REPO
from gwm.feedback.gt_metrics import evaluate, load_states


def perturb(program: dict, kind: str) -> dict:
    """每种扰动只破坏一个方面，便于判断指标有没有指到正确的地方。"""
    p = copy.deepcopy(program)
    objs = p["objects"]
    if kind == "运动类型改错":           # 期望：运动类型准确率掉
        for obj in objs:
            if (obj.get("motion") or {}).get("type", "static") != "static":
                obj["motion"] = {"type": "static"}
                break
    elif kind == "单个物体平移":          # 期望：轨迹误差涨，运动类型不变
        # 只挪一个。全场景一起挪是规范自由度，评测前会被 align_gauge 对齐掉，
        # 拿它当敏感性探针探的是指标故意不测的量。
        for o in objs:
            if "pose" in o: o["pose"]["pos"][0] += 0.5
            for inst in o.get("instances", []): inst["pos"][0] += 0.5
            break
    elif kind == "整体平移（规范自由度）":  # 期望：轨迹误差**不**涨，这是在验对齐有没有生效
        for o in objs:                    # 物体可以用 pose，也可以用 instances 放多份
            if "pose" in o: o["pose"]["pos"][0] += 0.5
            for inst in o.get("instances", []): inst["pos"][0] += 0.5
    elif kind == "删掉一个物体":          # 期望：召回掉
        gone = objs.pop()["id"]
        # 绑定里还引用着它的话编译过不去，一并清掉，保证这次扰动只影响召回
        slots = p.get("binding", {}).get("slots", {})
        for key in ("collectibles", "hazards"):
            slots[key] = [x for x in slots.get(key, []) if x != gone and not x.startswith(gone + "#")]
    elif kind == "多造一个物体":          # 期望：误检涨（precision 掉）
        extra = copy.deepcopy(objs[0]); extra["id"] = "ghost"
        while extra["id"] in {obj["id"] for obj in objs}: extra["id"] += "_extra"
        placements = [extra["pose"]] if "pose" in extra else extra.get("instances", [])
        for placement in placements:
            placement["pos"] = [x + 6.0 for x in placement["pos"]]
        objs.append(extra)
    elif kind == "周期改错":              # 期望：轨迹误差涨；末段不保证总是更大
        for o in objs:
            m = o.get("motion") or {}
            if m.get("type") == "periodic_translate": m["period"] = float(m["period"]) * 1.25
    else:
        raise ValueError(kind)
    return p


def states_for(program: dict, harness: str, work: Path, name: str, duration: float, fps: int) -> list[dict]:
    """编译并录一遍，拿到逐帧状态。运动语义只有内核这一份实现。"""
    pdir = work / name; pdir.mkdir(parents=True, exist_ok=True)
    pjson = pdir / "program.json"; pjson.write_text(json.dumps(program, ensure_ascii=False))
    game, rec = pdir / "game", pdir / "record"
    subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(pjson), str(game)],
                   check=True, cwd=REPO, capture_output=True)
    subprocess.run([harness, "record.mjs", "--game", str(game), "--out", str(rec),
                    "--fps", str(fps), "--duration", str(duration), "--width", "320", "--height", "180"],
                   check=True, cwd=REPO, capture_output=True)
    return load_states(rec / "gt_states.jsonl")


def sensitivity_check(base: dict, result: dict, kind: str) -> dict:
    """Judge measured degradation, not merely whether recording returned."""
    def number(report, group, key):
        value = report.get("full", {}).get(group, {}).get(key)
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None

    if kind == "基准":
        values = [number(result, "trajectory", "max_m"), number(result, "motion_type", "accuracy"),
                  number(result, "objects", "recall"), number(result, "objects", "precision")]
        passed = (all(value is not None for value in values)
                  and abs(values[0]) <= 1e-8 and all(abs(value - 1) <= 1e-8 for value in values[1:])
                  and result.get("full", {}).get("trajectory", {}).get("status") == "complete")
    else:
        group, key, direction = {
            "运动类型改错": ("motion_type", "accuracy", -1),
            "单个物体平移": ("trajectory", "max_m", 1),
            "整体平移（规范自由度）": ("trajectory", "max_m", 0),
            "删掉一个物体": ("objects", "recall", -1),
            "多造一个物体": ("objects", "precision", -1),
            "周期改错": ("trajectory", "max_m", 1),
        }[kind]
        before, after = number(base, group, key), number(result, group, key)
        if direction == 0:
            # 整体平移是观测不到的规范差异，指标该把它对齐掉。涨了说明对齐没生效，
            # 那是在罚一个视频里确定不了的量；这一项查的是「不该动」。
            passed = before is not None and after is not None and abs(after - before) <= 1e-6
        else:
            passed = before is not None and after is not None and direction * (after - before) > 1e-8
    return {"status": "pass" if passed else "fail",
            "reason": "expected metric response observed" if passed else "metric unavailable or expected response absent"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default="examples/handwritten/program.json")
    ap.add_argument("--harness", default=str(REPO / "scripts" / "node_harness.sh"))
    ap.add_argument("--work", required=True, help="放中间产物的目录，用 $SLURM_TMPDIR")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="docs/results/gt_metrics_sensitivity.json")
    a = ap.parse_args(argv)

    gt = json.loads((REPO / a.program).read_text())
    duration = float(gt.get("meta", {}).get("duration") or 8.0)
    work = Path(a.work)

    try:
        gt_states = states_for(gt, a.harness, work, "gt", duration, a.fps)
        base = evaluate(gt, gt, gt_states, gt_states)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        gt_states, base = [], {"error": str(exc), "status": "execution_failed"}
    rows = [{"kind": "基准", **base, "check": sensitivity_check(base, base, "基准")}]
    if rows[0]["check"]["status"] == "pass":
        for kind in ("运动类型改错", "单个物体平移", "整体平移（规范自由度）",
                     "删掉一个物体", "多造一个物体", "周期改错"):
            bad = perturb(gt, kind)
            if bad == gt:
                rows.append({"kind": kind, "check": {"status": "not_applicable", "reason": "no eligible target"}})
                continue
            try:
                st = states_for(bad, a.harness, work, f"bad_{len(rows)}", duration, a.fps)
                r = evaluate(gt, bad, gt_states, st)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                r = {"error": str(exc), "status": "execution_failed"}
            rows.append({"kind": kind, **r, "check": sensitivity_check(base, r, kind)})

    out = REPO / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(f"  {row['kind']}: {row['check']['status']} — {row['check']['reason']}")
    print(f"\n  明细写到 {out}")
    return 0 if all(row["check"]["status"] == "pass" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
