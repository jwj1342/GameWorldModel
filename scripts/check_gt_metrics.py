#!/usr/bin/env python3
"""敏感性检查：把一份正确的程序程序化改坏，看指标掉不掉。

issue #4 要求的第一步。指标写出来了但其实测不到东西，是最尴尬的情况，
这个脚本就是用来挡住它的。每种扰动只破坏一个方面，期望对应的那项指标恶化，
不相关的项基本不动。

轨迹不在 Python 里重算，两份程序都经 harness 跑出逐帧状态再比，
所以这个脚本需要一个能跑无头浏览器的环境。
"""
from __future__ import annotations
import argparse, copy, json, subprocess, sys, tempfile
from pathlib import Path

from gwm.config import REPO
from gwm.feedback.gt_metrics import evaluate, load_states


def perturb(program: dict, kind: str) -> dict:
    """每种扰动只破坏一个方面，便于判断指标有没有指到正确的地方。"""
    p = copy.deepcopy(program)
    objs = p["objects"]
    if kind == "运动类型改错":           # 期望：运动类型准确率掉
        objs[0]["motion"] = {"type": "static"}
    elif kind == "轨迹整体平移":          # 期望：轨迹误差涨，运动类型不变
        for o in objs:                    # 物体可以用 pose，也可以用 instances 放多份
            if "pose" in o: o["pose"]["pos"][0] += 0.5
            for inst in o.get("instances", []): inst["pos"][0] += 0.5
    elif kind == "删掉一个物体":          # 期望：召回掉
        gone = objs.pop()["id"]
        # 绑定里还引用着它的话编译过不去，一并清掉，保证这次扰动只影响召回
        slots = p.get("binding", {}).get("slots", {})
        for key in ("collectibles", "hazards"):
            slots[key] = [x for x in slots.get(key, []) if not x.startswith(gone)]
    elif kind == "多造一个物体":          # 期望：误检涨（precision 掉）
        extra = copy.deepcopy(objs[0]); extra["id"] = "ghost"
        extra["pose"]["pos"] = [x + 6.0 for x in extra["pose"]["pos"]]
        objs.append(extra)
    elif kind == "周期改错":              # 期望：轨迹误差涨，且留出段比整段更明显
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

    gt_states = states_for(gt, a.harness, work, "gt", duration, a.fps)
    base = evaluate(gt, gt, gt_states, gt_states)
    print(f"  基准（真值和自己比，应当接近完美）")
    print(f"    轨迹中位误差 {base['full']['trajectory']['median_m']:.4f} m, "
          f"运动类型准确率 {base['full']['motion_type']['accuracy']:.2f}, "
          f"召回 {base['full']['objects']['recall']:.2f}, 误检率 {1-base['full']['objects']['precision']:.2f}")
    print()

    rows = [{"kind": "基准", **base}]
    for kind in ("运动类型改错", "轨迹整体平移", "删掉一个物体", "多造一个物体", "周期改错"):
        bad = perturb(gt, kind)
        st = states_for(bad, a.harness, work, f"bad_{len(rows)}", duration, a.fps)
        r = evaluate(gt, bad, gt_states, st)
        f, h = r["full"], r["holdout"]
        print(f"  {kind}")
        print(f"    整段   轨迹 中位{f['trajectory']['median_m']:.3f} 最大{f['trajectory']['max_m']:.3f} m  "
              f"类型 {f['motion_type']['accuracy']:.2f}  召回 {f['objects']['recall']:.2f}  误检 {1-f['objects']['precision']:.2f}")
        print(f"    留出段 轨迹 中位{h['trajectory']['median_m']:.3f} 最大{h['trajectory']['max_m']:.3f} m  "
              f"类型 {h['motion_type']['accuracy']:.2f}")
        rows.append({"kind": kind, **r})

    out = REPO / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    print(f"\n  明细写到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
