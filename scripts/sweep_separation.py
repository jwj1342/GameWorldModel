#!/usr/bin/env python
"""扫「重力稳定之前先跑多久纯分离」这个参数，用实测可交互性判好坏。

为什么要分离这一段：`harness/interaction_probe.mjs` 量出来，拖不动的物体插入深度
中位数 0.032 m，拖得动的只有 0.005 m——拖不动的根因是生成的时候就和别的东西插在
一起。而重力稳定解不开这种穿插：重力只有竖直方向的分量，两个物体横向咬住的时候，
重力没有把它们分开的方向可用。

不重跑感知。每条片段的 `program_before_settle.json` 就是稳定之前那一版，从那里分叉，
只有稳定这一步的参数不同，所以几个配置之间唯一的差别就是这一刀。

用法:
  python -m scripts.sweep_separation --report $GWM_SCRATCH/dataset/pilot20_quat \
      --work $SLURM_TMPDIR/sep --seconds 0,0.5,1.0 --limit 10
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def build_variant(clip_dir: Path, out: Path, separate: float, cfg: dict,
                  refit: bool = True) -> dict | None:
    """编译 -> 稳定（带分离）-> 采纳 -> 重编译 -> 探一遍交互。"""
    from gwm.compiler.compile import compile_program
    from gwm.synthesis.contact import settle_on_ground
    from gwm.synthesis.stabilise import apply_settled
    from gwm.feedback.interaction_probe import measure

    source = clip_dir / "program_before_settle.json"
    if not source.is_file():
        return None
    program = json.loads(source.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    result = compile_program(program, out / "game", cfg)
    if not result["ok"]:
        return {"clip": clip_dir.name, "ok": False, "reason": "compile_failed"}
    settle_json = out / "settle.json"
    try:
        subprocess.run([str(REPO / "scripts/node_harness.sh"), "settle.mjs",
                        "--game", str(out / "game"), "--out", str(settle_json),
                        "--seconds", "1.5", "--separate-seconds", str(separate)],
                       check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as error:
        return {"clip": clip_dir.name, "ok": False, "reason": type(error).__name__}
    rows = apply_settled(program, json.loads(settle_json.read_text(encoding="utf-8")))
    # 管线在采纳之后还会再过一次接触装配：仿真允许极小的穿透来保持接触稳定，而校验器
    # 按几何判定，两者容差对不上，不补这一下有些片段会重编译不过。这里必须照做，
    # 否则扫出来的数和真正跑出来的不是一回事——而且它正好能回答一个问题：
    # 这次按地面贴合的调整，会不会把刚刚分开的物体又推回去。
    refit_count = 0
    if refit and any(r.get("applied") for r in rows):
        refit_count = len([m for m in settle_on_ground(program) if m["applied"]])
    (out / "program.json").write_text(json.dumps(program, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    result = compile_program(program, out / "game", cfg)
    if not result["ok"]:
        return {"clip": clip_dir.name, "ok": False, "reason": "recompile_failed"}
    probe = measure(Path(result["game_dir"]), out, str(REPO / "scripts/node_harness.sh"))
    if not probe.get("ok"):
        return {"clip": clip_dir.name, "ok": False, "reason": probe.get("reason")}
    s = probe["summary"]
    pens = [o["drag"].get("penetration") for o in probe["objects"]
            if (o.get("drag") or {}).get("mode") == "free"]
    pens = [p for p in pens if isinstance(p, (int, float))]
    return {"clip": clip_dir.name, "ok": True,
            "applied": sum(1 for r in rows if r.get("applied")), "refit": refit_count,
            "declared": s["declared_grabbable"], "works": s["works"],
            "success": s["manipulation_success_fraction"],
            "clickable": s["median_clickable_fraction"],
            "penetration_median": round(statistics.median(pens), 4) if pens else None,
            "penetration_over_1cm": round(sum(1 for p in pens if p > 0.01) / len(pens), 3) if pens else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="跑过 pilot_report 的目录")
    ap.add_argument("--work", required=True)
    ap.add_argument("--seconds", default="0,0.5,1.0")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--no-refit", action="store_true",
                    help="跳过稳定之后那次按地面的接触装配，用来看它是不是把刚分开的又推回去了")
    a = ap.parse_args()
    from gwm.config import load_config, site_name
    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"]
    cfg = load_config([f for f in files if Path(f).is_file()])
    clips = [p for p in sorted(Path(a.report).iterdir())
             if (p / "program_before_settle.json").is_file()][:a.limit]
    print(f"片段 {len(clips)} 条，配置 {a.seconds}")
    table = {}
    for raw in a.seconds.split(","):
        separate = float(raw)
        rows = []
        for clip_dir in clips:
            row = build_variant(clip_dir, Path(a.work) / f"sep_{separate}" / clip_dir.name,
                                separate, cfg, refit=not a.no_refit)
            if row:
                rows.append(row)
                print(f"  [{separate}] {row['clip']:<14} "
                      f"{'成功率 ' + str(row.get('success')) if row['ok'] else row.get('reason')}",
                      flush=True)
        table[separate] = rows
        Path(a.work, f"rows_{separate}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    def med(rows, key):
        v = [r[key] for r in rows if r.get("ok") and isinstance(r.get(key), (int, float))]
        return round(statistics.median(v), 3) if v else None

    print(f"\n{'分离秒数':>8}{'片段':>6}{'声明能抓':>10}{'真拖得动':>10}{'汇总成功率':>12}"
          f"{'插入深度中位数':>16}{'插超1cm比例':>14}{'贴合改动':>10}")
    for separate, rows in table.items():
        ok = [r for r in rows if r.get("ok")]
        declared = sum(r["declared"] for r in ok)
        works = sum(r["works"] for r in ok)
        print(f"{separate:>8.2f}{len(ok):>6}{declared:>10}{works:>10}"
              f"{(works / declared if declared else 0):>12.3f}"
              f"{str(med(rows, 'penetration_median')):>16}"
              f"{str(med(rows, 'penetration_over_1cm')):>14}"
              f"{sum(r.get('refit') or 0 for r in ok):>10}")
    print("\n看的是「真拖得动」那一列：插入深度降下去但拖得动的没变多，这一步就不值得做。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
