#!/usr/bin/env python
"""扫重复检出的合并阈值，用真值指标判好坏。

#56 量过：没配上真值的预测物体里 38% 离最近的真阳性不到一个物体尺度，是重复检出；
剩下 62% 是无中生有，这一步碰不到。所以去重的上限是精确率 0.173 -> 0.239，而**代价**
是可能误伤——堆叠的箱子、紧挨的栏杆本来就该是两个物体，合掉了召回就掉。

阈值卡在哪只能用数据定。在**已经跑好的**产物上离线合并，不重跑感知，所以几个阈值
之间唯一的差别就是这一刀。

评测的配对容差要按场景尺度给，别用默认的 3 米——见 #68 的更正，绝对容差在小场景里
等于「什么都能配上」。

用法:
  python -m scripts.sweep_dedup --report $GWM_SCRATCH/dataset/ab015_report \
      --gt-root $GWM_SCRATCH/dataset/ab015 --work $SLURM_TMPDIR/dd \
      --max-dist 0.45 --thresholds 0,0.3,0.5,0.7
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--max-dist", type=float, default=None)
    ap.add_argument("--thresholds", default="0,0.3,0.5,0.7")
    a = ap.parse_args()
    from gwm.synthesis.dedup import merge_duplicates

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    wanted = [float(v) for v in a.thresholds.split(",")]
    if len(set(wanted)) != len(wanted):
        raise SystemExit("--thresholds 里有重复的值；表会少一行而且看不出来")
    summary = {}
    for threshold in wanted:
        variant = work / f"t{threshold}"
        dropped = 0
        clips = [p for p in sorted(Path(a.report).iterdir())
                 if (p / "game" / "program.json").is_file()]
        for clip_dir in clips:
            program = json.loads((clip_dir / "game" / "program.json").read_text(encoding="utf-8"))
            ev_file = clip_dir / "perception" / "evidence.json"
            evidence = json.loads(ev_file.read_text(encoding="utf-8")) if ev_file.is_file() else None
            dropped += len(merge_duplicates(program, threshold, evidence=evidence))
            target = variant / clip_dir.name
            target.mkdir(parents=True, exist_ok=True)
            (target / "program.json").write_text(
                json.dumps(program, indent=1, ensure_ascii=False), encoding="utf-8")
            (target / "run.json").write_text(json.dumps(
                {"clip": clip_dir.name, "args": {"config": []},
                 "config_identity": f"dedup{threshold}"}), encoding="utf-8")
        runs = work / f"runs_{threshold}.txt"
        runs.write_text("\n".join(str(p) for p in sorted(variant.iterdir()) if p.is_dir()),
                        encoding="utf-8")
        out = work / f"eval_{threshold}.json"
        cmd = [sys.executable, "-m", "scripts.eval", "--runs-from", str(runs),
               "--work", str(work / "ev"), "--gt-root", a.gt_root, "--out", str(out)]
        if a.max_dist is not None:
            cmd += ["--max-dist", str(a.max_dist)]
        print(f"\n########## 阈值 {threshold}：合掉 {dropped} 个 ##########", flush=True)
        subprocess.run(cmd, cwd=REPO, check=False, capture_output=True)
        if out.is_file():
            rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
            g = p = m = 0
            for r in rows:
                o = ((r.get("metrics") or {}).get("full") or {}).get("objects") or {}
                g += o.get("gt") or 0; p += o.get("pred") or 0; m += o.get("matched") or 0
            summary[threshold] = (dropped, g, p, m)
    print(f"\n{'阈值':>6}{'合掉':>6}{'真值':>7}{'预测':>7}{'配上':>7}{'召回':>9}{'精确率':>10}{'F1':>9}")
    for threshold, (dropped, g, p, m) in summary.items():
        rc = m / g if g else 0.0
        pr = m / p if p else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"{threshold:>6.2f}{dropped:>6}{g:>7}{p:>7}{m:>7}{rc:>9.3f}{pr:>10.3f}{f1:>9.3f}")
    print("\n召回掉下去就说明误伤了真物体——堆叠和紧挨的本来就该是两个。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
