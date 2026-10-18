#!/usr/bin/env python
"""扫短命轨迹过滤的阈值，用真值指标判对错。

只被看到几帧的轨迹基本是噪声（实测 0-10% 那一桶的运动估计成功率是 0%），而它们把物体
精确率压得很低。但「丢多少」不能凭感觉定——丢多了会把真物体一起丢掉，查全跟着掉。

做法是在**已经跑好的**小批量产物上离线过滤：按被看到的帧数比例筛掉物体，写成一份新的
程序，再用 scripts/eval.py 对着真值算指标。不重跑感知，所以几个阈值之间唯一的差别就是
这一刀切在哪。

用法:
  python -m scripts.sweep_track_filter --report $GWM_SCRATCH/dataset/pilot20_quat \
      --gt-root $GWM_SCRATCH/dataset/pilot50 --work $SLURM_TMPDIR/tf --fracs 0,0.15,0.25,0.4
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


def observed_fraction(evidence: dict, object_id: str) -> float:
    """这个物体在多少比例的帧里被看到。"""
    frames = max(len(evidence.get("frames") or []), 1)
    for obj in evidence.get("objects", []):
        if obj["id"] == object_id:
            return len(obj.get("obb") or []) / frames
    return 1.0                      # 证据里找不到就不过滤，宁可留着


def build_variant(report: Path, out: Path, frac: float) -> int:
    """按阈值筛一遍，写出新的运行目录。返回保留的物体总数。"""
    from gwm.taxonomy import slug
    kept_total = 0
    for clip_dir in sorted(p for p in report.iterdir() if (p / "program.json").is_file()):
        program = json.loads((clip_dir / "program.json").read_text(encoding="utf-8"))
        evidence_file = clip_dir / "perception" / "evidence.json"
        if evidence_file.is_file() and frac > 0:
            evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
            by_slug = {slug(o["id"]): o["id"] for o in evidence.get("objects", [])}
            program["objects"] = [
                node for node in program.get("objects", [])
                if observed_fraction(evidence, by_slug.get(node["id"], node["id"])) >= frac]
        target = out / clip_dir.name
        target.mkdir(parents=True, exist_ok=True)
        (target / "program.json").write_text(json.dumps(program, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
        (target / "run.json").write_text(json.dumps(
            {"clip": clip_dir.name, "args": {"config": []},
             "config_identity": f"frac{frac}"}), encoding="utf-8")
        kept_total += len(program.get("objects", []))
    return kept_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="pilot_report 的输出目录")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--fracs", default="0,0.15,0.25,0.4")
    a = ap.parse_args()
    report, work = Path(a.report), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for raw in a.fracs.split(","):
        frac = float(raw)
        variant = work / f"frac_{frac}"
        kept = build_variant(report, variant, frac)
        runs = work / f"runs_{frac}.txt"
        runs.write_text("\n".join(str(p) for p in sorted(variant.iterdir()) if p.is_dir()),
                        encoding="utf-8")
        print(f"\n########## 过滤阈值 {frac}：保留 {kept} 个物体 ##########", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.eval", "--runs-from", str(runs),
                        "--work", str(work / "ev"), "--gt-root", a.gt_root,
                        "--out", str(work / f"eval_{frac}.json")],
                       cwd=REPO, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
