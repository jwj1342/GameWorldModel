#!/usr/bin/env python
"""标定假设对不上取景，会把真值评测压成什么样。

`configs/default.yaml` 的 `assumed_camera_height_m` 是 1.2——按「一个人举着手机站在
地上拍」去把相对尺度的重建标定到米。而合成数据集的真值相机高度中位数是 7.385 米，
是从七米多高俯看的大场景。重建因此被整体压小十来倍，而物体配对的容差是 3 米，
于是绝大多数真值物体找不到对应。

这个脚本把假设换成真值里那个高度（只是**诊断**，不是修法——真实视频里没有真值可用），
整体缩放已经产出的程序再重跑评测，量出这一条占了多少。不重跑感知，所以几个配置之间
唯一的差别就是这个比例因子。

用法:
  python -m scripts.probe_scale_gauge --report $GWM_SCRATCH/dataset/pilot20_fix \
      --gt-root $GWM_SCRATCH/dataset/pilot50 --work $SLURM_TMPDIR/sg
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


def camera_height(program: dict) -> float | None:
    keys = (program.get("camera") or {}).get("keyframes") or []
    heights = [float(k["pos"][1]) for k in keys if (k.get("pos") or [None] * 3)[1] is not None]
    return statistics.median(heights) if heights else None


def scale_program(program: dict, k: float) -> dict:
    """把整个场景乘上 k。位置、尺寸、相机轨迹都要跟着走，否则不是相似变换。"""
    out = json.loads(json.dumps(program))
    for node in (out.get("objects") or []) + (out.get("static") or []):
        pose = node.get("pose") or {}
        if isinstance(pose.get("pos"), list):
            pose["pos"] = [float(v) * k for v in pose["pos"]]
        geom = node.get("geom") or {}
        for key in ("extent", "size"):
            if isinstance(geom.get(key), list):
                geom[key] = [float(v) * k for v in geom[key]]
        for key in ("radius", "radius_top", "radius_bottom", "height"):
            if isinstance(geom.get(key), (int, float)):
                geom[key] = float(geom[key]) * k
        for inst in node.get("instances") or []:
            if isinstance(inst.get("pos"), list):
                inst["pos"] = [float(v) * k for v in inst["pos"]]
        motion = node.get("motion") or {}
        for key in ("linear_velocity", "amplitude", "offset"):
            if isinstance(motion.get(key), list):
                motion[key] = [float(v) * k for v in motion[key]]
        for key in ("speed", "radius", "distance"):
            if isinstance(motion.get(key), (int, float)):
                motion[key] = float(motion[key]) * k
    for frame in (out.get("camera") or {}).get("keyframes") or []:
        if isinstance(frame.get("pos"), list):
            frame["pos"] = [float(v) * k for v in frame["pos"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="跑过 pilot_report 的目录")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--assumed", type=float, default=1.2,
                    help="链路当前假设的相机高度，用来反推比例因子")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    ratios = []
    for name, use_truth in (("asis", False), ("rescaled", True)):
        variant = work / name
        for clip_dir in sorted(p for p in Path(a.report).iterdir()
                               if (p / "game" / "program.json").is_file()):
            program = json.loads((clip_dir / "game" / "program.json").read_text(encoding="utf-8"))
            gt_file = Path(a.gt_root) / clip_dir.name / "program.json"
            k = 1.0
            if use_truth and gt_file.is_file():
                gt_h = camera_height(json.loads(gt_file.read_text(encoding="utf-8")))
                if gt_h and gt_h > 0:
                    k = gt_h / a.assumed
                    ratios.append(k)
            target = variant / clip_dir.name
            target.mkdir(parents=True, exist_ok=True)
            (target / "program.json").write_text(
                json.dumps(scale_program(program, k), indent=1, ensure_ascii=False), encoding="utf-8")
            (target / "run.json").write_text(json.dumps(
                {"clip": clip_dir.name, "args": {"config": []}, "config_identity": name}),
                encoding="utf-8")
        runs = work / f"runs_{name}.txt"
        runs.write_text("\n".join(str(p) for p in sorted(variant.iterdir()) if p.is_dir()),
                        encoding="utf-8")
        print(f"\n########## {name} ##########", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.eval", "--runs-from", str(runs),
                        "--work", str(work / "ev"), "--gt-root", a.gt_root,
                        "--out", str(work / f"eval_{name}.json")], cwd=REPO, check=False)
    if ratios:
        print(f"\n按真值相机高度反推的比例因子：中位数 {statistics.median(ratios):.2f}，"
              f"范围 {min(ratios):.2f}~{max(ratios):.2f}")
    print("\n这是诊断不是修法：真实视频里没有真值相机高度可用。它回答的是"
          "「标定假设对不上取景，占了现在这组数里多少」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
