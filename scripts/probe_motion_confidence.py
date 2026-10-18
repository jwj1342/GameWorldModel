#!/usr/bin/env python
"""运动类型判不出来，是感知给的轨迹不行，还是分类器的置信度定义不行。

`scripts/probe_motion_scale.py` 在**神经感知**的证据上量到：三十三个样本里二十五个
判成 unknown，原因都是「motion hypotheses are not sufficiently separated」，最强候选
的置信度中位数 0.017，而门槛是 0.55——差三十倍。尺度已经排除（放大一到十二倍，准确率
一点没变）。

这里换成**真值感知**再量一遍。真值感知下掩码和深度都是完美的，轨迹误差只有 0.140 米。
如果置信度在这种条件下能上到门槛，问题就定位在感知；如果仍然上不去，那分类器的置信度
定义本身要重做——它在一条好轨迹上也给不出把握。

真值感知还有一个好处：物体 id 直接来自真值程序，不需要规范对齐和配对，省掉一层混淆。

用法:
  python -m scripts.probe_motion_confidence --rendered $GWM_SCRATCH/dataset/pilot50_rendered \
      --gt-root $GWM_SCRATCH/dataset/pilot50 --work $SLURM_TMPDIR/mc --limit 10
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.config import load_config, site_name              # noqa: E402
from gwm.perception.run import run_perception              # noqa: E402


def perceive(clip_dir: Path, gt_program: Path, work: Path) -> dict:
    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml",
             REPO / "configs/perception/gt.yaml"]
    cfg = load_config([f for f in files if Path(f).is_file()])
    cfg["perception"]["gt_program"] = str(gt_program)
    return run_perception(str(clip_dir / "clip.mp4"), clip_dir.name, work / clip_dir.name, cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in Path(a.rendered).iterdir() if (p / "clip.mp4").is_file())[:a.limit]

    hit = total = 0
    kinds: Counter = Counter()
    reasons: Counter = Counter()
    confs: list[float] = []
    margins: list[float] = []
    by_truth: dict[str, Counter] = {}
    angles: dict[str, list] = {}
    structure: dict[str, list] = {}
    for clip_dir in clips:
        gt_file = Path(a.gt_root) / clip_dir.name / "program.json"
        if not gt_file.is_file():
            continue
        try:
            evidence = perceive(clip_dir, gt_file, work)
        except Exception as error:                      # noqa: BLE001
            print(f"  {clip_dir.name} 感知失败 {type(error).__name__}", flush=True)
            continue
        truth = {n["id"]: str((n.get("motion") or {}).get("type", "static"))
                 for n in json.loads(gt_file.read_text(encoding="utf-8")).get("objects", [])}
        # 相机的平均朝向：拿首尾两帧相机位置之差不行（相机可能绕着转），
        # 用每帧的朝向向量求平均
        view = None
        poses = ((evidence.get("camera") or {}).get("poses")) or []
        if poses:
            dirs = []
            for p in poses:
                q = p.get("quat")
                if not q or len(q) != 4:
                    continue
                x, y, z, w = (float(v) for v in q)
                # 相机看向局部 -z，把它转到世界
                dirs.append([-2 * (x * z + w * y), -2 * (y * z - w * x), -(1 - 2 * (x * x + y * y))])
            if dirs:
                m = np.mean(np.array(dirs), axis=0)
                if np.linalg.norm(m) > 1e-9:
                    view = m / np.linalg.norm(m)
        for obj in evidence.get("objects", []):
            want = truth.get(obj["id"])
            if want is None:
                continue
            mg = obj.get("motion_guess") or {}
            got = str(mg.get("type", "unknown"))
            total += 1
            hit += int(got == want)
            kinds[got] += 1
            by_truth.setdefault(want, Counter())[got] += 1
            if got == "unknown":
                reasons[str(mg.get("notes") or "没写原因")] += 1
            # 判成沿轴滑动的，量一下那根轴和相机视线的夹角。配准的漂移多半沿视线方向
            # （深度最不可观测的那个方向），真的滑动没有理由和视线对齐。如果两者分得开，
            # 就有一个判别子能把「静止物体被判成在滑」挑出来。
            axis = mg.get("axis") or (mg.get("parameters") or {}).get("axis")
            if got in ("prismatic", "periodic_translate", "trajectory") and axis and view is not None:
                v = np.array([float(x) for x in axis], float)
                n = np.linalg.norm(v)
                if n > 1e-9:
                    cos = abs(float(np.dot(v / n, view)))
                    angles.setdefault(want, []).append(np.degrees(np.arccos(min(1.0, cos))))
            # 结构判据：真的滑轨有行程端点，会往复；自由轨迹和漂移是单向的。
            # 沿主方向投影，数速度变号的次数，再看有多少时间几乎静止。
            boxes = obj.get("obb") or []
            if len(boxes) >= 6:
                pts = np.array([[float(v) for v in b["center"]] for b in boxes])
                delta = pts[-1] - pts[0]
                span = np.linalg.norm(delta)
                principal = delta / span if span > 1e-9 else np.array([1.0, 0.0, 0.0])
                along = pts @ principal
                vel = np.diff(along)
                scale = np.max(np.abs(vel)) if len(vel) else 0.0
                if scale > 1e-9:
                    sign = np.sign(vel)
                    nonzero = sign[np.abs(vel) > 0.1 * scale]
                    reversals = int(np.sum(np.diff(nonzero) != 0)) if len(nonzero) > 1 else 0
                    rest = float(np.mean(np.abs(vel) < 0.1 * scale))
                    structure.setdefault(want, []).append((reversals, rest))
            details = mg.get("candidate_details") or []
            if details and isinstance(details[0].get("confidence"), (int, float)):
                confs.append(float(details[0]["confidence"]))
            if isinstance(mg.get("candidate_margin"), (int, float)):
                margins.append(float(mg["candidate_margin"]))
        print(f"  {clip_dir.name} 累计 {total} 个物体，判对 {hit}", flush=True)

    if not total:
        print("没有样本")
        return 1
    print(f"\n真值感知下：{total} 个物体，判对 {hit}，准确率 {hit / total:.3f}")
    print(f"  预测类型分布 {kinds.most_common(6)}")
    if reasons:
        print(f"  判成 unknown 的原因 {reasons.most_common(3)}")
    if confs:
        print(f"  最强候选置信度：中位数 {statistics.median(confs):.3f}，"
              f"范围 {min(confs):.3f}~{max(confs):.3f}（门槛 0.55）")
    if margins:
        print(f"  与第二名的差距：中位数 {statistics.median(margins):.3f}（门槛 0.10）")
    print("\n  按真值类型看判成了什么：")
    for want, got in sorted(by_truth.items()):
        print(f"    {want:<20} {got.most_common(4)}")
    if angles:
        print("\n  判成「沿某根轴移动」的那些，轴和相机视线的夹角（度，0 = 完全沿视线）：")
        for want, vals in sorted(angles.items()):
            vals = sorted(vals)
            print(f"    真值 {want:<20} n={len(vals):<3} 中位数 {vals[len(vals)//2]:.1f}  "
                  f"范围 {vals[0]:.1f}~{vals[-1]:.1f}")
        print("    真值静止那一行如果明显更小，说明漂移沿视线，可以拿夹角当判别子。")
    if structure:
        print("\n  沿主方向的速度结构（真滑轨应该往复、有停顿；自由轨迹和漂移单向）：")
        print(f"    {'真值类型':<20}{'样本':>5}{'往复次数中位数':>14}{'几乎静止的时间占比':>18}")
        def q(v, p):
            v = sorted(v)
            return v[min(len(v) - 1, int(p * len(v)))]
        for want, vals in sorted(structure.items()):
            rev = [v[0] for v in vals]; rest = [v[1] for v in vals]
            print(f"    {want:<20}{len(vals):>5}"
                  f"  往复 {q(rev,0):>2}/{q(rev,0.5):>2}/{q(rev,0.99):>2}"
                  f"   静止占比 {q(rest,0):.2f}/{q(rest,0.5):.2f}/{q(rest,0.99):.2f}")
        print("    三个数分别是最小、中位、最大。要当判别子，两类的区间不能重叠太多。")
    print("\n置信度如果在这里能上到门槛，问题在感知；如果仍然上不去，"
          "分类器的置信度定义本身要重做——它在一条好轨迹上也给不出把握。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
