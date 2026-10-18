#!/usr/bin/env python
"""运动类型判不出来，有多少是因为尺度不对。

`gwm/perception/motion.py` 的判据是**绝对米制**的：位移不到 `static_pos_thresh_m`
（0.05 米）就算静止，平移假设至少要 `translation_min_displacement_m`（0.08 米）。
而 #68 量出来重建比真值整体小六到十二倍——一个真实移动 0.3 米的物体，在重建里看起来
只移动 0.05 米，正好卡在静止线上。这能解释混淆矩阵里「会动的物体几乎全被判成静止」。

放大位置和缩小阈值是等价的，所以这里不改配置，直接把证据里的位置乘一个系数再送进
同一个分类器——走的是一模一样的代码路径，省得改阈值时漏掉某个耦合的地方。

配对用评测已经算好的那份（真值 id -> 预测节点 id），再顺着 slug 回到证据对象。
不重跑感知。

用法:
  python -m scripts.probe_motion_scale --eval $GWM_SCRATCH/dataset/pilot20_fix_eval.json \
      --report $GWM_SCRATCH/dataset/pilot20_fix --gt-root $GWM_SCRATCH/dataset/pilot50
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.config import load_config, site_name          # noqa: E402
from gwm.perception.motion import estimate_motion      # noqa: E402
from gwm.taxonomy import slug                          # noqa: E402


def sequence(obj: dict):
    """从证据里的包围盒序列取出时间、中心、朝向。"""
    boxes = obj.get("obb") or []
    if len(boxes) < 2:
        return None
    times = np.array([float(b["t"]) for b in boxes])
    pos = np.array([[float(v) for v in b["center"]] for b in boxes])
    quat = np.array([[float(v) for v in b["quat"]] for b in boxes])
    size = np.median(np.array([[float(v) for v in b["size"]] for b in boxes]), axis=0)
    return times, pos, quat, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="scripts.eval 出的 json，用它的配对")
    ap.add_argument("--report", required=True, help="跑过 pilot_report 的目录")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--scales", default="1,2,4,6,8,12")
    a = ap.parse_args()

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"]
    cfg = load_config([f for f in files if Path(f).is_file()])

    # 评测的配对：真值 id -> 预测节点 id
    rows = json.loads(Path(a.eval).read_text(encoding="utf-8"))["rows"]
    samples = []
    for row in rows:
        clip = row["clip"]
        pairs = ((row.get("metrics") or {}).get("matching") or {}).get("pairs") or {}
        gt_file = Path(a.gt_root) / clip / "program.json"
        ev_file = Path(a.report) / clip / "perception" / "evidence.json"
        if not (gt_file.is_file() and ev_file.is_file()):
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        gt_type = {n["id"]: str((n.get("motion") or {}).get("type", "static"))
                   for n in gt.get("objects", [])}
        ev = json.loads(ev_file.read_text(encoding="utf-8"))
        by_node = {slug(o["id"]): o for o in ev.get("objects", [])}
        for g, p in pairs.items():
            if not p or g not in gt_type:
                continue
            obj = by_node.get(p)
            seq = sequence(obj) if obj else None
            if seq:
                samples.append((gt_type[g], seq))
    print(f"配对且证据齐全的样本 {len(samples)} 个")
    if not samples:
        return 1
    print(f"真值类型分布: {Counter(t for t, _ in samples).most_common()}")

    print(f"\n{'放大倍数':>9}{'判对':>7}{'准确率':>9}  预测类型分布")
    for raw in a.scales.split(","):
        s = float(raw)
        hit = 0
        pred_counts: Counter = Counter()
        reasons: Counter = Counter()
        margins: list = []
        tops: list = []
        for truth, (times, pos, quat, size) in samples:
            out = estimate_motion(times, pos * s, quat, cfg, size=size * s,
                                  coordinate_space="world")
            mg = out["motion_guess"]
            kind = str(mg.get("type", "unknown"))
            pred_counts[kind] += 1
            if kind == "unknown":
                reasons[str(mg.get("notes") or "没写原因")] += 1
                margins.append(mg.get("candidate_margin"))
                tops.append((mg.get("candidate_details") or [{}])[0].get("confidence"))
            hit += int(kind == truth)
        print(f"{s:>9.1f}{hit:>7}{hit / len(samples):>9.3f}  {pred_counts.most_common(5)}")
        if reasons and s == 1.0:
            print("    判成 unknown 的原因:")
            for why, n in reasons.most_common(6):
                print(f"      {n:>3}  {why[:100]}")
            good = [v for v in tops if isinstance(v, (int, float))]
            gm = [v for v in margins if isinstance(v, (int, float))]
            if good:
                good.sort()
                print(f"    最强候选的置信度：中位数 {good[len(good)//2]:.3f}，"
                      f"范围 {good[0]:.3f}~{good[-1]:.3f}（门槛 "
                      f"{cfg['perception']['motion']['min_primary_confidence']}）")
            if gm:
                gm.sort()
                print(f"    与第二名的差距：中位数 {gm[len(gm)//2]:.3f}（门槛 "
                      f"{cfg['perception']['motion']['min_hypothesis_margin']}）")
    print("\n放大位置和缩小阈值是等价的。如果准确率随倍数明显变好，"
          "说明判不出来主要是因为尺度，而不是判据本身不行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
