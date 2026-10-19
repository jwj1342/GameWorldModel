#!/usr/bin/env python
"""短轨迹里，哪些是真东西只露了几帧，哪些是检测器在噪声上闪。

`min_track_frame_frac: 0.30` 把精确率从 0.214 提到 0.300、场景精确率 0.285 提到
0.389，**但召回从 0.407 掉到 0.356**。代价来自这个判据太粗：它只数「被看到几帧」，
而一个**真的**物体如果在视频后段才进画面，本来就只有几帧。

要分开这两种，得看帧的**分布**而不只是**数量**：

  连续度 = 出现帧数 / （最后一帧 - 第一帧 + 1）

  真东西只露几帧 —— 它一出现就连着在，连续度接近 1。
  检测器在噪声上闪 —— 帧是散的，连续度低。

这个脚本量的就是：在短轨迹里，**配上真值的那些**连续度是不是明显高于**配不上的**。
是的话，判据可以改成「帧少 **且** 散」，把真东西留下来，召回的代价就能收回去。

用法:
  python -m scripts.probe_short_track_kind --report <未过滤的产物目录> --eval <eval.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def describe(name: str, xs: list[float]) -> str:
    if not xs:
        return f"  {name:<24} （没有样本）"
    a = np.sort(np.asarray(xs, float))
    return (f"  {name:<24} n={len(a):<4} 中位 {np.median(a):.3f}  "
            f"四分位 {np.percentile(a, 25):.3f} / {np.percentile(a, 75):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--frac", type=float, default=0.30,
                    help="当前判据的阈值：被看到的帧数低于这个比例就算短轨迹")
    a = ap.parse_args(argv)

    rows = json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", [])
    short_hit: list[float] = []      # 短轨迹 且 配上真值
    short_miss: list[float] = []     # 短轨迹 且 配不上
    long_hit: list[float] = []
    long_miss: list[float] = []
    n_short = n_short_hit = 0

    for r in rows:
        scene = ((r.get("metrics") or {}).get("full") or {}).get("scene") or {}
        matched = {q for q in (scene.get("pairs") or {}).values() if q}
        mi_path = Path(a.report) / r["clip"] / "perception" / "masks_index.json"
        geo = Path(a.report) / r["clip"] / "perception" / "geometry.npz"
        if not (mi_path.is_file() and geo.is_file()):
            continue
        mi = json.loads(mi_path.read_text(encoding="utf-8"))
        n_total = len(np.load(geo)["frame_indices"])
        for oid, m in (mi.get("objects") or {}).items():
            frames = sorted(int(f) for f in (m.get("frames") or []))
            if len(frames) < 2 or n_total < 2:
                continue
            span = frames[-1] - frames[0] + 1
            contiguity = len(frames) / max(span, 1)
            is_short = len(frames) / n_total < a.frac
            hit = oid in matched or f"static::{oid}" in matched
            if is_short:
                n_short += 1
                n_short_hit += hit
                (short_hit if hit else short_miss).append(contiguity)
            else:
                (long_hit if hit else long_miss).append(contiguity)

    if not (short_hit or short_miss):
        print("没有短轨迹样本"); return 1

    print(f"短轨迹判据：被看到的帧数 / 总帧数 < {a.frac}\n")
    print(f"短轨迹一共 {n_short} 条，其中配上真值的 {n_short_hit} 条"
          f"（{n_short_hit / max(n_short, 1):.3f}）")
    print("\n连续度 = 出现帧数 /（最后一帧 - 第一帧 + 1）。1 表示一出现就连着在。\n")
    print(describe("短轨迹·配上真值", short_hit))
    print(describe("短轨迹·配不上", short_miss))
    print(describe("长轨迹·配上真值", long_hit))
    print(describe("长轨迹·配不上", long_miss))

    if short_hit and short_miss:
        sh, sm = np.asarray(short_hit), np.asarray(short_miss)
        print(f"\n短轨迹里两组的中位数：{np.median(sh):.3f}（配上） 对 "
              f"{np.median(sm):.3f}（配不上）")
        print("\n「帧少**且**散」这个判据能救回多少：")
        print(f"{'连续度阈值':>12}{'保住的真东西':>14}{'仍然丢掉的噪声':>16}{'差':>9}")
        best = (None, -1.0)
        for thr in np.percentile(np.concatenate([sh, sm]), [10, 25, 50, 75, 90]):
            keep = float((sh >= thr).mean())      # 连续度高就留下 -> 救回来的真东西
            still = float((sm < thr).mean())      # 连续度低仍然丢 -> 仍挡住的噪声
            if keep + still - 1 > best[1]:
                best = (float(thr), keep + still - 1)
            print(f"{thr:>12.3f}{keep:>14.3f}{still:>16.3f}{keep + still - 1:>+9.3f}")
        print(f"\n最好的阈值 {best[0]:.3f}，保住率加挡住率减一 = {best[1]:+.3f}。")
        print("接近零就是这个判据分不开，那现在那条「只数帧数」的粗判据已经到顶，")
        print("召回的代价收不回来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
