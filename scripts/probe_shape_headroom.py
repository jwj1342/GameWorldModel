#!/usr/bin/env python
"""形状种类全错，到底值多少分——用预言机定上界。

这是「加深组合深度」那条路的 MVP。在投入做"什么时候拆、怎么拆"之前，先回答一个
更基本的问题：**把形状种类修对，最多能买到多少？**

做法是把感知误差完全去掉：拿**真值程序**，只把每个物体的几何退化成**同尺寸的盒子**
（位置、朝向、尺寸、运动、材质一律不动），再拿这份退化版和原真值比。

这样测出来的差，就是「形状种类全错」的纯代价，也是修对它的收益上界。
实测我们现在的形状种类准确率是 0.300，所有物体都渲成盒子——正是这个退化版的处境。

读法：

  **上界很小** → 组合深度不是杠杆，别投。现有指标对形状种类不敏感，
                 真正拖低分数的是别的东西。
  **上界很大** → 值得投，而且知道了天花板在哪。

用法:
  python -m scripts.probe_shape_headroom --gt-root <真值根> --work <临时目录>
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

from gwm.feedback.gt_metrics import effective_shape, evaluate, node_extent  # noqa: E402
from scripts.eval import states_of, work_dir                                # noqa: E402


def boxify(program: dict) -> tuple[dict, int, dict]:
    """把每个节点的几何换成**同尺寸的盒子**，其余不动。

    尺寸取 `node_extent`，它对基本体和资产都给得出来（球按直径、锥柱按外接盒）。
    这样"退化"只丢掉形状种类，不丢位置也不丢大小——测的正好是种类那一项。
    """
    out = json.loads(json.dumps(program))
    changed = 0
    before: dict[str, int] = {}
    for node in (out.get("objects") or []) + (out.get("static") or []):
        shape = effective_shape(node)
        if shape:
            before[shape] = before.get(shape, 0) + 1
        ext = node_extent(node)
        if ext is None or not np.isfinite(ext).all():
            continue
        if shape == "box":
            node["geom"] = {"kind": "primitive", "shape": "box",
                            "extent": [float(v) for v in ext]}
            continue
        node["geom"] = {"kind": "primitive", "shape": "box",
                        "extent": [float(v) for v in ext]}
        changed += 1
    return out, changed, before


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--harness", default="scripts/node_harness.sh")
    ap.add_argument("--max-dist", type=float, default=0.45)
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args(argv)
    root, work = Path(a.gt_root), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    clips = sorted(d for d in root.iterdir() if (d / "program.json").is_file())[:a.limit]
    if not clips:
        print("没有真值片段"); return 1

    tot = {k: 0 for k in ("gt", "pred", "matched", "sgt", "spred", "smatched",
                          "kind_n", "kind_ok", "changed")}
    shapes: dict[str, int] = {}
    shape_errs, scale_ratios = [], []
    for clip in clips:
        gt = json.loads((clip / "program.json").read_text(encoding="utf-8"))
        boxed, changed, before = boxify(gt)
        for k, v in before.items():
            shapes[k] = shapes.get(k, 0) + v
        tot["changed"] += changed
        dur = float((gt.get("meta") or {}).get("duration") or 8.0)

        gt_path = clip / "program.json"
        boxed_path = work / f"{clip.name}_boxed.json"
        boxed_path.write_text(json.dumps(boxed), encoding="utf-8")
        gs = states_of(gt_path, work_dir(work, f"{clip.name}_gt", clip.name), dur, a.harness)
        bs = states_of(boxed_path, work_dir(work, f"{clip.name}_box", clip.name), dur, a.harness)
        m = evaluate(gt, boxed, gs, bs, max_dist_m=a.max_dist)
        full = m.get("full") or {}
        o, sc, sk = full.get("objects") or {}, full.get("scene") or {}, full.get("shape_kind") or {}
        tot["gt"] += o.get("gt", 0); tot["pred"] += o.get("pred", 0); tot["matched"] += o.get("matched", 0)
        tot["sgt"] += sc.get("gt", 0); tot["spred"] += sc.get("pred", 0); tot["smatched"] += sc.get("matched", 0)
        tot["kind_n"] += sk.get("n", 0); tot["kind_ok"] += sk.get("correct", 0)
        if o.get("shape_ok_frac") is not None:
            shape_errs.append(float(o["shape_ok_frac"]))
        if o.get("scale_ratio") is not None:
            scale_ratios.append(float(o["scale_ratio"]))
        print(f"  {clip.name:<14} 退化了 {changed:>2} 个几何   "
              f"物体 {o.get('matched',0)}/{o.get('gt',0)}   "
              f"形状种类 {sk.get('correct',0)}/{sk.get('n',0)}")

    print(f"\n真值里的形状种类构成：{dict(sorted(shapes.items(), key=lambda kv: -kv[1]))}")
    print(f"其中被退化成盒子的（即种类本来不是盒子的）：{tot['changed']}")
    print("\n—— 把真值的形状全退化成同尺寸盒子之后，拿它和真值比 ——\n")
    print(f"  物体召回    {tot['matched']}/{tot['gt']} = {tot['matched']/max(tot['gt'],1):.3f}")
    print(f"  物体精确率  {tot['matched']}/{tot['pred']} = {tot['matched']/max(tot['pred'],1):.3f}")
    print(f"  场景召回    {tot['smatched']}/{tot['sgt']} = {tot['smatched']/max(tot['sgt'],1):.3f}")
    print(f"  形状种类    {tot['kind_ok']}/{tot['kind_n']} = {tot['kind_ok']/max(tot['kind_n'],1):.3f}")
    if shape_errs:
        print(f"  形状对（边长比例）中位 {np.median(shape_errs):.3f}")
    if scale_ratios:
        print(f"  尺度之比 中位 {np.median(scale_ratios):.3f}")

    print("\n怎么读：")
    print("  这一行是**纯形状种类的代价**——位置、尺寸、运动、材质全是真值，")
    print("  只有形状种类被毁掉。所以：")
    print("    物体召回和精确率若仍接近 1.000，说明**现有的配对指标对形状种类不敏感**，")
    print("      那么把形状修对不会让主表上的数变好，组合深度不是推动那几个数的杠杆。")
    print("    形状种类那一列会掉到盒子的占比——这是定义使然，不是发现。")
    print("  要注意：这里测的是**指标空间**的代价，不是**观感**的代价。")
    print("  一个该是球却渲成盒子的物体，在配对指标上可能毫发无损，看起来却明显是错的。")
    print("  所以这个上界小，不等于组合深度对论文没用——只等于它推不动当前这几个数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
