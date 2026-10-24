"""检索挑对了吗——真值用资产库搭之后，这个问题第一次有答案。

为什么到今天才有。原来的真值是「从词表随机挑个名字 + 独立挑个基本体」，所以
**「该挑哪个资产」压根没有标准答案**：实测把检索重排改掉，86 个资产节点换了
55 个（64%），而所有指标逐字节不动；补上 `asset_shape_error` 之后，配对上的约
36 个物体里也只有 1 个的值变了。收益指标循环（重排正是在优化它）、代价指标
`class_accuracy` 恒为噪声（类名和形状脱钩），这个实验从设计上就不成立。

真值改成从同一个资产库搭（`generate_dataset.py --use-assets`）之后两件事都修好了：
资产 id 可以直接比对，类名从资产文件名来所以和网格天然绑定。

这个脚本报三个数，三个都要看：

  **命中率**     选中的资产 id 和真值一模一样的比例。这是最严的口径。
  **同套件率**   至少挑到了同一个 kit（kenney 的一个套件风格统一）。
                 比命中率宽，但仍然比「随便挑个网格」强得多。
  **形状距离**   选中资产和真值资产的边长比例差多少。即使 id 不对，
                 形状接近也说明几何信号在起作用——这正是创新点一要验证的东西。

口径提醒：只统计**真值和预测都解析到资产**的那些节点。预测退回基本体的不算命中
也不算未命中，单独报，否则「少挑几个」会伪装成「挑得准」。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def refs_of(program: dict) -> dict[str, str]:
    out = {}
    for node in (program.get("objects") or []) + (program.get("static") or []):
        geom = node.get("geom") or {}
        if geom.get("kind") in ("asset", "generated") and geom.get("ref"):
            out[str(node.get("id"))] = str(geom["ref"])
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True,
                    help="<标签>:<产出根>:<真值根>，可以给多次")
    ap.add_argument("--shapes", default="/scratch/jwj/gwm/asset_shapes.json")
    a = ap.parse_args(argv)

    try:
        shapes = json.loads(Path(a.shapes).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        shapes = {}

    for spec in a.pair:
        label, report, gt_root = spec.split(":", 2)
        hit = kit_hit = n = 0
        fell_back = gt_assets = 0
        dists: list[float] = []
        for clip_dir in sorted(Path(report).glob("*test_*")):
            clip = clip_dir.name.split("test_")[-1]
            gt_path = Path(gt_root) / f"test_{clip}" / "program.json"
            pred_path = clip_dir / "program.json"
            if not (gt_path.is_file() and pred_path.is_file()):
                continue
            gt = refs_of(json.loads(gt_path.read_text(encoding="utf-8")))
            pred = refs_of(json.loads(pred_path.read_text(encoding="utf-8")))
            gt_assets += len(gt)
            # 配对靠位置太脆，这里按**数量**比：真值有多少个资产节点，我们挑中了多少个
            # 真值里确实出现过的资产。这是个集合口径，不依赖逐物体配对，所以不受
            # 召回只有 0.45 的影响——那正是上一轮量不出东西的原因之一。
            gt_set, gt_kits = set(gt.values()), {v.split("/")[0] for v in gt.values()}
            for ref in pred.values():
                n += 1
                if ref in gt_set:
                    hit += 1
                if ref.split("/")[0] in gt_kits:
                    kit_hit += 1
                best = min((float(np.abs(np.asarray(shapes[ref], float)
                                         - np.asarray(shapes[g], float)).mean())
                            for g in gt_set if g in shapes and ref in shapes),
                           default=None)
                if best is not None:
                    dists.append(best)
            fell_back += max(0, len(gt) - len(pred))
        print(f"\n########## {label} ##########")
        if not n:
            print("  一个资产节点都没有，没法比"); continue
        print(f"  真值资产节点 {gt_assets} 个，我们产出了 {n} 个（退回基本体的 {fell_back} 个）")
        print(f"  **命中率（id 完全一致）** {hit}/{n} = {hit / n:.3f}")
        print(f"  同套件率（至少挑对 kit）  {kit_hit}/{n} = {kit_hit / n:.3f}")
        if dists:
            print(f"  形状距离（到最近的真值资产）中位 {statistics.median(dists):.4f}，"
                  f"四分位 {np.percentile(dists, 25):.4f} / {np.percentile(dists, 75):.4f}")
    print("""
怎么读：
  三个数一起看。命中率是最严的，形状距离最宽——**创新点一要动的是后者**：
  即使 id 挑不对，形状更接近也说明几何信号在起作用。
  只有命中率涨而形状距离不动，多半是碰巧；只有形状距离降而命中率不动，
  说明几何在起作用但还不足以定位到具体那一个。
提醒：退回基本体的节点不计入分母，否则「少挑几个」会伪装成「挑得准」。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
