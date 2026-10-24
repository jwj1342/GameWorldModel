"""给资产库预计算「边长比例」，让几何能参与资产检索——创新点一在管线里的落点。

为什么要它。`shape_kind_accuracy` 在这条管线上**结构性恒零**：`effective_shape()`
对解析到资产的节点返回 `"mesh"`，而真值全是基本体，93% 的物体都解析到了资产。
所以那个指标不是创新点一的落点。

真正有数、也真正对应「稠密几何监督离散选择」的是 **`shape_ok`**（比的是三条边的
**比例**，现在 0.6923）。而资产检索目前只看**类名和图像**——`Asset` 这个结构里
一个几何字段都没有。观测到的包围盒边长比例完全没参与「选哪个资产」这个离散选择。

这个脚本把每个资产的边长比例算出来存成缓存（采点 → 轴对齐包围盒 → 三边排序后
除以最长边）。之后检索就能按「观测的比例和资产的比例差多少」重排。

口径：比例是**无量纲**的，所以不受资产本身建模单位的影响——这正是要的，
因为尺度在下游另有一步去拟合。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.assets.library import load_catalog, library_root     # noqa: E402
from gwm.assets.mesh import load_points                       # noqa: E402


def edge_ratios(path: Path, samples: int = 1500) -> list[float] | None:
    pts = load_points(path, samples=samples)
    if pts is None or len(pts) < 20:
        return None
    ext = np.ptp(np.asarray(pts, float), axis=0)
    if not np.isfinite(ext).all() or ext.max() <= 1e-9:
        return None
    return [float(v) for v in np.sort(ext)[::-1] / ext.max()]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只算前 N 个，用来先探成本")
    a = ap.parse_args(argv)

    root = library_root({})
    if root is None:
        print("没有资产库，什么都不做"); return 1
    catalog = load_catalog(root)
    if a.limit:
        catalog = catalog[:a.limit]
    print(f"资产 {len(catalog)} 个，库在 {root}")
    out: dict[str, list[float]] = {}
    failed = 0
    for i, asset in enumerate(catalog):
        r = edge_ratios(root / asset.path)
        if r is None:
            failed += 1
        else:
            out[asset.id] = [round(v, 4) for v in r]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(catalog)}，失败 {failed}", flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"写出 {len(out)} 个（失败 {failed}）到 {a.out}")
    if out:
        arr = np.asarray(list(out.values()), float)
        print(f"  中间边/最长边 中位 {np.median(arr[:, 1]):.3f}，"
              f"最短边/最长边 中位 {np.median(arr[:, 2]):.3f}")
        print("  这两个数说明库里资产的形状有多扁。都接近 1 的话几何没有区分度，"
              "那这条路一开始就不成立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
