"""图像嵌入封在 0.284，是**嵌入本身不行**还是**视频域不行**——留一视角法。

背景。端到端挑对率分解成 覆盖率 0.284 x 条件准确率 0.19，大头是覆盖率，
而覆盖率是图像嵌入在全库 4812 个里的召回。两条全局低维描述子的腿
（边长比例、Hu 矩）都证实当不了候选生成器，所以能不能突破 0.284
就全看图像嵌入还有没有余量。

要分清两种原因，它们的修法完全相反：

  (A) 嵌入做不了**跨视角的实例检索**——同一个网格换个角度就认不出。
      那改裁剪、改光照都没用，得换/微调嵌入。
  (B) 嵌入跨视角没问题，是**视频域**和渲染缩略图差太远（光照、掩码毛边、
      分辨率、遮挡）。那不用动模型，把两侧的域拉近就能涨。

判据。库里每个资产存了 4 个视角的向量。拿资产 A 的第 k 个视角当查询，
**把 A 的第 k 个视角从库里挖掉**，问 A 还排第几（A 的另外三个视角还在）。
这是纯渲染对纯渲染、只差视角，所以它量的就是 (A)。

  排名接近 1 → 嵌入跨视角没问题，0.284 的锅在视频域，是 (B)，有救。
  排名也很差 → 嵌入本身就做不了这件事，是 (A)，换模型才有戏。

全程只用存好的 .npz 向量，不跑模型、不碰管线。

`--index` 指到灰度索引时，这个探针变成**灰度实验的对照**：灰度把颜色从两侧
一起去掉，所以「灰度之后覆盖率没涨」有两种读法——颜色本来就不是主因，或者
灰度在修好域的同时也毁掉了同样多的分辨力。跑一遍灰度版的留一视角就能分开：
灰度 LOO 仍然接近 0.99 → 分辨力没丢，覆盖率的变化可以全算到颜色头上；
灰度 LOO 明显掉 → 两件事混在一起了，那个对比不能用。
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.assets.library import library_root          # noqa: E402
from gwm.config import load_config, site_name        # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--index", default="", help="换一个索引 npz（比如灰度那个）")
    ap.add_argument("--sample", type=int, default=600, help="抽多少个资产来问")
    ap.add_argument("--admit", type=int, default=64)
    a = ap.parse_args(argv)

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *a.config]
    cfg = load_config([f for f in files if Path(f).is_file()])
    root = library_root(cfg)
    if root is None:
        print("没有资产库"); return 2
    index_path = Path(a.index) if a.index else (Path(root) / "dinov2_index.npz")
    print(f"索引 {index_path.name}")
    data = np.load(index_path, allow_pickle=False)
    ids = np.asarray([str(x) for x in data["asset_ids"]])
    vecs = np.asarray(data["vectors"], dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    uniq = sorted(set(ids.tolist()))
    print(f"库 {len(uniq)} 个资产 / {len(ids)} 个视角向量，维度 {vecs.shape[1]}")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(uniq), size=min(a.sample, len(uniq)), replace=False)
    want = [uniq[i] for i in pick]
    by_asset: dict[str, list[int]] = {}
    for i, k in enumerate(ids):
        by_asset.setdefault(k, []).append(i)
    pos = {k: i for i, k in enumerate(uniq)}
    owner_idx = np.asarray([pos[k] for k in ids])

    ranks, singles = [], 0
    for aid in want:
        rows = by_asset.get(aid) or []
        if len(rows) < 2:
            singles += 1
            continue
        q = vecs[rows[0]]
        sims = vecs @ q
        sims[rows[0]] = -2.0                 # 挖掉查询自己那一个视角
        # 每个资产取视角最大值（和线上 AssetIndex.scores 一致），用分组最大值做，
        # 逐行 Python 循环是 600x19248 次。
        best = np.full(len(uniq), -3.0, dtype=np.float32)
        np.maximum.at(best, owner_idx, sims)
        ranks.append(int((best > best[pos[aid]]).sum()) + 1)

    if not ranks:
        print("没有可用样本"); return 1
    ranks.sort()
    n = len(uniq)
    q1, q3 = ranks[len(ranks)//4], ranks[3*len(ranks)//4]
    top = sum(1 for r in ranks if r <= a.admit) / len(ranks)
    print(f"\n留一视角自匹配（纯渲染对纯渲染，只差视角）：样本 {len(ranks)} 个"
          f"{'，只有一个视角被跳过 ' + str(singles) + ' 个' if singles else ''}")
    print(f"  中位名次 {statistics.median(ranks):.0f} / {n}  四分位 {q1} / {q3}  "
          f"第一名的比例 {sum(1 for r in ranks if r == 1)/len(ranks):.3f}  "
          f"进前 {a.admit} 名 {top:.3f}")
    print()
    if top > 0.8:
        print(f"  → 嵌入**跨视角没问题**（前 {a.admit} 名覆盖 {top:.3f}）。")
        print(f"    那端到端那 0.284 的锅就在**视频域**：光照、掩码毛边、分辨率、遮挡。")
        print(f"    修法不是换模型，是把查询侧和库侧的域拉近。")
    elif top < 0.3:
        print(f"  → 嵌入**跨视角就做不到**（前 {a.admit} 名只有 {top:.3f}）。")
        print(f"    纯渲染、只差视角都这样，视频域再怎么修也顶不上去。")
        print(f"    要突破 0.284 只能换嵌入或在资产渲染图上微调。")
    else:
        print(f"  → 居中（{top:.3f}）：两个原因都占一部分，先修便宜的那个（域）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
