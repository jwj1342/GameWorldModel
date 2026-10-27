"""**降采样的查询**去打**原生分辨率的库**——这才是管线真正在做的事。

分辨率阶梯有个设计缺陷：它把两侧**一起**降采样（16px 的查询打 16px 的库），
量出来 0.958，于是我判「分辨率不是原因」。但真实管线是**不对称**的：
查询是一张 16px 的视频裁剪图，而库里的条目是 160px 的干净缩略图。
降采样对降采样比降采样对原生**容易**，所以那个 0.958 可能高估了。

这里补上：拿资产 A 第 0 个视角的**降采样向量**，去打**原生索引**
（挖掉 A 的第 0 个视角，A 另外三个原生视角还在），问 A 排第几。
只读存好的 .npz，不跑模型。

  掉得厉害 → 跨分辨率匹配本身就是问题，阶梯那个结论要收回
  还是很高 → 分辨率确实不是原因，证伪站得住
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load(path: Path):
    d = np.load(path, allow_pickle=False)
    ids = np.asarray([str(x) for x in d["asset_ids"]])
    v = np.asarray(d["vectors"], dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return ids, v


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--query-index", required=True, help="降采样建的那个索引，只取它的向量当查询")
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--admit", type=int, default=64)
    a = ap.parse_args(argv)

    nid, nv = load(Path(a.native))
    qid, qv = load(Path(a.query_index))
    if len(nid) != len(qid) or not (nid == qid).all():
        print("两个索引的条目顺序对不上，拒绝出数"); return 1
    uniq = sorted(set(nid.tolist()))
    pos = {k: i for i, k in enumerate(uniq)}
    owner = np.asarray([pos[k] for k in nid])
    rows_of: dict[str, list[int]] = {}
    for i, k in enumerate(nid):
        rows_of.setdefault(k, []).append(i)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(uniq), size=min(a.sample, len(uniq)), replace=False)
    ranks = []
    for ai in pick:
        aid = uniq[ai]
        rows = rows_of.get(aid) or []
        if len(rows) < 2:
            continue
        q = qv[rows[0]]                 # 降采样的查询
        sims = nv @ q                   # 打原生的库
        sims[rows[0]] = -2.0            # 挖掉同一个视角
        best = np.full(len(uniq), -3.0, dtype=np.float32)
        np.maximum.at(best, owner, sims)
        ranks.append(int((best > best[pos[aid]]).sum()) + 1)
    if not ranks:
        print("没有样本"); return 1
    ranks.sort()
    n = len(uniq)
    top = sum(1 for r in ranks if r <= a.admit) / len(ranks)
    print(f"查询={Path(a.query_index).name}  库={Path(a.native).name}  样本 {len(ranks)}")
    print(f"  中位名次 {statistics.median(ranks):.0f} / {n}  "
          f"四分位 {ranks[len(ranks)//4]} / {ranks[3*len(ranks)//4]}  "
          f"第一名 {sum(1 for r in ranks if r==1)/len(ranks):.3f}  "
          f"进前 {a.admit} 名 **{top:.3f}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
