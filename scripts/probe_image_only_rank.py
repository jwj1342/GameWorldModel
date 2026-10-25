"""图像分单独在全库里排序，正确的资产排第几名——创新点一的生死判据。

为什么要这一个数。「图像进候选」那个 A/B 两臂逐字节相同，我一度读成「图像分
定位不到正确资产」。查下来不是：`retrieve()` 里 extras 被接在文字命中表的**尾巴**
后面，而融合重排只动前 `shortlist`（24）个，于是 122 条决策里只有 8 条的候选表
真的变了（那 8 条文字只召回 2 个，表短，extras 才进了重排窗口），一条都没改选中。
那个 A/B 什么都没测到。

所以得先回答更底层的问题，而且这个问题不该被任何管线细节污染：
**把视频里物体的裁剪图嵌入，在全库 4812 个资产里按图像相似度排序，
正确那个排第几？**

  排不进前几百 → DINOv2 在低多边形资产上没有分辨力，这条路从根上不通，
                 修 `retrieve()` 的拼接顺序也救不回来。
  排得进前几十 → 瓶颈纯粹是那段拼接，修它就能把命中率从 0 抬起来。

两条纪律：

**不配对。** 预测物体和真值节点的配对条件里含被测量，这坑踩过。改用集合判据：
问「**本片段真值资产集合里排名最靠前的那个**排第几」。这是放水的上界——
给足好处之后还差，那就是真差。

**带零假设。** 同一张裁剪图，换成**另一条片段**的真值集合（大小匹配），再问一次
排名。真集合明显靠前才说明嵌入里有信号；两边差不多就是没有。

**自检。** 重算出来的 `max(image_scores)` 必须和 `rebuild.json` 里记的
`image_score_best` 对上（1e-3）。对不上说明我没跑在官方那条路径上，拒绝出数。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.assets.annotate import bbox_of, best_observation, crop_object   # noqa: E402
from gwm.assets.embed import AssetIndex, Embedder                        # noqa: E402
from gwm.assets.library import load_catalog, library_root                # noqa: E402
from gwm.config import load_config, site_name                        # noqa: E402
from gwm.perception.run import load_masks                            # noqa: E402


def gt_refs(program: dict) -> set[str]:
    out = set()
    for node in (program.get("objects") or []) + (program.get("static") or []):
        geom = node.get("geom") or {}
        if geom.get("kind") in ("asset", "generated") and geom.get("ref"):
            out.add(str(geom["ref"]))
    return out


def best_rank(order: list[str], wanted: set[str]) -> int | None:
    """`order` 是按图像分降序的全库 id。返回 wanted 里最靠前那个的名次（1 起）。"""
    for i, aid in enumerate(order):
        if aid in wanted:
            return i + 1
    return None


def _ratios(extent) -> list[float] | None:
    """边长比例：降序 + 归一化到最长边 = 1。只看形状，不看尺度。"""
    try:
        e = sorted((abs(float(x)) for x in extent), reverse=True)
    except Exception:
        return None
    if len(e) != 3 or e[0] <= 1e-9:
        return None
    return [x / e[0] for x in e]


def _shape_order(obs: list[float], shapes: dict) -> list[str]:
    return sorted(shapes, key=lambda k: sum(abs(x - y) for x, y in zip(obs, shapes[k])))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="跑过的产出根（里面是 *test_* 目录）")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--admit", type=int, default=32, help="报「前 N 名里有没有」的那个 N")
    ap.add_argument("--shapes", default="/scratch/jwj/gwm/asset_shapes.json",
                    help="资产边长比例缓存；加上它就同时量形状这条腿和两条腿的并集")
    a = ap.parse_args(argv)

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *a.config]
    cfg = load_config([f for f in files if Path(f).is_file()])
    root = library_root(cfg)
    if root is None:
        print("没有资产库，量不了"); return 2
    catalog = load_catalog(root)
    index = AssetIndex.load(Path(root) / "dinov2_index.npz")
    embedder = Embedder(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])
    print(f"库 {len(catalog)} 个资产", flush=True)
    shapes: dict[str, list[float]] = {}
    if a.shapes and Path(a.shapes).is_file():
        for k, v in json.loads(Path(a.shapes).read_text()).items():
            r = _ratios(v)
            if r is not None:
                shapes[k] = r
        print(f"形状库 {len(shapes)} 条可用", flush=True)
    else:
        print("没有形状缓存，只量图像这一条腿", flush=True)

    # 先把每条片段的真值集合读齐，零假设要从别的片段里抽
    gt_by_clip: dict[str, set[str]] = {}
    for clip_dir in sorted(Path(a.report).glob("*test_*")):
        clip = clip_dir.name.split("test_")[-1]
        p = Path(a.gt_root) / f"test_{clip}" / "program.json"
        if p.is_file():
            gt_by_clip[clip] = gt_refs(json.loads(p.read_text(encoding="utf-8")))

    rng = random.Random(20261023)
    real_ranks: list[int] = []
    null_ranks: list[int] = []
    in_topn = null_in_topn = 0
    checked = mismatched = 0
    n_crops = 0
    chosen_ranks: list[int] = []
    shape_ranks: list[int] = []
    shape_null_ranks: list[int] = []
    oracle_ranks: list[int] = []
    union_n = union_img = union_shape_saves = 0

    for clip_dir in sorted(Path(a.report).glob("*test_*")):
        clip = clip_dir.name.split("test_")[-1]
        wanted = gt_by_clip.get(clip)
        if not wanted:
            continue
        others = [c for c in gt_by_clip if c != clip and gt_by_clip[c]]
        ev_path = clip_dir / "perception" / "evidence.json"
        frames_path = clip_dir / "perception" / "frames" / "frames.json"
        if not (ev_path.is_file() and frames_path.is_file()):
            continue
        evidence = json.loads(ev_path.read_text(encoding="utf-8"))
        frames = json.loads(frames_path.read_text(encoding="utf-8"))["frames"]
        frame_by_index = {f.get("index"): f for f in frames}
        masks = load_masks(clip_dir / "perception")
        # 重建出来的盒子边长（带 VGGT 压扁）和真值边长（预言机），按节点 id 索引
        ext_pred: dict[str, list[float]] = {}
        pp = clip_dir / "program.json"
        if pp.is_file():
            _pg = json.loads(pp.read_text(encoding="utf-8"))
            for _n in (_pg.get("objects") or []) + (_pg.get("static") or []):
                _r = _ratios((_n.get("geom") or {}).get("extent"))
                if _r is not None:
                    ext_pred[str(_n.get("id"))] = _r
        ext_gt: list[list[float]] = []
        _gp = Path(a.gt_root) / f"test_{clip}" / "program.json"
        if _gp.is_file():
            _gg = json.loads(_gp.read_text(encoding="utf-8"))
            for _n in (_gg.get("objects") or []) + (_gg.get("static") or []):
                _r = _ratios((_n.get("geom") or {}).get("extent"))
                if _r is not None:
                    ext_gt.append(_r)
        # 官方记下来的那一路，用来自检
        recorded = {}
        rb = clip_dir / "rebuild.json"
        if rb.is_file():
            d = json.loads(rb.read_text(encoding="utf-8"))
            recs = d if isinstance(d, list) else (d.get("decisions") or d.get("records") or [])
            if isinstance(recs, dict):
                recs = list(recs.values())
            for r in recs:
                if isinstance(r, dict) and r.get("image_score_best") is not None:
                    recorded[str(r.get("evidence_id") or r.get("id"))] = r

        # 预言机：拿**真值**边长去全库排序，形状这条腿的天花板。
        # 逐个真值节点算，和预测物体不做任何配对（配对条件含被测量，坑踩过）。
        if shapes:
            for _r in ext_gt:
                _o = best_rank(_shape_order(_r, shapes), wanted)
                if _o is not None:
                    oracle_ranks.append(_o)

        for obj in evidence.get("objects") or []:
            observation = best_observation(obj)
            if observation is None:
                continue
            frame = frame_by_index.get(observation.get("frame_index"))
            mask = (masks.get(obj["id"]) or {}).get(observation.get("frame_index"))
            if not frame or mask is None:
                continue
            rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
            box, box_size = bbox_of(observation)
            if box_size and box_size[0]:
                k = rgb.shape[1] / float(box_size[0])
                box = [v * k for v in box]
            crop = crop_object(rgb, np.asarray(mask), box)
            if crop is None:
                continue
            vector = embedder.embed_images([Image.fromarray(crop)])[0]
            scores = index.scores(vector)
            n_crops += 1

            rec = recorded.get(str(obj["id"]))
            if rec is not None:
                checked += 1
                if abs(float(max(scores.values())) - float(rec["image_score_best"])) > 1e-3:
                    mismatched += 1

            order = [aid for aid, _s in sorted(scores.items(), key=lambda kv: -kv[1])]
            r = best_rank(order, wanted)
            if r is not None:
                real_ranks.append(r)
                in_topn += 1 if r <= a.admit else 0
            if others:
                # 零假设的集合**大小匹配**：别用一个更大的集合去比，名次天然会靠前
                donor = gt_by_clip[rng.choice(others)]
                fake = set(rng.sample(sorted(donor), min(len(wanted), len(donor))))
                rn = best_rank(order, fake)
                if rn is not None:
                    null_ranks.append(rn)
                    null_in_topn += 1 if rn <= a.admit else 0
            if rec is not None and rec.get("chosen"):
                cr = best_rank(order, {str(rec["chosen"])})
                if cr is not None:
                    chosen_ranks.append(cr)

            # ---- 形状这条腿：全库按边长比例排序，同样的集合判据 ----
            if shapes and rec is not None:
                obs = ext_pred.get(str(rec.get("id")))
                if obs is not None:
                    sorder = _shape_order(obs, shapes)
                    sr = best_rank(sorder, wanted)
                    if sr is not None:
                        shape_ranks.append(sr)
                    if others:
                        donor2 = gt_by_clip[rng.choice(others)]
                        fake2 = set(rng.sample(sorted(donor2), min(len(wanted), len(donor2))))
                        srn = best_rank(sorder, fake2)
                        if srn is not None:
                            shape_null_ranks.append(srn)
                    # 并集：同样的预算 admit，一半给图像一半给形状
                    half = max(1, a.admit // 2)
                    if r is not None and sr is not None:
                        union_n += 1
                        in_img = r <= half
                        union_img += 1 if in_img else 0
                        if sr <= half and not in_img:
                            union_shape_saves += 1


    print(f"\n自检：对上官方记录的裁剪图 {checked} 个，其中分数对不上的 {mismatched} 个")
    if checked == 0 or mismatched:
        print("**自检没过，不出数。** 说明这里跑的不是官方那条路径。")
        return 1

    n = len(catalog)
    def show(name: str, rs: list[int], hits: int):
        if not rs:
            print(f"  {name}: 没有样本"); return
        q = statistics.quantiles(rs, n=4) if len(rs) > 1 else [rs[0], rs[0], rs[0]]
        print(f"  {name}: 中位名次 {statistics.median(rs):.0f} / {n}"
              f"（四分位 {q[0]:.0f} / {q[2]:.0f}，最好 {min(rs)}）"
              f"  进前 {a.admit} 名的比例 {hits/len(rs):.3f}  样本 {len(rs)}")

    print(f"\n裁剪图 {n_crops} 张，库 {n} 个资产")
    show("真·本片段真值集合", real_ranks, in_topn)
    show("零假设·别片段的集合", null_ranks, null_in_topn)
    if chosen_ranks:
        print(f"  管线实际选中的那个，在图像分上的名次：中位 "
              f"{statistics.median(chosen_ranks):.0f} / {n}  样本 {len(chosen_ranks)}")
    if real_ranks and null_ranks:
        d = statistics.median(null_ranks) - statistics.median(real_ranks)
        print(f"\n  **真集合比零假设靠前 {d:+.0f} 名**"
              f"（随机的话该是 0 附近）")

    if shape_ranks:
        print(f"\n########## 形状这条腿（全库按边长比例） ##########")
        show("真·重建 extent", shape_ranks,
             sum(1 for x in shape_ranks if x <= a.admit))
        show("零假设·别片段的集合", shape_null_ranks,
             sum(1 for x in shape_null_ranks if x <= a.admit))
        if oracle_ranks:
            show("预言机·真值 extent", oracle_ranks,
                 sum(1 for x in oracle_ranks if x <= a.admit))
        print("  零假设和真实打平 → 形状里没信号；"
              "预言机远高于真实 → 是 VGGT 压扁吃掉的，修几何有用")

    if union_n:
        half = max(1, a.admit // 2)
        both = (union_img + union_shape_saves) / union_n
        print(f"\n########## 两条腿是不是在同一批物体上失败 ##########")
        print(f"  同时有两种名次的节点 {union_n} 个，预算 {a.admit} 拆成 {half}+{half}")
        print(f"  只用图像前 {half}      覆盖 {union_img}/{union_n} = {union_img/union_n:.3f}")
        print(f"  形状额外救回来的    {union_shape_saves}/{union_n} = {union_shape_saves/union_n:.3f}")
        print(f"  **并集覆盖 {both:.3f}**（同样 {a.admit} 个候选的预算）")
        if union_shape_saves == 0:
            print("  形状一个也没救回来 → 两条腿在同一批物体上失败，"
                  "候选生成这条路别再修了")
    print(f"""
怎么读：
  进前 {a.admit} 名的比例是关键——那正是「图像进候选{a.admit}」能救回来的上限。
  这个比例接近 0，说明修 retrieve() 的拼接顺序也没用，图像这条路从根上不通。
  真集合和零假设的名次差要先是正的、而且明显，后面的数才有意义。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
