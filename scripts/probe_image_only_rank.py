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


def gt_refs_by_tint(program: dict) -> tuple[set[str], set[str]]:
    """真值资产分两堆：**渲染时会被染色的**和**用 GLB 原生材质的**。

    判据直接抄渲染那一侧 `kernel/scene.js`：

        const measured = typeof spec.material === 'object' && spec.material
                         ? spec.material.base_color : null;
        group.add(measured && TINT_ASSETS ? tintToMeasured(fitted, measured) : fitted);

    **只有 material 是带 base_color 的字典时才染色。** 是字符串时（包括
    十六进制字符串 "#4fce98"）typeof 是 string，measured 为 null，用原生材质。
    实测 assetgt60 里只有 37.6% 会被染色。

    这个分桶是**免费的对照**：如果 0.284 的锅主要在颜色，原生那一堆该明显好。
    """
    tinted, native = set(), set()
    for n in (program.get("objects") or []) + (program.get("static") or []):
        ref = ((n.get("geom") or {}).get("ref") or "").strip()
        if not ref:
            continue
        m = n.get("material")
        if isinstance(m, dict) and m.get("base_color"):
            tinted.add(ref)
        else:
            native.add(ref)
    return tinted, native


def gt_nodes_with_ref(program: dict) -> tuple[set[str], set[str]]:
    """真值节点 id 分两堆：**有资产真值的**和**没有的**。

    为什么要分。实测 assetgt60：`obj_*` 290 个**全部**有 ref，而 `rail_*`(249)、
    `block_*`(60)、`ground_*`(60) **一个都没有**——真值明说结构件是基本体，
    不是资产，**没有「正确答案」可挑**。

    但管线在真值分割这一档会拿到真值的掩码和 id（结构件也在里面），然后把它们
    统统当成 asset 节点送进检索。实测 273 个决策里 **151 个（55.3%）是结构件，
    命中率恒为 0**——那不是检索失败，是**根本不该做的决策**。

    只有真值分割这一档 id 能对上（id 来自真值程序）。神经分割那档对不上，
    这时两堆的重叠会很小，调用方应当据此跳过这个分桶。
    """
    with_ref, without = set(), set()
    for n in (program.get("objects") or []) + (program.get("static") or []):
        nid = str(n.get("id") or "")
        if not nid:
            continue
        if ((n.get("geom") or {}).get("ref") or "").strip():
            with_ref.add(nid)
        else:
            without.add(nid)
    return with_ref, without


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


def _unflatten(r: list[float], k: float = 0.526) -> list[float]:
    """把已知的压扁补回去再比形状。

    VGGT 沿相机视线把物体压扁，实测**最短轴只剩正确比例的 0.526**。
    形状这条腿拿重建边长的比例去和资产库比，那个系统偏差是直接带进去的。
    最短轴除以 0.526 再归一化，是对这个已知偏差的一阶校正——不是新信号，
    是把一个量过的常数用上。校正后仍不动，说明偏差不是靠常数能修的
    （比如它随视角变），那就得修几何本身。
    """
    e = sorted(r, reverse=True)
    e[-1] = e[-1] / k if k > 1e-6 else e[-1]
    e = sorted(e, reverse=True)
    return [x / e[0] for x in e] if e[0] > 1e-9 else r


_SIL_MATRIX = None


def _sil_matrix(catalog):
    """全库轮廓描述子矩阵：(资产数 x 4 视角, D)、每行属于哪个资产的整数下标、资产 id 表。

    **为什么要这条腿。** 已经坐实根因是 VGGT 沿相机视线把物体压扁（最短轴只剩
    0.526），而**沿视线的误差投影到像平面上恰好是零**——轮廓是唯一一个绕开这个
    根因的判据，实测也确实比深度残差好 2.6 倍。

    但到今天为止轮廓只被用来**重排一个由外观生成的 64 人池子**，从没被允许
    **在全库 4812 个里生成候选**。而那个池子的覆盖率 0.284 正是端到端的天花板。
    """
    global _SIL_MATRIX
    if _SIL_MATRIX is not None:
        return _SIL_MATRIX
    from scripts.probe_motion_signal import _asset_silhouettes
    rows, idx, ids = [], [], []
    for asset in catalog:
        descs = _asset_silhouettes(asset.id)
        if not descs:
            continue
        here = len(ids)
        ids.append(asset.id)
        for d in descs:
            rows.append(d)
            idx.append(here)
    if not rows:
        _SIL_MATRIX = (None, None, None)
        return _SIL_MATRIX
    _SIL_MATRIX = (np.stack(rows), np.asarray(idx), ids)
    print(f"轮廓库 {len(ids)} 个资产 / {len(rows)} 个视角", flush=True)
    return _SIL_MATRIX


def _sil_order(obs_descs, catalog) -> list[str] | None:
    """观测轮廓对**全库**排序。协议和 _silhouette_pick_rate 一致：
    对资产的 4 个视角取最近，对帧取中位；距离是描述子上的 L1。

    用 np.minimum.at 做分组最小值——逐行 Python 循环是 4800x4x帧数x物体数 次，
    跑不完。
    """
    mat, idx, ids = _sil_matrix(catalog)
    if mat is None or not obs_descs:
        return None
    per_frame = []
    for o in obs_descs:
        d = np.abs(mat - o[None, :]).sum(axis=1)
        best = np.full(len(ids), np.inf)
        np.minimum.at(best, idx, d)
        per_frame.append(best)
    agg = np.median(np.stack(per_frame), axis=0)
    order = np.argsort(agg)
    return [ids[i] for i in order]


def _shape_order(obs: list[float], shapes: dict) -> list[str]:
    return sorted(shapes, key=lambda k: sum(abs(x - y) for x, y in zip(obs, shapes[k])))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="跑过的产出根（里面是 *test_* 目录）")
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--admit", type=int, default=32, help="报「前 N 名里有没有」的那个 N")
    ap.add_argument("--pipeline-pool", action="store_true",
                    help="额外量**管线真实候选池**（retrieve 的 text+image 融合）的覆盖率")
    ap.add_argument("--frames", type=int, default=1,
                    help="每个物体用几帧做检索（>1 就对帧取分数最大值再排序）")
    ap.add_argument("--index", default="", help="换一个索引 npz（比如灰度那个）")
    ap.add_argument("--silhouette-index", action="store_true",
                    help="查询侧也转成黑底白形的二值轮廓（和轮廓索引配对用）")
    ap.add_argument("--gray", action="store_true",
                    help="查询侧也转灰度；必须和 --index 指的那个索引同口径，否则量的是噪声")
    ap.add_argument("--silhouette", action="store_true",
                    help="再加一条腿：观测轮廓的 Hu 矩在全库里排序（绕开压扁那个根因）")
    ap.add_argument("--shapes", default="/scratch/jwj/gwm/asset_shapes.json",
                    help="资产边长比例缓存；加上它就同时量形状这条腿和两条腿的并集")
    a = ap.parse_args(argv)

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *a.config]
    cfg = load_config([f for f in files if Path(f).is_file()])
    root = library_root(cfg)
    if root is None:
        print("没有资产库，量不了"); return 2
    catalog = load_catalog(root)
    index_path = Path(a.index) if a.index else (Path(root) / "dinov2_index.npz")
    index = AssetIndex.load(index_path)
    print(f"索引 {index_path.name}（查询侧{'转灰度' if a.gray else '不转'}）", flush=True)
    if a.gray and "gray" not in index_path.name:
        print("!! 查询转了灰度但索引名里没有 gray，两侧口径可能对不上", flush=True)
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
    tint_by_clip: dict[str, tuple[set[str], set[str]]] = {}
    refnodes_by_clip: dict[str, tuple[set[str], set[str]]] = {}
    for clip_dir in sorted(Path(a.report).glob("*test_*")):
        clip = clip_dir.name.split("test_")[-1]
        p = Path(a.gt_root) / f"test_{clip}" / "program.json"
        if p.is_file():
            _pg = json.loads(p.read_text(encoding="utf-8"))
            gt_by_clip[clip] = gt_refs(_pg)
            tint_by_clip[clip] = gt_refs_by_tint(_pg)
            refnodes_by_clip[clip] = gt_nodes_with_ref(_pg)

    rng = random.Random(20261023)
    real_ranks: list[int] = []
    null_ranks: list[int] = []
    in_topn = null_in_topn = 0
    checked = mismatched = 0
    n_crops = 0
    chosen_ranks: list[int] = []
    size_rows: list[tuple[float, int]] = []   # (裁剪图短边像素, 名次)
    pool_hit: list[int] = []                  # 管线真实候选池覆盖到没有
    pool_wp: list[int] = []                   # 同上，只算良定义节点
    wellposed_ranks: list[int] = []
    illposed_ranks: list[int] = []
    tint_ranks: list[int] = []
    native_ranks: list[int] = []
    shape_ranks: list[int] = []
    shape_fix_ranks: list[int] = []
    sil_ranks: list[int] = []
    sil_null_ranks: list[int] = []
    union_sil_saves = 0
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

        # 拿**真值**边长去全库排序。注意这是**循环**的（见下方说明）：数据集
        # 等比缩放，真值 extent 的比例就是资产原生比例。保留它是因为它回答了
        # 另一个问题——「描述子本身最多能分辨到什么程度」。
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
            # **多帧检索。** 一个物体在视频里被看了几十帧，而这里一直只用
            # 「框最大的那一帧」一张裁剪图。残差判据那边量过多视角的收益很大
            # （1/3/7 帧 = 0.149/0.243/0.311），图像检索这一路从没试过。
            # 对帧取**分数最大值**再排序：库侧本来就是对 4 个视角取 max，
            # 两边聚合方式一致；而且物体朝向任意，只要有一帧对得上就算像。
            _obs_list = [observation]
            if a.frames > 1:
                _cands = []
                for _ob in (obj.get("observations") or []):
                    _fi = _ob.get("frame_index")
                    if _fi is None or _fi not in (masks.get(obj["id"]) or {}):
                        continue
                    if frame_by_index.get(_fi) is None:
                        continue
                    _cands.append(_ob)
                if len(_cands) > a.frames:
                    _st = len(_cands) / float(a.frames)
                    _cands = [_cands[min(len(_cands) - 1, int(_st * (i + 0.5)))]
                              for i in range(a.frames)]
                if _cands:
                    _obs_list = _cands
            _scores_acc: dict[str, float] = {}
            _short = None
            _got = 0
            for _ob in _obs_list:
                _fr = frame_by_index.get(_ob.get("frame_index"))
                _mk = (masks.get(obj["id"]) or {}).get(_ob.get("frame_index"))
                if not _fr or _mk is None:
                    continue
                _rgb = np.asarray(Image.open(_fr["file"]).convert("RGB"))
                _bx, _bs = bbox_of(_ob)
                if _bs and _bs[0]:
                    _k = _rgb.shape[1] / float(_bs[0])
                    _bx = [v * _k for v in _bx]
                _cr = crop_object(_rgb, np.asarray(_mk), _bx)
                if _cr is None:
                    continue
                if _short is None:
                    _short = float(min(_cr.shape[0], _cr.shape[1]))
                _im = Image.fromarray(_cr)
                if a.silhouette_index:
                    _m2 = np.asarray(_mk).astype(bool)
                    if _m2.shape != _rgb.shape[:2]:
                        from gwm.perception.masks import resize_mask
                        _m2 = resize_mask(_m2, (_rgb.shape[1], _rgb.shape[0])).astype(bool)
                    _x0, _y0, _x1, _y1 = (int(v) for v in _bx)
                    _x0, _y0 = max(0, _x0), max(0, _y0)
                    _x1, _y1 = min(_rgb.shape[1], _x1), min(_rgb.shape[0], _y1)
                    if _x1 - _x0 < 8 or _y1 - _y0 < 8:
                        continue
                    _sub = _m2[_y0:_y1, _x0:_x1]
                    _im = Image.fromarray(
                        np.where(_sub[:, :, None], 255, 0).astype(np.uint8).repeat(3, axis=2))
                elif a.gray:
                    _im = _im.convert("L").convert("RGB")
                _v = embedder.embed_images([_im])[0]
                for _aid, _sc in index.scores(_v).items():
                    if _sc > _scores_acc.get(_aid, -1.0):
                        _scores_acc[_aid] = _sc
                _got += 1
            if not _got or _short is None:
                continue

            rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
            box, box_size = bbox_of(observation)
            if box_size and box_size[0]:
                k = rgb.shape[1] / float(box_size[0])
                box = [v * k for v in box]
            crop = crop_object(rgb, np.asarray(mask), box)
            if crop is None:
                continue
            _img = Image.fromarray(crop)
            if a.silhouette_index:
                # 查询侧的轮廓直接用掩码裁出来——掩码本来就是前景，
                # 比再从裁剪图按背景色抠一遍准，而且两侧都只剩形状。
                _m = np.asarray(mask).astype(bool)
                if _m.shape != rgb.shape[:2]:
                    from gwm.perception.masks import resize_mask
                    _m = resize_mask(_m, (rgb.shape[1], rgb.shape[0])).astype(bool)
                _x0, _y0, _x1, _y1 = (int(v) for v in box)
                _x0, _y0 = max(0, _x0), max(0, _y0)
                _x1, _y1 = min(rgb.shape[1], _x1), min(rgb.shape[0], _y1)
                if _x1 - _x0 < 8 or _y1 - _y0 < 8:
                    continue
                _sub = _m[_y0:_y1, _x0:_x1]
                _img = Image.fromarray(
                    np.where(_sub[:, :, None], 255, 0).astype(np.uint8).repeat(3, axis=2))
            elif a.gray:
                _img = _img.convert("L").convert("RGB")
            vector = embedder.embed_images([_img])[0]
            # 多帧那一路已经把分数按帧取过最大值了，直接用它；单帧时两者等价。
            scores = _scores_acc if a.frames > 1 else index.scores(vector)
            n_crops += 1

            rec = recorded.get(str(obj["id"]))
            # 换了索引、或者多帧聚合，都别和 rebuild.json 对账——
            # 那条记录是**单帧**跑出来的 image_score_best，多帧取 max 之后
            # 本来就该不一样（实测 363 个里 343 个对不上，自检正确地拦了下来）。
            if rec is not None and not a.index and a.frames <= 1:
                checked += 1
                if abs(float(max(scores.values())) - float(rec["image_score_best"])) > 1e-3:
                    mismatched += 1

            order = [aid for aid, _s in sorted(scores.items(), key=lambda kv: -kv[1])]
            # **管线真实候选池的覆盖率。** 上面那个覆盖率只用图像分，而管线的候选
            # 是 retrieve() 做的 text+image 融合。两者差多少直接决定分解式里的
            # 条件准确率是不是 0.86——如果真实池覆盖率高得多，那条件准确率就低得多，
            # 「判据不是瓶颈」这个结论就要收回。所以必须按管线那条路径量一遍。
            if a.pipeline_pool:
                from gwm.assets.retrieve import retrieve as _retrieve
                _q = str(obj.get("class_guess") or "").strip() or "object"
                _settings = (cfg.get("assets") or {})
                _hits = _retrieve(_q, catalog, k=a.admit, image_scores=scores,
                                  image_weight=float(_settings.get("image_weight", 0.6)),
                                  image_admit=int(_settings.get("image_admit", 0) or a.admit))
                _ids = {h[0].id for h in _hits}
                _ok = 1 if (_ids & wanted) else 0
                pool_hit.append(_ok)
                _wr, _wo = refnodes_by_clip.get(clip, (set(), set()))
                if str(obj["id"]) in _wr:
                    pool_wp.append(_ok)

            r = best_rank(order, wanted)
            if r is not None:
                real_ranks.append(r)
                in_topn += 1 if r <= a.admit else 0
                size_rows.append((_short, r))
                # 这个节点在真值里到底有没有资产答案
                _wr, _wo = refnodes_by_clip.get(clip, (set(), set()))
                _oid = str(obj["id"])
                if _oid in _wr:
                    wellposed_ranks.append(r)
                elif _oid in _wo:
                    illposed_ranks.append(r)
            if others:
                # 零假设的集合**大小匹配**：别用一个更大的集合去比，名次天然会靠前
                donor = gt_by_clip[rng.choice(others)]
                fake = set(rng.sample(sorted(donor), min(len(wanted), len(donor))))
                rn = best_rank(order, fake)
                if rn is not None:
                    null_ranks.append(rn)
                    null_in_topn += 1 if rn <= a.admit else 0
            # 染色 / 原生 分桶。两堆**大小匹配**后再比——
            # 「集合里最靠前的那个」的期望名次本来就随集合大小变，
            # 不匹配的话比的是集合大小不是颜色。
            _t, _n = tint_by_clip.get(clip, (set(), set()))
            _k = min(len(_t), len(_n))
            if _k > 0:
                _ts = set(rng.sample(sorted(_t), _k))
                _ns = set(rng.sample(sorted(_n), _k))
                _tr, _nr = best_rank(order, _ts), best_rank(order, _ns)
                if _tr is not None:
                    tint_ranks.append(_tr)
                if _nr is not None:
                    native_ranks.append(_nr)

            if rec is not None and rec.get("chosen"):
                cr = best_rank(order, {str(rec["chosen"])})
                if cr is not None:
                    chosen_ranks.append(cr)

            # ---- 轮廓这条腿：全库按 Hu 矩排序 ----
            sil_r = None
            if a.silhouette:
                from scripts.probe_motion_signal import _hu_moments
                per_obj = masks.get(obj["id"]) or {}
                avail = sorted(per_obj)
                if len(avail) > 5:
                    st = len(avail) / 5.0
                    avail = [avail[min(len(avail) - 1, int(st * (i + 0.5)))]
                             for i in range(5)]
                descs = [d for d in (_hu_moments(np.asarray(per_obj[f]).astype(bool))
                                     for f in avail) if d is not None]
                sorder = _sil_order(descs, catalog)
                if sorder:
                    sil_r = best_rank(sorder, wanted)
                    if sil_r is not None:
                        sil_ranks.append(sil_r)
                    if others:
                        donor3 = gt_by_clip[rng.choice(others)]
                        fake3 = set(rng.sample(sorted(donor3), min(len(wanted), len(donor3))))
                        srn3 = best_rank(sorder, fake3)
                        if srn3 is not None:
                            sil_null_ranks.append(srn3)
                    half3 = max(1, a.admit // 2)
                    if r is not None and sil_r <= half3 and r > half3:
                        union_sil_saves += 1

            # ---- 形状这条腿：全库按边长比例排序，同样的集合判据 ----
            if shapes and rec is not None:
                obs = ext_pred.get(str(rec.get("id")))
                if obs is not None:
                    sorder = _shape_order(obs, shapes)
                    sr = best_rank(sorder, wanted)
                    if sr is not None:
                        shape_ranks.append(sr)
                    sfr = best_rank(_shape_order(_unflatten(obs), shapes), wanted)
                    if sfr is not None:
                        shape_fix_ranks.append(sfr)
                    if others:
                        donor2 = gt_by_clip[rng.choice(others)]
                        fake2 = set(rng.sample(sorted(donor2), min(len(wanted), len(donor2))))
                        srn = best_rank(sorder, fake2)
                        if srn is not None:
                            shape_null_ranks.append(srn)
                    # 并集：同样的预算 admit，一半给图像一半给形状
                    half = max(1, a.admit // 2)
                    # 形状这条腿取**今天能拿到的最好一档**（压扁校正后），
                    # 不是未校正那一档——拿最弱的变体去算并集会低估两条腿的互补性。
                    sr_best = sfr if sfr is not None else sr
                    if r is not None and sr_best is not None:
                        union_n += 1
                        in_img = r <= half
                        union_img += 1 if in_img else 0
                        if sr_best <= half and not in_img:
                            union_shape_saves += 1


    print(f"\n自检：对上官方记录的裁剪图 {checked} 个，其中分数对不上的 {mismatched} 个")
    if a.index or a.frames > 1:
        print(f"（{'换了索引' if a.index else ''}{'、' if a.index and a.frames > 1 else ''}"
              f"{'多帧聚合' if a.frames > 1 else ''}，跳过和 rebuild.json 的分数对账；"
              f"替代自检是：单帧那一臂必须复现出已知的 0.243）")
    elif checked == 0 or mismatched:
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
        if shape_fix_ranks:
            show("重建 extent + 压扁校正", shape_fix_ranks,
                 sum(1 for x in shape_fix_ranks if x <= a.admit))
        if oracle_ranks:
            show("真值 extent（循环·描述子上限）", oracle_ranks,
                 sum(1 for x in oracle_ranks if x <= a.admit))
        # 结论要按数说，不能无条件印——这行以前是写死的，而实测真实
        # (1329) 和零假设 (2081) 根本没打平，写死的那句话是错的。
        if shape_ranks and shape_null_ranks:
            mr = statistics.median(shape_ranks); mn = statistics.median(shape_null_ranks)
            gap = mn - mr
            print(f"  真集合比零假设靠前 {gap:+.0f} 名"
                  + ("（差得不明显，形状里基本没信号）" if gap < 200 else
                     "（有分离，形状里有信号，只是进不了前几十名）"))
        # **这一档是循环的，别当「几何的天花板」读。** 实测这个数据集对资产是
        # **等比缩放**：真值 extent 的边长比例和资产原生比例 L1 差中位 0.0003。
        # 所以「拿真值 extent 去全库排序」= 拿资产自己的比例去找它自己。
        # 和类名那个坑同构：真值派生出来的量，同时当了查询和键。
        #
        # 但它仍然有用，读法要反过来：**即使给了精确的边长比例，前 64 名也只覆盖
        # 0.350**——4812 个资产里长宽比撞车的太多。所以三个数的边长比例描述子
        # **本身就太粗**，这和几何好不好无关，修几何也顶不破这个数。
        if oracle_ranks and shape_ranks:
            o = sum(1 for x in oracle_ranks if x <= a.admit) / len(oracle_ranks)
            r_ = sum(1 for x in shape_ranks if x <= a.admit) / len(shape_ranks)
            f_ = (sum(1 for x in shape_fix_ranks if x <= a.admit) / len(shape_fix_ranks)
                  if shape_fix_ranks else float("nan"))
            print(f"  天花板 {o:.3f}（真值 extent） vs 现状 {r_:.3f} / 校正后 {f_:.3f}")
            print(f"  注意这一档**循环**（数据集等比缩放，真值 extent 的比例就是资产原生比例），"
                  f"所以 {o:.3f} 不是「几何的天花板」，是**描述子本身的天花板**："
                  f"给了精确比例也只能覆盖这么多。修几何顶不破它。")

    if pool_hit:
        print(f"\n########## 管线真实候选池（retrieve 的 text+image 融合） ##########")
        print(f"  全部节点   覆盖 {sum(pool_hit)}/{len(pool_hit)} = {sum(pool_hit)/len(pool_hit):.3f}")
        if pool_wp:
            _c = sum(pool_wp) / len(pool_wp)
            print(f"  **良定义节点 覆盖 {sum(pool_wp)}/{len(pool_wp)} = {_c:.3f}**")
            print(f"  端到端挑对率 0.209 ÷ 这个覆盖率 = 条件准确率 {0.209/max(_c,1e-9):.2f}")
            print(f"  （只用图像分时覆盖 0.243、条件准确率 ≤0.86。这里才是管线的真实口径）")

    if wellposed_ranks:
        print(f"\n########## 候选生成的余量：覆盖率随候选数怎么走 ##########")
        print("  （只算良定义节点 obj_*；这是**不依赖类名**的纯图像检索）")
        for k in (16, 32, 64, 128, 256, 512, 1024, 2048, 4812):
            c = sum(1 for x in wellposed_ranks if x <= k) / len(wellposed_ranks)
            bar = "#" * int(c * 40)
            print(f"    前 {k:>4} 名  覆盖 {c:.3f}  {bar}")
        print("  读法：曲线在哪里变平，就是「多给候选」这条路的尽头；")
        print("  平得早 => 要换嵌入或换信号，不是加候选数。")

    if wellposed_ranks or illposed_ranks:
        _tot = len(wellposed_ranks) + len(illposed_ranks)
        print(f"\n########## 这个节点在真值里有没有资产答案 ##########")
        print(f"  能对上真值 id 的 {_tot} 个（对不上说明不是真值分割那一档，别读这节）")
        if wellposed_ranks:
            show("有资产真值（obj_*）", wellposed_ranks,
                 sum(1 for x in wellposed_ranks if x <= a.admit))
        if illposed_ranks:
            show("真值说是基本体（rail/block/ground）", illposed_ranks,
                 sum(1 for x in illposed_ranks if x <= a.admit))
            print(f"  后者占 {len(illposed_ranks)/max(_tot,1):.1%}——**真值里没有正确答案**，"
                  f"它们进分母只会把覆盖率稀释掉")

    if len(size_rows) >= 20:
        print(f"\n########## 按裁剪图短边分桶（分辨率是不是主因） ##########")
        print("  资产缩略图 160x160，DINOv2 输入 224x224；视频里的物体要小得多。")
        size_rows.sort(key=lambda t: t[0])
        m, b = len(size_rows), 4
        for i in range(b):
            lo, hi = i * m // b, (i + 1) * m // b
            ch = size_rows[lo:hi]
            if not ch:
                continue
            cov = sum(1 for _s, r in ch if r <= a.admit) / len(ch)
            med = statistics.median([r for _s, r in ch])
            note = "  （样本太少，不解读）" if len(ch) < 10 else ""
            print(f"  第{i+1}桶 短边 [{ch[0][0]:.0f}, {ch[-1][0]:.0f}]px  "
                  f"进前 {a.admit} 名 {cov:.3f}  中位名次 {med:.0f}  n={len(ch)}{note}")
        big = size_rows[3 * m // 4:]
        small = size_rows[:m // 4]
        if len(big) >= 10 and len(small) >= 10:
            cb = sum(1 for _s, r in big if r <= a.admit) / len(big)
            cs = sum(1 for _s, r in small if r <= a.admit) / len(small)
            print(f"  最大那 25% {cb:.3f} vs 最小那 25% {cs:.3f}")
            if cb > 2 * max(cs, 0.001):
                # **别印「分辨率是主因」。** 这个分桶只说明「裁剪图大的那批表现好」，
                # 而降采样阶梯已经证伪了「像素数本身是原因」：把缩略图两侧一起降到
                # 短边 16px，留一视角自匹配仍有 **0.958**（原生 0.990），按实测裁剪
                # 尺寸分布预测覆盖率 0.965，而实测只有 0.284。
                # 所以尺寸是**别的东西的代理**——遮挡、掩码边缘、斜视角、运动模糊、
                # 距离远——不是像素数。两条结论都要一起读。
                print(f"  → 大裁剪图确实更准（{cb/max(cs,1e-9):.1f} 倍）。但**不要据此说"
                      f"分辨率是主因**：降采样阶梯里把两侧一起降到 16px，留一视角仍有"
                      f" 0.958。所以尺寸是遮挡/掩码边缘/斜视角这些东西的**代理**，"
                      f"不是像素数本身。")
            elif abs(cb - cs) < 0.05:
                print(f"  → 大小没关系，分辨率不是主因")
            else:
                print(f"  → 分辨率占一部分")

    if tint_ranks and native_ranks:
        print(f"\n########## 染色的 vs 原生材质的（大小匹配） ##########")
        show("会被染色的真值资产", tint_ranks,
             sum(1 for x in tint_ranks if x <= a.admit))
        show("用 GLB 原生材质的", native_ranks,
             sum(1 for x in native_ranks if x <= a.admit))
        _tc = sum(1 for x in tint_ranks if x <= a.admit) / len(tint_ranks)
        _nc = sum(1 for x in native_ranks if x <= a.admit) / len(native_ranks)
        if _nc > 2 * max(_tc, 0.001):
            print(f"  原生那堆明显好（{_nc:.3f} vs {_tc:.3f}）→ 颜色是主因，去色有救")
        elif abs(_nc - _tc) < 0.05:
            print(f"  两堆基本一样（{_nc:.3f} vs {_tc:.3f}）→ **颜色不是主因**，"
                  f"锅在光照/分辨率/遮挡/掩码边缘，去色那条帮不上")
        else:
            print(f"  原生略好（{_nc:.3f} vs {_tc:.3f}）→ 颜色占一部分，但不是全部")

    if sil_ranks:
        print(f"\n########## 轮廓这条腿（全库按 Hu 矩） ##########")
        show("真·观测轮廓", sil_ranks, sum(1 for x in sil_ranks if x <= a.admit))
        show("零假设·别片段的集合", sil_null_ranks,
             sum(1 for x in sil_null_ranks if x <= a.admit))
        if real_ranks:
            ri = sum(1 for x in real_ranks if x <= a.admit) / len(real_ranks)
            rs = sum(1 for x in sil_ranks if x <= a.admit) / len(sil_ranks)
            print(f"  对比图像那条腿：轮廓 {rs:.3f} vs 图像 {ri:.3f}（同样前 {a.admit} 名）")
        if union_sil_saves:
            print(f"  图像没覆盖、轮廓前 {max(1, a.admit//2)} 覆盖了的："
                  f"{union_sil_saves} 个 → 两条腿互补")
        else:
            print("  轮廓没有救回任何一个图像漏掉的物体")

    if union_n:
        half = max(1, a.admit // 2)
        both = (union_img + union_shape_saves) / union_n
        print(f"\n########## 两条腿是不是在同一批物体上失败 ##########")
        print(f"  同时有两种名次的节点 {union_n} 个，预算 {a.admit} 拆成 {half}+{half}")
        print(f"  只用图像前 {half}      覆盖 {union_img}/{union_n} = {union_img/union_n:.3f}")
        print(f"  形状额外救回来的    {union_shape_saves}/{union_n} = {union_shape_saves/union_n:.3f}")
        print(f"  **并集覆盖 {both:.3f}**（同样 {a.admit} 个候选的预算）")
        if union_shape_saves == 0:
            print(f"  形状一个也没救回来。注意这只说明**当前几何下**两条腿一起失败；"
                  f"真值 extent 那一档的覆盖率才是形状的天花板，别混为一谈")
        else:
            print(f"  两条腿在不同物体上失败 → 候选生成该用两条腿，"
                  f"而且这是同预算下的净赚")
    print(f"""
怎么读：
  进前 {a.admit} 名的比例是关键——那正是「图像进候选{a.admit}」能救回来的上限。
  这个比例接近 0，说明修 retrieve() 的拼接顺序也没用，图像这条路从根上不通。
  真集合和零假设的名次差要先是正的、而且明显，后面的数才有意义。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
