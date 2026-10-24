"""给程序里的 asset 节点挑一个具体资产，结果写进 `geom.ref`。

分工：模型（或直译）只负责说「这里有一个火车那么大的东西」，也就是 `query` 和 `extent`；
挑哪一个具体模型是这一步做的，因为它需要看画面，而模型那一步手上只有文字证据。
`ref` 是事后标注，不是模型要预测的字段——训练数据里不带它。

挑的办法是文字分数和图像分数加权：文字来自资产名与 kit 名，图像来自把视频里这个物体
抠出来、和资产缩略图比 DINOv2 相似度。只有文字的时候查 "toy train" 会选中
`train-connector` 这种零件，加上图像才分得出哪个像视频里那辆车。
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .retrieve import retrieve

# 缩略图的背景色，抠图时用同一个值填充，免得模型把背景差异当成形状差异
THUMB_BG = (242, 242, 242)


def bbox_of(observation: dict) -> tuple[list[float], list[int] | None] | None:
    """取出 (xyxy 四个数, 这个框所在的图像尺寸)。

    证据里的 bbox 是 `{format, values, space, image_size}` 这种带单位的结构，而不是裸的
    四元组，而且它的坐标是在**掩码那一档分辨率**上的（几何后端把帧缩到 14 的倍数），
    和原帧不是一个尺寸。拿它直接去裁原帧会裁错地方，所以这里把尺寸一并带出去。
    """
    box = observation.get("bbox")
    if isinstance(box, dict):
        values, size = box.get("values"), box.get("image_size")
        if box.get("format") not in (None, "xyxy") or not (isinstance(values, (list, tuple)) and len(values) == 4):
            return None
        return [float(v) for v in values], (list(size) if isinstance(size, (list, tuple)) and len(size) == 2 else None)
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(v) for v in box], None
    return None


def best_observation(obj: dict) -> dict | None:
    """挑这个物体看得最清楚的一帧：框的面积最大的那一帧。"""
    best, best_area = None, -1.0
    for ob in obj.get("observations", []):
        parsed = bbox_of(ob)
        if parsed is None:
            continue
        box = parsed[0]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area > best_area:
            best, best_area = ob, area
    return best


def crop_object(frame_rgb: np.ndarray, mask: np.ndarray, bbox, pad: float = 0.12):
    """按掩码把物体抠到中性背景上，再按包围盒裁剪。失败返回 None。

    必须抠掉背景：资产缩略图是干净底色上的单个物体，拿一整块带房间背景的裁剪去比，
    比出来的是「这两张图的背景像不像」。
    """
    height, width = frame_rgb.shape[:2]
    if mask.shape != (height, width):
        from ..perception.masks import resize_mask
        mask = resize_mask(mask, (width, height))
    if not mask.any():
        return None
    composited = np.where(mask[:, :, None], frame_rgb, np.array(THUMB_BG, dtype=frame_rgb.dtype))
    x0, y0, x1, y1 = (float(v) for v in bbox)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = max(0, int(x0 - dx)), max(0, int(y0 - dy))
    x1, y1 = min(width, int(x1 + dx)), min(height, int(y1 + dy))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return composited[y0:y1, x0:x1]


def load_shape_cache(cfg: dict | None = None) -> dict:
    """资产 id → 边长比例（三边排序后除以最长边）。没有缓存就返回空，行为不变。

    由 `scripts/build_asset_shape_cache.py` 预计算。路径从配置
    `assets.shape_cache` 读，其次环境变量 GWM_ASSET_SHAPES。
    """
    import os
    path = ((cfg or {}).get("assets") or {}).get("shape_cache") or os.environ.get("GWM_ASSET_SHAPES")
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, list) and len(v) == 3}


def observed_edge_ratios(obj: dict | None) -> list[float] | None:
    """观测到的边长比例，口径和资产缓存**完全一致**：三边排序后除以最长边。

    两边口径不一致的话这个重排就是在比噪声。资产那边是轴对齐包围盒的三边，
    这边取逐帧包围盒尺寸的中位数——都是「这个物体三个方向上各有多长」。
    """
    obbs = (obj or {}).get("obb") or []
    sizes = [b.get("size") for b in obbs if isinstance(b.get("size"), list) and len(b["size"]) == 3]
    if not sizes:
        return None
    med = np.median(np.asarray(sizes, float), axis=0)
    if not np.isfinite(med).all() or med.max() <= 1e-9:
        return None
    return [float(v) for v in np.sort(med)[::-1] / med.max()]


def shape_rerank(hits: list, observed: list[float] | None, shapes: dict,
                 weight: float = 0.5) -> list:
    """按「资产的边长比例和观测差多少」重排候选。

    **为什么这是创新点一在管线里的落点。** `shape_kind_accuracy` 结构性恒零
    （`effective_shape()` 对解析到资产的节点返回 "mesh"，而真值全是基本体），
    所以它量不出任何改进。真正有数的是 `shape_ok`——它比的正是**三条边的比例**。
    而资产检索原来只看类名和图像，`Asset` 里一个几何字段都没有：**观测到的包围盒
    比例这个纯几何量，完全没参与「选哪个资产」这个离散选择**。

    这就是「用连续稠密表征去监督离散的符号程序」最直接的一个实例。

    只重排、不改收不收。`retrieve()` 的注释写明文字分决定收不收、图像分只在候选里
    重排，理由是量纲不同、混进阈值会误杀。形状分同理：它是个 0~1 的比例距离，
    和文字分不可比，所以只参与排序。

    观测拿不到比例（没有 obb）时原样返回，行为不变。
    """
    if not observed or not shapes or len(hits) < 2:
        return hits
    obs = np.asarray(observed, float)

    def distance(asset_id: str) -> float:
        r = shapes.get(asset_id)
        return 1.0 if r is None else float(np.abs(np.asarray(r, float) - obs).mean())

    scored = [(a, s, distance(a.id)) for a, s in hits]
    best_text = max((s for _a, s, _d in scored), default=1.0) or 1.0
    scored.sort(key=lambda t: (-((1 - weight) * (t[1] / best_text) + weight * (1 - t[2])), t[0].id))
    return [(a, s) for a, s, _d in scored]


def residual_pick(clouds: list, candidates: list, root, top: int = 0,
                  n_views: int = 7) -> tuple:
    """**用配准残差从候选里挑**：把每个候选网格缩到观测点云的尺度，ICP 配一次，
    取残差最小的。返回 (资产, 残差) 或 (None, None)。

    这是「用连续稠密表征监督离散符号选择」的落地形式。梯度传不过去不要紧——
    「哪个网格配得上这团点」这个判据本来就不需要梯度。

    实测（assetgt60，74 个带真值 ref 的物体，按样本加权）：

        真值 + 32 个随机干扰          挑对 0.284（随机猜 0.030，9.4 倍）
        真值 + 32 个**形状最接近**的干扰 挑对 **0.243**（随机猜 0.030，**8.0 倍**）

    第二行是密度匹配的零假设——随机资产长得和目标完全不像，而真实候选集个个都像。
    换成难干扰项只掉了 0.04，说明残差不是靠排除「长得完全不像的」在赢。
    真值静止的物体上难干扰项 33 选 1 挑对 0.500。

    **尺度要缩到逐帧包围盒的中位对角线上，不能用并帧的。** 并帧点云随视角扫动，
    实测比逐帧中位大 **1.89 倍**，而受控扫描说模板相对点云大 1.2 倍就越过对照、
    1.3 倍崩掉。这个坑踩过一次（资产模板的角度 RMSE 80.92°，改掉之后 32.54°）。
    偏小是安全的，所以宁可缩小一点。

    **已知边界：物体在转的时候这条就垮**（periodic_rotate 挑对 0.000、spin 0.125，
    而静止 0.500）。调用方拿不准的时候该退回文字那条，而不是硬挑。
    """
    import numpy as _np
    from ..perception.registration import icp
    from .mesh import load_points

    if not clouds or not candidates:
        return None, None
    per_frame = [float(_np.linalg.norm(_np.asarray(c, float).max(0) - _np.asarray(c, float).min(0)))
                 for c in clouds if len(_np.asarray(c, float)) >= 10]
    if not per_frame:
        return None, None
    obs_diag = float(_np.median(per_frame))
    # **用几帧，不是一帧。** 一帧只是一个局部视角，同一包围盒比例下完全不同的物体
    # 在单个视角上可能都配得差不多。实测（33 个形状最接近的候选里挑对的比例）：
    #
    #     1 帧 0.149 ／ 3 帧 0.243 ／ **7 帧 0.297** ／ 15 帧 0.284 ／ 全部 0.311
    #
    # 平台在 7 帧，静止物体上 0.625。另外**起点不能放在第 0 帧**：只用首帧 0.149、
    # 只用中间帧 0.243，首帧的可见面最偏。均匀抽时起点放到半个间隔处，避开两端。
    #
    # 已试过并否掉的：绕竖轴加 4/8 个初始偏航、每帧取最小残差，反而从 0.297 掉到
    # 0.216——取最小等于给每个候选更大的搜索预算，而错的候选从中得益更多。
    # 「挑哪个」这类任务里，候选之间的分必须可比。
    if 0 < n_views < len(clouds):
        stride = len(clouds) / float(n_views)
        picks = [clouds[min(len(clouds) - 1, int(stride * (i + 0.5)))] for i in range(n_views)]
    else:
        picks = clouds
    queries = [_np.asarray(c, float) for c in picks if len(_np.asarray(c, float)) >= 30]
    if not queries:
        return None, None
    best, best_rmse = None, float("inf")
    for asset in (candidates[:top] if top > 0 else candidates):
        try:
            pts = load_points(Path(root) / asset.path, samples=3000)
        except (OSError, ValueError):
            continue
        if pts is None or len(pts) < 10:
            continue
        diag = float(_np.linalg.norm(pts.max(0) - pts.min(0)))
        if diag <= 1e-9 or obs_diag <= 1e-9:
            continue
        model = (pts - pts.mean(0)) * (obs_diag / diag)
        rs = []
        for q in queries:
            step = icp(q, model)
            if step["ok"] and step["rmse"] is not None:
                rs.append(float(step["rmse"]))
        if not rs:
            continue
        # 取中位数不取均值：某一帧掩码漏了一块就会拉出离谱的残差。
        score = float(_np.median(rs))
        if score < best_rmse:
            best_rmse, best = score, asset
    return best, (best_rmse if best is not None else None)


def choose_assets(program: dict, evidence: dict, frames: list[dict], masks: dict,
                  catalog: list, index=None, embedder=None,
                  image_weight: float = 0.6, min_score: float = 0.45,
                  top_k: int = 5, shapes: dict | None = None,
                  shape_weight: float = 0.0, image_admit: int = 0,
                  image_gate_rank: int = 0, clouds: dict | None = None,
                  residual_top: int = 0, root=None) -> list[dict]:
    """就地给每个 asset 节点填上 `geom.ref`，返回每个决策的记录。"""
    from PIL import Image

    # 程序里的 id 是证据 id 过了一遍 slug 的（会截断、会换掉非法字符），所以两边
    # 不能直接对上。两种写法都登记进来，掩码仍然按证据 id 取——masks.npz 的键是证据 id。
    from ..taxonomy import slug
    by_id = {}
    for obj in evidence.get("objects", []):
        by_id[obj["id"]] = obj
        by_id.setdefault(slug(obj["id"]), obj)
    frame_by_index = {f.get("index"): f for f in frames}
    decisions = []
    for node in program.get("objects", []):
        geom = node.get("geom", {})
        if geom.get("kind") != "asset":
            continue
        query = geom.get("query") or node.get("class") or ""
        record = {"id": node["id"], "query": query, "image_used": False}

        image_scores = None
        obj = by_id.get(node["id"])
        observation = best_observation(obj) if obj else None
        # 走不到图像这一路的时候要说清楚卡在哪一步。只写一个「没用图像」的话，
        # id 对不上和索引没加载这两种完全不同的毛病看起来一模一样。
        if index is None or embedder is None:
            record["image_skipped"] = "no_index"
        elif obj is None:
            record["image_skipped"] = f"no_evidence_object_for_{node['id']}"
        elif observation is None:
            record["image_skipped"] = "no_observation_with_bbox"
        if observation is not None and index is not None and embedder is not None:
            frame = frame_by_index.get(observation.get("frame_index"))
            mask = (masks.get(obj["id"]) or {}).get(observation.get("frame_index"))
            if not frame:
                record["image_skipped"] = f"no_frame_{observation.get('frame_index')}"
            elif mask is None:
                record["image_skipped"] = f"no_mask_for_{obj['id']}@{observation.get('frame_index')}"
            if frame and mask is not None:
                try:
                    rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
                    box, box_size = bbox_of(observation)
                    # 框是在掩码那一档分辨率上给的，按宽度比例缩放到原帧坐标系
                    if box_size and box_size[0]:
                        k = rgb.shape[1] / float(box_size[0])
                        box = [v * k for v in box]
                    crop = crop_object(rgb, np.asarray(mask), box)
                    if crop is not None:
                        vector = embedder.embed_images([Image.fromarray(crop)])[0]
                        image_scores = index.scores(vector)
                        record["image_used"] = True
                        record["frame_index"] = observation.get("frame_index")
                    record["evidence_id"] = obj["id"]
                except (OSError, ValueError, KeyError) as error:
                    record["image_error"] = f"{type(error).__name__}: {error}"[:160]

        # 残差挑资产要在**大候选集**上做才有意义：图像嵌入把正确资产放在中位
        # 第 365 名，前 5 个里基本不会有它。所以开了残差就把 k 放大到 residual_top。
        _k = max(top_k, residual_top) if residual_top > 0 else top_k
        hits = retrieve(query, catalog, k=_k, image_scores=image_scores,
                        image_weight=image_weight if image_scores else 0.0,
                        image_admit=image_admit)
        # 形状重排：把观测到的边长比例用进「选哪个资产」这个离散选择。
        # shape_weight 为 0（默认）时完全不动，行为和以前一致。
        if shape_weight > 0 and shapes:
            obs_ratio = observed_edge_ratios(obj)
            if obs_ratio:
                hits = shape_rerank(hits, obs_ratio, shapes, weight=shape_weight)
                record["shape_reranked"] = True
                record["observed_ratio"] = [round(v, 3) for v in obs_ratio]
        # **用配准残差从候选里挑。** 这是「连续稠密表征监督离散符号选择」的落地形式：
        # 文字和图像只负责**提候选**，谁配得上这团点由几何说了算。见 residual_pick。
        _cloud = (clouds or {}).get(obj["id"]) if obj else None
        if residual_top > 0 and _cloud and root is not None and hits:
            _pick, _rmse = residual_pick(_cloud, [a for a, _s in hits], root,
                                         top=residual_top)
            if _pick is not None:
                _by_id = {a.id: s for a, s in hits}
                record["residual_rmse"] = round(float(_rmse), 5)
                record["residual_rank"] = next(
                    (i for i, (a, _s) in enumerate(hits) if a.id == _pick.id), None)
                hits = [(_pick, _by_id.get(_pick.id, 0.0))] + [
                    pair for pair in hits if pair[0].id != _pick.id]
        record["candidates"] = [{"asset_id": a.id, "score": s} for a, s in hits[:5]]
        # 图像那一路的最高分记进**决策记录**，不进程序。程序是一份受校验的契约：
        # schema 里 assetGeom 的 additionalProperties 是 false，往 geom 里塞一个
        # ref_image_score 会让每个资产节点都校验失败——试过，十二条片段只剩两条能跑完，
        # 资产覆盖率掉到 0。诊断信息归 manifest，不归程序。
        if image_scores is not None:
            try:
                record["image_score_best"] = round(float(max(image_scores.values())), 4)
            except (AttributeError, ValueError, TypeError):
                pass
        # 收不收这个资产。原来只有一道门：**文字分**过 min_score。
        #
        # 图像塞进来的候选文字分恒为 0，所以就算融合之后排第一也必然被这道门挡掉、
        # 退回基本体——「图像进候选」这个开关在这道门下面从原理上不可能兑现。
        # 所以图像这一路要有自己的门，而且不能是拍脑袋的绝对阈值（DINOv2 比真实
        # 照片和低多边形渲染，余弦值天然低，量纲和文字分对不上，这是当初不混进
        # 阈值的理由，仍然成立）。
        #
        # 用**本次裁剪图在全库里的名次**当判据：名次是自归一化的，不受这一张图
        # 整体相似度高低的影响。标定来自实测——本片段真值资产的中位名次 365、
        # 进前 32 名 0.176，而别片段集合的零假设是 1117 / 0.031。名次越靠前，
        # 富集倍数越高。
        chosen_pair = hits[0] if hits else None
        record["chosen"] = None
        if chosen_pair is not None:
            top_asset, top_text = chosen_pair
            passed = top_text >= min_score
            # 文字分为 0 说明这个候选是图像塞进来的——自检要数它，否则又会出现
            # 「旋钮没点着却以为测到了」那种事。
            record["admitted_by_image"] = bool(image_scores) and top_text <= 0.0
            if not passed and image_scores and image_gate_rank > 0:
                own = float(image_scores.get(top_asset.id, 0.0))
                rank = 1 + sum(1 for v in image_scores.values() if v > own)
                record["image_gate_rank"] = rank
                passed = rank <= image_gate_rank
            if passed:
                geom["ref"] = top_asset.id
                record.update(chosen=top_asset.id, score=top_text)
        decisions.append(record)
    return decisions
