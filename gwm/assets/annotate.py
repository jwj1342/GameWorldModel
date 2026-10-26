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


def choose_assets(program: dict, evidence: dict, frames: list[dict], masks: dict,
                  catalog: list, index=None, embedder=None,
                  image_weight: float = 0.6, min_score: float = 0.45,
                  top_k: int = 5, shapes: dict | None = None,
                  shape_weight: float = 0.0) -> list[dict]:
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

        hits = retrieve(query, catalog, k=top_k, image_scores=image_scores,
                        image_weight=image_weight if image_scores else 0.0)
        # 形状重排：把观测到的边长比例用进「选哪个资产」这个离散选择。
        # shape_weight 为 0（默认）时完全不动，行为和以前一致。
        if shape_weight > 0 and shapes:
            obs_ratio = observed_edge_ratios(obj)
            if obs_ratio:
                # A shape score cannot make a text-ineligible asset acceptable.
                # Rank within the eligible set so a low-text first hit cannot
                # hide a valid hit and turn the decision into a fallback.
                hits = [(a, score) for a, score in hits if score >= min_score]
                hits = shape_rerank(hits, obs_ratio, shapes, weight=shape_weight)
                record["shape_reranked"] = True
                record["observed_ratio"] = [round(v, 3) for v in obs_ratio]
        record["candidates"] = [{"asset_id": a.id, "score": s} for a, s in hits]
        # 图像那一路的最高分记进**决策记录**，不进程序。程序是一份受校验的契约：
        # schema 里 assetGeom 的 additionalProperties 是 false，往 geom 里塞一个
        # ref_image_score 会让每个资产节点都校验失败——试过，十二条片段只剩两条能跑完，
        # 资产覆盖率掉到 0。诊断信息归 manifest，不归程序。
        if image_scores is not None:
            try:
                record["image_score_best"] = round(float(max(image_scores.values())), 4)
            except (AttributeError, ValueError, TypeError):
                pass
        if hits and hits[0][1] >= min_score:
            geom["ref"] = hits[0][0].id
            record.update(chosen=hits[0][0].id, score=hits[0][1])
        else:
            record["chosen"] = None
        decisions.append(record)
    return decisions
