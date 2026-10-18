"""给程序里的 asset 节点挑一个具体资产，结果写进 `geom.ref`。

分工：模型（或直译）只负责说「这里有一个火车那么大的东西」，也就是 `query` 和 `extent`；
挑哪一个具体模型是这一步做的，因为它需要看画面，而模型那一步手上只有文字证据。
`ref` 是事后标注，不是模型要预测的字段——训练数据里不带它。

挑的办法是文字分数和图像分数加权：文字来自资产名与 kit 名，图像来自把视频里这个物体
抠出来、和资产缩略图比 DINOv2 相似度。只有文字的时候查 "toy train" 会选中
`train-connector` 这种零件，加上图像才分得出哪个像视频里那辆车。
"""
from __future__ import annotations

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


def choose_assets(program: dict, evidence: dict, frames: list[dict], masks: dict,
                  catalog: list, index=None, embedder=None,
                  image_weight: float = 0.6, min_score: float = 0.45,
                  top_k: int = 5) -> list[dict]:
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
        record["candidates"] = [{"asset_id": a.id, "score": s} for a, s in hits]
        # 图像那一路的最高分单独记下来。它和文字分是两回事：文字分决定收不收，
        # 图像分决定选哪个。记它有两个用处——事后能看出「这个物体为什么选了这个资产」，
        # 以及它本身是一个连续信号：背景碎片和库里任何东西都不像，分应该明显低。
        if image_scores is not None:
            try:
                record["image_score_best"] = round(float(max(image_scores.values())), 4)
            except (AttributeError, ValueError, TypeError):
                pass
        if hits and hits[0][1] >= min_score:
            geom["ref"] = hits[0][0].id
            record.update(chosen=hits[0][0].id, score=hits[0][1])
            if "image_score_best" in record:
                geom["ref_image_score"] = record["image_score_best"]
        else:
            record["chosen"] = None
        decisions.append(record)
    return decisions
