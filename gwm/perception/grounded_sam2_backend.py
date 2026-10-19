"""Detection-provider adapter plus optional multi-frame association and SAM 2.1 propagation."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..taxonomy import slug
from .association import associate_detections
from .base import DetectionProvider, DetectionRecord, Tracks, TrackedObject

class GroundingDinoDetectionProvider:
    """Grounding DINO adapter; it emits records and owns no association policy."""
    name = "grounding_dino"

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

    def _detect_joined(self, image, phrases, proc, model, device, pc):
        """原来的做法：所有短语拼成一串问一次。"""
        text = ". ".join(p.strip().rstrip(".").lower() for p in phrases) + "."
        inputs = proc(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        try:
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=pc["grounding_box_threshold"],
                text_threshold=pc["grounding_text_threshold"], target_sizes=[image.size[::-1]])[0]
        except TypeError:
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=pc["grounding_box_threshold"],
                text_threshold=pc["grounding_text_threshold"], target_sizes=[image.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        return [(float(s), [float(v) for v in b], str(l))
                for s, b, l in zip(res["scores"].tolist(), res["boxes"].tolist(), labels)]

    @staticmethod
    def _iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        return inter / max(area_a + area_b - inter, 1e-9)

    def _detect_per_phrase(self, image, phrases, proc, model, device, pc) -> list[DetectionRecord]:
        """每个短语单独问一遍，再按框的重叠合并。

        合并是必要的：同一个物体可能被好几个短语找到（「crate」和「box」），不合并就
        变成重复检出。保留得分最高的那个，口径和 gwm/synthesis/dedup.py 一致——
        先按得分排序，后来的和已保留的重叠超过阈值就丢掉。
        """
        iou_cut = float(pc.get("detect_per_phrase_iou", 0.6))
        found: list[tuple[float, list[float], str]] = []
        for phrase in phrases:
            text = phrase.strip().rstrip(".").lower() + "."
            inputs = proc(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            try:
                res = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=pc["grounding_box_threshold"],
                    text_threshold=pc["grounding_text_threshold"], target_sizes=[image.size[::-1]])[0]
            except TypeError:
                res = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=pc["grounding_box_threshold"],
                    text_threshold=pc["grounding_text_threshold"], target_sizes=[image.size[::-1]])[0]
            for score, box in zip(res["scores"].tolist(), res["boxes"].tolist()):
                found.append((float(score), [float(v) for v in box], phrase.strip().lower()))
        found.sort(key=lambda row: -row[0])
        # 跨短语的合并只在这里做一次：同一个物体可能被好几个短语找到（crate 和 box）。
        # 后面公共那段还会按 0.7 再去一次重，这里用的阈值更松，专门对付跨短语的重复。
        kept: list[tuple[float, list[float], str]] = []
        for score, box, label in found:
            if any(self._iou(box, other[1]) >= iou_cut for other in kept):
                continue
            kept.append((score, box, label))
        return kept

    def detect(self, image: Image.Image, frame_index: int, phrases: list[str], cfg: dict) -> list[DetectionRecord]:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pc = cfg["perception"]
        proc = AutoProcessor.from_pretrained(str(self.model_dir)); model = AutoModelForZeroShotObjectDetection.from_pretrained(str(self.model_dir)).to(device).eval()
        if pc.get("detect_per_phrase", False):
            # 逐短语分别问，而不是把所有类名拼成一串。拼串的问题是一个框落在多个短语的
            # 词上时返回的是拼接（#67 修的那些词片碎屑就是这么来的），而且短语之间会
            # 互相争夺同一个框。代价是每帧多跑 N 次前向，模型只加载一次所以是线性的。
            raw = self._detect_per_phrase(image, phrases, proc, model, device, pc)
        else:
            raw = self._detect_joined(image, phrases, proc, model, device, pc)
        raw.sort(key=lambda detection: (-detection[0], detection[2], detection[1]))
        keep = []
        width, height = image.size
        for score, bbox, label in raw:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 0.0005 * width * height or area > 0.9 * width * height: continue
            if any(_iou(bbox, previous[1]) > 0.7 for previous in keep): continue
            keep.append((score, bbox, label))
            if len(keep) >= pc["max_objects"]: break
        del model; torch.cuda.empty_cache()
        return [DetectionRecord(detection_id=f"f{frame_index}_{i:03d}", frame_index=frame_index,
                                class_label=snap_label(label, phrases),
                                score=score, bbox=tuple(bbox), source=self.name)
                for i, (score, bbox, label) in enumerate(keep)]


def _normalise_label(text: str) -> str:
    """把标签和查询短语放到同一个口径上再比。

    检测器返回的是**词片**拼起来的字符串，不是我们问的那句话。两种实测到的走样：

    * `##ool` —— BERT 的子词标记。"spool" 被切成 "sp" + "##ool"，框只落在后半片上，
      于是标签里只剩一个带井号的尾巴。
    * `cone _ marker` —— 下划线被当成独立 token，还原的时候两边各加了一个空格。

    这两种在原样比较下都匹配不上任何短语，于是整串走样的文本被当成类名写进程序，
    产出 `ool_1` 这样的物体。规范化只做三件事：去掉子词标记、把下划线当成空格、
    把连续空白压成一个。不做词干还原之类的事——那会引入新的误匹配。
    """
    cleaned = re.sub(r"##", "", (text or "").strip().lower())
    cleaned = cleaned.replace("_", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def snap_label(label: str, phrases: list[str]) -> str:
    """把检测器返回的标签收敛到我们真正问过的某一个短语上。

    Grounding DINO 的提示词是把所有短语用句点拼成一串的，一个框如果同时落在好几个
    短语的词上，返回的标签就是它们的拼接——实测见过 "wooden block cardboard box
    storage box drawer" 这种。这样的类名在库里查不到任何资产，而且写进程序之后
    也没法读。

    三条规则，依次试：

    1. 规范化之后正好等于某个短语。
    2. 某个短语整体出现在标签里（拼接的情况），取**最长的**那个。拼接本身没有可靠的
       主次顺序，所以不去猜哪个是主体；但一个具体的类名总比一串拼接有用。
    3. 反过来，标签是某个短语的一段（词片丢了前半截的情况）。这条容易误伤，所以加两道
       闸：片段至少三个字符，而且只能命中唯一一个短语——"ool" 只可能来自 "spool"，
       但 "box" 可能来自好几个短语，那种情况宁可不认。

    三条都不中就原样返回，不编。
    """
    text = (label or "").strip().lower()
    if not text:
        return label
    norm_text = _normalise_label(text)
    pairs = [(p.strip().lower(), _normalise_label(p)) for p in phrases if p and p.strip()]
    pairs = [(raw, norm) for raw, norm in pairs if norm]
    for raw, norm in pairs:
        if norm == norm_text:
            return raw
    contained = [raw for raw, norm in pairs if norm in norm_text]
    if contained:
        contained.sort(key=lambda c: (-len(c), c))
        return contained[0]
    if len(norm_text) >= 3:
        fragment_of = [raw for raw, norm in pairs if norm_text in norm]
        if len(set(fragment_of)) == 1:
            return fragment_of[0]
    return label

class GroundedSAM2Backend:
    name = "grounded_sam2"
    def __init__(self, gdino_dir: str | Path, sam2_dir: str | Path, detection_provider: DetectionProvider | None = None):
        self.gdino_dir = Path(gdino_dir); self.sam2_dir = Path(sam2_dir)
        self.detection_provider = detection_provider or GroundingDinoDetectionProvider(self.gdino_dir)

    def detect(self, image: Image.Image, phrases: list[str], box_thr: float, text_thr: float, max_objects: int):
        local_cfg = {"perception": {"grounding_box_threshold": box_thr, "grounding_text_threshold": text_thr, "max_objects": max_objects}}
        records = self.detection_provider.detect(image, 0, phrases, local_cfg)
        return [(record.score, list(record.bbox), record.class_label) for record in records]

    def segment(self, frame_files: list[str], frame_indices: list[int], phrases: list[str], cfg: dict) -> Tracks:
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pc = cfg["perception"]
        frames = [Image.open(f).convert("RGB") for f in frame_files]
        W, H = frames[0].size
        detection_keyframes = max(1, int(pc.get("detection_keyframes", 1)))
        association = None
        dets, det_frame = [], 0
        if detection_keyframes == 1:
            # Preserve the original first-frame / middle-frame fallback.
            for cand in (0, len(frames) // 2):
                records = self.detection_provider.detect(frames[cand], frame_indices[cand], phrases, cfg)
                dets = [(record.score, list(record.bbox), record.class_label) for record in records]
                if dets: det_frame = cand; break
            objs = [TrackedObject(id=f"{slug(label.lower(), 24)}_{i+1}", phrase=label, score=score,
                                  source_detection_ids=[f"f{frame_indices[det_frame]}_{i:03d}"])
                    for i, (score, _, label) in enumerate(dets)]
        else:
            positions = sorted(set(int(round(value)) for value in np.linspace(0, len(frames) - 1, min(detection_keyframes, len(frames)))))
            records = []
            for position in positions:
                records.extend(self.detection_provider.detect(frames[position], frame_indices[position], phrases, cfg))
            association = associate_detections(records, cfg)
            objs = []
            for track in association.tracks:
                scores = [record.score for record in track.detections]
                objs.append(TrackedObject(id=track.track_id, phrase=track.class_label, score=float(np.mean(scores)),
                                          identity_hypotheses=track.identity_hypotheses,
                                          source_detection_ids=[record.detection_id for record in track.detections]))
        if not objs:
            return Tracks(objects=[], frame_size=(W, H), backend=self.name,
                          association_diagnostics=association.diagnostics if association else [])
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = Sam2VideoModel.from_pretrained(str(self.sam2_dir), dtype=dtype).to(device).eval()
        proc = Sam2VideoProcessor.from_pretrained(str(self.sam2_dir))
        video = [np.asarray(f) for f in frames]

        if association is not None:
            local_by_global = {global_index: local_index for local_index, global_index in enumerate(frame_indices)}
            for k, track in enumerate(association.tracks):
                session = proc.init_video_session(video=video, inference_device=device, dtype=dtype)
                local_detections = [(local_by_global[record.frame_index], record) for record in track.detections if record.frame_index in local_by_global]
                for local_index, record in local_detections:
                    proc.add_inputs_to_inference_session(inference_session=session, frame_idx=local_index, obj_ids=1,
                                                         input_boxes=[[[float(value) for value in record.bbox]]])
                if not local_detections:
                    del session
                    continue
                start = min(local_index for local_index, _ in local_detections)
                with torch.no_grad():
                    model(inference_session=session, frame_idx=start)
                    for out in model.propagate_in_video_iterator(session, start_frame_idx=start, reverse=False):
                        masks = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                        mask = masks.reshape(-1, H, W)[0].cpu().numpy().astype(bool)
                        if mask.sum() > 0:
                            fi = frame_indices[out.frame_idx]; objs[k].masks[fi] = mask
                            ys, xs = np.where(mask); objs[k].boxes[fi] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                del session; torch.cuda.empty_cache()
            del model; torch.cuda.empty_cache()
            return Tracks(objects=objs, frame_size=(W, H), backend=self.name,
                          association_diagnostics=association.diagnostics)

        boxes = [[float(v) for v in b] for (_, b, _) in dets]

        def propagate(session, obj_slots):
            """Run forward on det_frame then propagate both directions; obj_slots maps output row -> objs index."""
            with torch.no_grad():
                model(inference_session=session, frame_idx=det_frame)
                passes = [dict(start_frame_idx=det_frame, reverse=False)] + ([dict(start_frame_idx=det_frame, reverse=True)] if det_frame > 0 else [])
                for kw in passes:
                    for out in model.propagate_in_video_iterator(session, **kw):
                        masks = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                        m = masks.reshape(-1, H, W).cpu().numpy().astype(bool)
                        fi = frame_indices[out.frame_idx]
                        for row, k in enumerate(obj_slots):
                            if row < m.shape[0] and m[row].sum() > 0:
                                objs[k].masks[fi] = m[row]
                                ys, xs = np.where(m[row]); objs[k].boxes[fi] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        try:  # all objects in one session (one prompt call with all boxes)
            session = proc.init_video_session(video=video, inference_device=device, dtype=dtype)
            proc.add_inputs_to_inference_session(inference_session=session, frame_idx=det_frame, obj_ids=list(range(1, len(boxes) + 1)), input_boxes=[boxes])
            propagate(session, list(range(len(objs))))
        except Exception as e:  # fall back to one session per object
            import logging; logging.getLogger(__name__).warning("multi-object SAM2 session failed (%r); falling back to per-object sessions", e)
            for k, b in enumerate(boxes):
                objs[k].masks.clear(); objs[k].boxes.clear()
                session = proc.init_video_session(video=video, inference_device=device, dtype=dtype)
                proc.add_inputs_to_inference_session(inference_session=session, frame_idx=det_frame, obj_ids=1, input_boxes=[[b]])
                propagate(session, [k])
                del session; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
        return Tracks(objects=objs, frame_size=(W, H), backend=self.name)

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
