"""Detection-provider adapter plus optional multi-frame association and SAM 2.1 propagation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .association import associate_detections
from .base import DetectionProvider, DetectionRecord, Tracks, TrackedObject

def _slug(s: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return (s or "obj")[:24]

class GroundingDinoDetectionProvider:
    """Grounding DINO adapter; it emits records and owns no association policy."""
    name = "grounding_dino"

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

    def detect(self, image: Image.Image, frame_index: int, phrases: list[str], cfg: dict) -> list[DetectionRecord]:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pc = cfg["perception"]
        proc = AutoProcessor.from_pretrained(str(self.model_dir)); model = AutoModelForZeroShotObjectDetection.from_pretrained(str(self.model_dir)).to(device).eval()
        text = ". ".join(p.strip().rstrip(".").lower() for p in phrases) + "."
        inputs = proc(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad(): out = model(**inputs)
        try:
            res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=pc["grounding_box_threshold"], text_threshold=pc["grounding_text_threshold"], target_sizes=[image.size[::-1]])[0]
        except TypeError:
            res = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=pc["grounding_box_threshold"], text_threshold=pc["grounding_text_threshold"], target_sizes=[image.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        raw = [(float(s), [float(v) for v in b], str(l)) for s, b, l in zip(res["scores"].tolist(), res["boxes"].tolist(), labels)]
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
        return [DetectionRecord(detection_id=f"f{frame_index}_{i:03d}", frame_index=frame_index, class_label=label,
                                score=score, bbox=tuple(bbox), source=self.name)
                for i, (score, bbox, label) in enumerate(keep)]

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
            objs = [TrackedObject(id=f"{_slug(label)}_{i+1}", phrase=label, score=score,
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
