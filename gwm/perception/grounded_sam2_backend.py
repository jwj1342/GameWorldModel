"""Grounding DINO (boxes on the first usable frame) + SAM 2.1 video propagation (transformers-native)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .base import Tracks, TrackedObject

def _slug(s: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return (s or "obj")[:24]

class GroundedSAM2Backend:
    name = "grounded_sam2"
    def __init__(self, gdino_dir: str | Path, sam2_dir: str | Path):
        self.gdino_dir = Path(gdino_dir); self.sam2_dir = Path(sam2_dir)

    def detect(self, image: Image.Image, phrases: list[str], box_thr: float, text_thr: float, max_objects: int):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(str(self.gdino_dir)); model = AutoModelForZeroShotObjectDetection.from_pretrained(str(self.gdino_dir)).to(device).eval()
        text = ". ".join(p.strip().rstrip(".").lower() for p in phrases) + "."
        inputs = proc(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad(): out = model(**inputs)
        try:
            res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[image.size[::-1]])[0]
        except TypeError:
            res = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=box_thr, text_threshold=text_thr, target_sizes=[image.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = [(float(s), [float(v) for v in b], str(l)) for s, b, l in zip(res["scores"].tolist(), res["boxes"].tolist(), labels)]
        dets.sort(key=lambda d: -d[0])
        # drop near-duplicate boxes (IoU>0.7) and tiny boxes
        keep = []
        W, H = image.size
        for s, b, l in dets:
            if (b[2] - b[0]) * (b[3] - b[1]) < 0.0005 * W * H: continue
            if (b[2] - b[0]) * (b[3] - b[1]) > 0.9 * W * H: continue
            if any(_iou(b, k[1]) > 0.7 for k in keep): continue
            keep.append((s, b, l))
            if len(keep) >= max_objects: break
        del model; torch.cuda.empty_cache()
        return keep

    def segment(self, frame_files: list[str], frame_indices: list[int], phrases: list[str], cfg: dict) -> Tracks:
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pc = cfg["perception"]
        frames = [Image.open(f).convert("RGB") for f in frame_files]
        W, H = frames[0].size
        # detect on the first frame; if nothing, try the middle frame
        dets, det_frame = [], 0
        for cand in (0, len(frames) // 2):
            dets = self.detect(frames[cand], phrases, pc["grounding_box_threshold"], pc["grounding_text_threshold"], pc["max_objects"])
            if dets: det_frame = cand; break
        objs = [TrackedObject(id=f"{_slug(l)}_{i+1}", phrase=l, score=s) for i, (s, b, l) in enumerate(dets)]
        if not objs:
            return Tracks(objects=[], frame_size=(W, H), backend=self.name)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = Sam2VideoModel.from_pretrained(str(self.sam2_dir), dtype=dtype).to(device).eval()
        proc = Sam2VideoProcessor.from_pretrained(str(self.sam2_dir))
        video = [np.asarray(f) for f in frames]
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
