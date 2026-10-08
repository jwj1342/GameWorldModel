"""Backend protocols and plain data containers for geometry and segmentation results."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np

# OpenCV 相机（x 右 y 下 z 前）到 three.js 相机（x 右 y 上 z 后）的轴变换，整个感知栈共用
CV2THREE = np.diag([1.0, -1.0, -1.0])

@dataclass
class Geometry:
    frame_indices: list[int]            # indices into the working frame list
    intrinsics: list[np.ndarray]        # per frame 3x3 (at depth resolution)
    cam_to_world: list[np.ndarray]      # per frame 4x4, OpenCV camera convention (x right, y down, z forward), world = first camera unless aligned
    depth: list[np.ndarray]             # per frame HxW float32 (0 = invalid)
    depth_conf: list[np.ndarray] | None = None
    dynamic_mask: list[np.ndarray] | None = None
    scale: str = "relative"             # relative | metric
    backend: str = ""
    notes: str = ""
    def size(self):
        return self.depth[0].shape[1], self.depth[0].shape[0]

@dataclass
class DetectionRecord:
    """One provider observation; association is intentionally external."""
    detection_id: str
    frame_index: int
    class_label: str
    score: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    position_3d: tuple[float, float, float] | None = None
    appearance: tuple[float, ...] | None = None
    source: str = ""

@dataclass
class TrackedObject:
    id: str
    phrase: str
    score: float
    masks: dict[int, np.ndarray] = field(default_factory=dict)   # frame index -> bool HxW (frame resolution)
    boxes: dict[int, tuple] = field(default_factory=dict)        # frame index -> (x1,y1,x2,y2)
    identity_hypotheses: list[dict] = field(default_factory=list)
    source_detection_ids: list[str] = field(default_factory=list)

@dataclass
class Tracks:
    objects: list[TrackedObject]
    frame_size: tuple[int, int]          # (W, H) of the masks
    backend: str = ""
    association_diagnostics: list[dict] = field(default_factory=list)

class GeometryBackend(Protocol):
    name: str
    def estimate(self, frame_files: list[str], frame_indices: list[int], cfg: dict) -> Geometry: ...

class SegmentationBackend(Protocol):
    name: str
    def segment(self, frame_files: list[str], frame_indices: list[int], phrases: list[str], cfg: dict) -> Tracks: ...

class NullSegmentationBackend:
    """什么都不检出。分割后端崩掉时用它兜底，管线退化成只有静态结构的场景，而不是空手而归。"""
    name = "none"
    def __init__(self, cfg: dict): pass
    def segment(self, frame_files, frame_indices, phrases, cfg) -> Tracks:
        from PIL import Image
        with Image.open(frame_files[0]) as im: size = im.size
        return Tracks(objects=[], frame_size=size, backend=self.name)

class DetectionProvider(Protocol):
    name: str
    def detect(self, image: Any, frame_index: int, phrases: list[str], cfg: dict) -> list[DetectionRecord]: ...
