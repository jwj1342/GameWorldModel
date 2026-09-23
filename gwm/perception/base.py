"""Backend protocols and plain data containers for geometry and segmentation results."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np

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

class DetectionProvider(Protocol):
    name: str
    def detect(self, image: Any, frame_index: int, phrases: list[str], cfg: dict) -> list[DetectionRecord]: ...
