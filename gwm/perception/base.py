"""Backend protocols and plain data containers for geometry and segmentation results."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
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
class TrackedObject:
    id: str
    phrase: str
    score: float
    masks: dict[int, np.ndarray] = field(default_factory=dict)   # frame index -> bool HxW (frame resolution)
    boxes: dict[int, tuple] = field(default_factory=dict)        # frame index -> (x1,y1,x2,y2)

@dataclass
class Tracks:
    objects: list[TrackedObject]
    frame_size: tuple[int, int]          # (W, H) of the masks
    backend: str = ""

class GeometryBackend(Protocol):
    name: str
    def estimate(self, frame_files: list[str], frame_indices: list[int], cfg: dict) -> Geometry: ...

class SegmentationBackend(Protocol):
    name: str
    def segment(self, frame_files: list[str], frame_indices: list[int], phrases: list[str], cfg: dict) -> Tracks: ...
