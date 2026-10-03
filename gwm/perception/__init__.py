"""Perception backends and the versioned Evidence contract."""

from .association import AssociationResult, AssociatedTrack, associate_detections
from .base import DetectionProvider, DetectionRecord
from .contract import adapt_evidence_v1, evidence_schema, validate_evidence
from .motion import estimate_motion, to_world_observations
from .quality import assess_evidence_quality

__all__ = [
    "AssociationResult", "AssociatedTrack", "DetectionProvider", "DetectionRecord",
    "adapt_evidence_v1", "assess_evidence_quality", "associate_detections",
    "estimate_motion", "evidence_schema", "to_world_observations", "validate_evidence",
]
