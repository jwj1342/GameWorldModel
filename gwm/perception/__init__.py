"""Perception backends and the versioned Evidence contract."""

from .association import AssociationResult, AssociatedTrack, associate_detections
from .base import DetectionProvider, DetectionRecord
from .contract import adapt_evidence_v1, evidence_schema, validate_evidence
from .motion import estimate_motion, to_world_observations
from .quality import assess_evidence_quality
from .update import apply_evidence_update, evidence_digest, rollback_evidence_update

__all__ = [
    "AssociationResult", "AssociatedTrack", "DetectionProvider", "DetectionRecord",
    "adapt_evidence_v1", "assess_evidence_quality", "associate_detections",
    "apply_evidence_update", "estimate_motion", "evidence_digest", "evidence_schema",
    "rollback_evidence_update", "to_world_observations", "validate_evidence",
]
