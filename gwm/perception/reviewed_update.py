"""Bind a reviewed targeted candidate to an existing Evidence observation."""
from __future__ import annotations

from ..feedback.active_perception import request_identifier
from .provenance import declared_video_sha256, evidence_sha256, file_sha256
from .update import apply_evidence_update


def apply_reviewed_targeted_candidate(base: dict, request_bundle: dict, targeted_report: dict,
                                      request_id: str, detection_id: str, identity_decision: dict,
                                      config: dict | None = None) -> tuple[dict, dict]:
    """Require explicit review and decoded PTS before reusing Evidence Update.

    Only an existing object/frame observation can be patched. A candidate never
    creates an identity or a frame registration implicitly.
    """
    requests = [item for item in request_bundle.get("requests", []) if item.get("request_id") == request_id]
    if len(requests) != 1 or request_identifier(requests[0]) != request_id:
        raise ValueError("request ID is missing, duplicated, or inconsistent")
    request = requests[0]
    approval = request.get("review_decision") or {}
    if request.get("review_status") != "approved" or approval.get("status") != "approved" or not approval.get("reviewer"):
        raise ValueError("request requires an explicit reviewer approval")
    if not isinstance(identity_decision, dict) or identity_decision.get("status") != "confirmed" or not identity_decision.get("reviewer"):
        raise ValueError("candidate requires an explicit reviewer identity confirmation")
    source_hash = declared_video_sha256(base)
    binding = request_bundle.get("source_binding") or {}
    provenance = targeted_report.get("provenance") or {}
    if not source_hash or binding.get("video_sha256") != source_hash or provenance.get("video_sha256") != source_hash:
        raise ValueError("request, targeted result, and Evidence video hashes differ")
    digest = evidence_sha256(base)
    if binding.get("evidence_sha256") != digest or provenance.get("evidence_sha256") != digest:
        raise ValueError("request or targeted result refers to different Evidence content")
    if request_id not in targeted_report.get("selected_request_ids", []):
        raise ValueError("targeted result did not select this request ID")
    matches = [item for item in targeted_report.get("results", []) if item.get("request_id") == request_id]
    if len(matches) != 1 or matches[0].get("status") not in {"observed", "partial"}:
        raise ValueError("targeted request has no usable result")
    result = matches[0]
    target = request.get("target") or {}
    if result.get("target") != target:
        raise ValueError("targeted result target differs from approved request")
    object_id, track_id = target.get("evidence_object_id"), target.get("track_id")
    if target.get("program_object_id") != object_id:
        raise ValueError("Program and Evidence object IDs are not verified")
    objects = [item for item in base.get("objects", []) if item.get("id") == object_id and item.get("track_id") == track_id]
    if len(objects) != 1:
        raise ValueError("approved object and track are absent from Evidence")
    candidates = [item for item in result.get("observations", [])
                  if (item.get("detection") or {}).get("detection_id") == detection_id]
    if len(candidates) != 1 or candidates[0].get("identity_status") != "candidate_unverified":
        raise ValueError("candidate detection is absent, duplicated, or already altered")
    candidate = candidates[0]
    if candidate.get("target_object_id") != object_id:
        raise ValueError("candidate target differs from approved object")
    selected = [frame for frame in result.get("selected_frames", []) if frame.get("file") == candidate.get("frame_ref")
                and frame.get("video_time_s") == candidate.get("video_time_s")]
    if len(selected) != 1 or selected[0].get("time_basis") != "decoded_source_pts":
        raise ValueError("candidate lacks a unique decoded source PTS")
    frame = selected[0]
    frame_index, pts = frame.get("source_frame_index"), frame["video_time_s"]
    if (frame.get("video_sha256") != source_hash or candidate.get("video_sha256") != source_hash
            or candidate.get("source_frame_index") != frame_index or candidate.get("time_basis") != "decoded_source_pts"):
        raise ValueError("candidate source frame record or video hash differs from extracted frame")
    image_hash = frame.get("image_sha256")
    if not image_hash or candidate.get("frame_sha256") != image_hash or file_sha256(frame["file"]) != image_hash:
        raise ValueError("candidate frame image hash differs from extracted frame")
    registry = [item for item in base.get("frames", []) if item.get("frame_index") == frame_index]
    if len(registry) != 1 or abs(registry[0]["t"] - pts) > 1e-4:
        raise ValueError("candidate frame index or decoded PTS differs from Evidence registry")
    observations = [item for item in objects[0].get("observations", []) if item.get("frame_index") == frame_index]
    if len(observations) != 1 or abs(observations[0]["t"] - pts) > 1e-4:
        raise ValueError("candidate has no matching existing track observation at decoded PTS")
    detection = candidate["detection"]
    bbox = detection.get("bbox")
    if not isinstance(bbox, dict) or bbox.get("image_size") != [frame.get("width"), frame.get("height")]:
        raise ValueError("candidate bbox does not match selected frame dimensions")
    source = dict(observations[0]["source"])
    source["geometry"] = detection.get("source") or provenance.get("provider") or "unknown"
    update = {"object_id": object_id, "track_id": track_id, "frame_index": frame_index,
              "fields": {"bbox": bbox, "visibility_state": "visible", "source": source}}
    update_provenance = {"source_kind": "perception_provider", "source_ref": detection_id,
                         "video_sha256": source_hash, "trigger_type": "verification_request",
                         "request_ref": request_id, "identity_decision": identity_decision,
                         "request_approval": approval, "frame_pts_s": pts,
                         "source_frame_index": frame_index}
    return apply_evidence_update(base, [update], update_provenance, config)
