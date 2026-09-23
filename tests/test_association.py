import copy
import random

import numpy as np

from gwm.perception.association import associate_detections, to_tracks
from gwm.perception.base import DetectionRecord


ASSOCIATION = {
    "max_frame_gap": 3,
    "max_bbox_center_distance_ratio": 3.0,
    "max_position_3d_distance": 5.0,
    "min_class_similarity": 0.5,
    "min_match_score": 0.30,
    "min_ambiguous_score": 0.20,
    "ambiguity_margin": 0.06,
    "max_identity_candidates": 3,
    "allow_class_mismatch": False,
    "time_decay_frames": 10.0,
    "weights": {"class": 0.10, "bbox_iou": 0.25, "mask_iou": 0.10, "position_3d": 0.10, "appearance": 0.40, "time": 0.05},
}


def _d(detection_id, frame, bbox, label="cube", appearance=None, position=None, mask=None):
    return DetectionRecord(detection_id=detection_id, frame_index=frame, class_label=label, score=0.9,
                           bbox=tuple(float(value) for value in bbox), appearance=appearance,
                           position_3d=position, mask=mask, source="synthetic")


def _signature(result):
    return [
        (track.track_id, [detection.detection_id for detection in track.detections], track.identity_hypotheses)
        for track in result.tracks
    ]


def test_late_object_creates_new_track():
    records = [
        _d("a0", 0, [0, 0, 10, 10]),
        _d("a1", 1, [1, 0, 11, 10]),
        _d("a2", 2, [2, 0, 12, 10]),
        _d("late2", 2, [50, 0, 60, 10], label="ball"),
    ]
    result = associate_detections(records, ASSOCIATION)
    assert len(result.tracks) == 2
    assert [d.detection_id for d in result.tracks[0].detections] == ["a0", "a1", "a2"]
    assert result.tracks[1].detections[0].detection_id == "late2"


def test_short_occlusion_recovers_existing_track():
    records = [_d("a0", 0, [0, 0, 10, 10]), _d("a3", 3, [1, 0, 11, 10])]
    result = associate_detections(records, ASSOCIATION)
    assert len(result.tracks) == 1
    assert [d.frame_index for d in result.tracks[0].detections] == [0, 3]
    strict = copy.deepcopy(ASSOCIATION); strict["max_frame_gap"] = 2
    assert len(associate_detections(records, strict).tracks) == 2


def test_similar_objects_cross_without_identity_swap_when_appearance_is_available():
    records = [
        _d("a0", 0, [0, 0, 10, 10], appearance=(1.0, 0.0)),
        _d("b0", 0, [20, 0, 30, 10], appearance=(-1.0, 0.0)),
        _d("a1", 1, [20, 0, 30, 10], appearance=(1.0, 0.0)),
        _d("b1", 1, [0, 0, 10, 10], appearance=(-1.0, 0.0)),
    ]
    result = associate_detections(records, ASSOCIATION)
    assert len(result.tracks) == 2
    assert [d.detection_id for d in result.tracks[0].detections] == ["a0", "a1"]
    assert [d.detection_id for d in result.tracks[1].detections] == ["b0", "b1"]


def test_ambiguous_match_splits_and_keeps_identity_candidates():
    records = [
        _d("a0", 0, [0, 0, 10, 10]),
        _d("b0", 0, [2, 0, 12, 10]),
        _d("mid1", 1, [1, 0, 11, 10]),
    ]
    result = associate_detections(records, ASSOCIATION)
    assert len(result.tracks) == 3
    split = result.tracks[2]
    assert split.identity_hypotheses[0]["kind"] == "identity"
    assert {candidate["ref"] for candidate in split.identity_hypotheses[0]["candidates"]} == {"cube_1", "cube_2"}
    converted = to_tracks(result, (100, 80))
    assert converted.objects[2].identity_hypotheses == split.identity_hypotheses


def test_implausible_jump_is_not_wrongly_merged_and_splits_track():
    records = [_d("a0", 0, [0, 0, 10, 10]), _d("jump1", 1, [100, 0, 110, 10])]
    result = associate_detections(records, ASSOCIATION)
    assert len(result.tracks) == 2
    assert all(len(track.detections) == 1 for track in result.tracks)


def test_class_incompatibility_prevents_wrong_merge():
    records = [_d("box0", 0, [0, 0, 10, 10], label="box"), _d("ball1", 1, [0, 0, 10, 10], label="ball")]
    result = associate_detections(records, ASSOCIATION)
    assert len(result.tracks) == 2


def test_all_optional_signals_contribute_when_available():
    mask = np.zeros((12, 12), dtype=bool); mask[2:8, 2:8] = True
    records = [
        _d("a0", 0, [2, 2, 8, 8], appearance=(1.0, 0.0), position=(0.0, 0.0, 0.0), mask=mask),
        _d("a1", 1, [2, 2, 8, 8], appearance=(1.0, 0.0), position=(0.1, 0.0, 0.0), mask=mask.copy()),
    ]
    result = associate_detections(records, ASSOCIATION)
    associated = next(diagnostic for diagnostic in result.diagnostics if diagnostic["code"] == "associated")
    assert {"class", "bbox_iou", "mask_iou", "position_3d", "appearance", "time"} <= set(associated["components"])


def test_empty_detections_and_deterministic_output():
    assert associate_detections([], ASSOCIATION).tracks == []
    records = [_d("b0", 0, [20, 0, 30, 10]), _d("a0", 0, [0, 0, 10, 10]),
               _d("b1", 1, [21, 0, 31, 10]), _d("a1", 1, [1, 0, 11, 10])]
    expected = _signature(associate_detections(records, ASSOCIATION))
    for seed in range(5):
        shuffled = copy.copy(records); random.Random(seed).shuffle(shuffled)
        assert _signature(associate_detections(shuffled, ASSOCIATION)) == expected
