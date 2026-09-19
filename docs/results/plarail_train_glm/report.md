# Run 20260919-145647-0dd8 — clip `plarail_train`

video: `data/clips/trimmed/plarail_train.mp4`  commit: `a415e3c`  node: rack06-15  started: 2026-09-19T14:56:47

## Perception
- geometry backend: vggt (scale relative); segmentation: grounded_sam2; fallbacks: ['geometry:vipe']
- frames: 80 @ 4.0 fps, duration 19.8 s; keyframes: 8; timing: {'frames': 0.6, 'geometry_vggt': 65.8, 'segmentation_grounded_sam2': 93.1, 'evidence': 0.8, 'total': 161.3}
- alignment: {'n_static_points': 35966, 'n_object_points': 2473, 'plane_candidates': [{'idx': 0, 'inliers': 9599, 'angle_deg': 59.1, 'cam_height': 0.3985, 'frac_obj_above': 0.393, 'angle_to_view_deg': 30.9}, {'idx': 1, 'inliers': 4771, 'angle_deg': 34.4, 'cam_height': 0.4008, 'frac_obj_above': 0.671, 'angle_to_view_deg': 56.0}, {'idx': 2, 'inliers': 4790, 'angle_deg': 61.3, 'cam_height': -1.002, 'frac_obj_above': 0.145, 'angle_to_view_deg': 32.2}, {'idx': 3, 'inliers': 3443, 'angle_deg': 61.2, 'cam_height': 0.4293, 'frac_obj_above': 0.489, 'angle_to_view_deg': 28.8}], 'ground': 'candidate_1', 'ground_normal_angle_to_up_deg': 34.4, 'ground_inliers': 4771, 'camera0_height_before_scale': 0.4008, 'scale_factor': 2.9942}

| evidence object | class guess | dynamic | motion guess | conf | frames |
|---|---|---|---|---|---|
| floor_1 | floor | False | static (static class 'floor' (structure), centroid drift 6.923 m ign) | 0.80 | 11 |
| toy_train_2 | toy train | True | trajectory (free path, linear fraction 0.63, yaw range 121) | 0.50 | 17 |
| toy_train_3 | toy train | True | trajectory (free path, linear fraction 0.91, yaw range 359) | 0.50 | 16 |
| floor_4 | floor | False | static (static class 'floor' (structure), centroid drift 2.587 m ign) | 0.80 | 8 |

## Program
- writer: {"seconds": 29.0, "stages": {"camera_static": {"ok": true, "tries": 1}, "objects": {"ok": true, "tries": 1}, "motion": {"ok": true, "tries": 1}}, "fallbacks": [], "final_validation": {"ok": true, "n_errors": 0, "warnings": []}}
- static: 4, objects: 2

| id | class | geom | motion |
|---|---|---|---|
| toy_train_2 | toy_train | asset:toy train [5.972, 0.94, 2.772] | trajectory |
| toy_train_3 | toy_train | asset:toy train [4.576, 0.736, 1.472] | trajectory |

## Feedback loop
| round | score | note |
|---|---|---|
| 0 | 0.12017292369747355 |  |
| 1 | 0.12017292369747355 | Two toy trains have pose issues with low mask IoU and high centroid error; their initial positions are corrected to match evidence centres. |
| 2 | 0.12017292369747355 | Two toy trains have pose errors; their positions are corrected using the evidence centres from the first time step. |

final score: 0.12017292369747355; binding slots: `{"player_spawn": [0.26399999999999935, 1.7999999999999998, -0.0007449293412772398], "goal_volume": {"pos": [17.064, 1.6, -13.46], "extent": [1.5, 2.0, 1.5]}, "walkable": "auto", "collectibles": [], "hazards": []}`

## Playtest
```json
{
 "reached_goal": true,
 "stuck": false,
 "fell": 0,
 "t_goal": 12.200000000000186,
 "deaths": 0,
 "collected": 0,
 "collectibles_total": 0,
 "sim_seconds": 12.200000000000186,
 "wall_ms": 2168,
 "sim_fps": 337.63837638376384,
 "playable": true,
 "review": {
  "related_to_video": false,
  "playable": true,
  "defect": "The generated 3D platformer lacks visual fidelity to the source video, featuring simplified shapes and colors instead of the toy train and environment."
 }
}
```

## Errors / fallbacks

- [perception/geometry_vipe_failed] RuntimeError('vipe not installed: ModuleNotFoundError("No module named \'vipe\'")') → next backend
- [feedback/no_improvement] round 1: best candidate 0.120 <= 0.120 → keep previous program
- [feedback/no_improvement] round 2: best candidate 0.120 <= 0.120 → keep previous program

## Files
- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`