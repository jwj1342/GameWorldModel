# Run 20260919-143631-e577 — clip `plarail_train`

video: `data/clips/trimmed/plarail_train.mp4`  commit: `a415e3c`  node: rack12-06  started: 2026-09-19T14:36:31

## Perception
- geometry backend: vggt (scale relative); segmentation: grounded_sam2; fallbacks: ['geometry:vipe']
- frames: 80 @ 4.0 fps, duration 19.8 s; keyframes: 8; timing: {'frames': 0.8, 'geometry_vggt': 76.2, 'segmentation_grounded_sam2': 113.3, 'evidence': 6.7, 'total': 198.0}
- alignment: {'n_static_points': 35966, 'n_object_points': 2473, 'plane_candidates': [{'idx': 0, 'inliers': 9599, 'angle_deg': 59.1, 'cam_height': 0.3985, 'frac_obj_above': 0.393, 'angle_to_view_deg': 30.9}, {'idx': 1, 'inliers': 4771, 'angle_deg': 34.4, 'cam_height': 0.4008, 'frac_obj_above': 0.671, 'angle_to_view_deg': 56.0}, {'idx': 2, 'inliers': 4790, 'angle_deg': 61.3, 'cam_height': -1.002, 'frac_obj_above': 0.145, 'angle_to_view_deg': 32.2}, {'idx': 3, 'inliers': 3443, 'angle_deg': 61.2, 'cam_height': 0.4293, 'frac_obj_above': 0.489, 'angle_to_view_deg': 28.8}], 'ground': 'candidate_1', 'ground_normal_angle_to_up_deg': 34.4, 'ground_inliers': 4771, 'camera0_height_before_scale': 0.4008, 'scale_factor': 2.9942}

| evidence object | class guess | dynamic | motion guess | conf | frames |
|---|---|---|---|---|---|
| floor_1 | floor | False | static (static class 'floor' (structure), centroid drift 6.923 m ign) | 0.80 | 11 |
| toy_train_2 | toy train | True | trajectory (free path, linear fraction 0.63, yaw range 121) | 0.50 | 17 |
| toy_train_3 | toy train | True | trajectory (free path, linear fraction 0.91, yaw range 359) | 0.50 | 16 |
| floor_4 | floor | False | static (static class 'floor' (structure), centroid drift 2.587 m ign) | 0.80 | 8 |

## Program
- writer: {"seconds": 0.0, "mode": "direct"}
- static: 3, objects: 2

| id | class | geom | motion |
|---|---|---|---|
| toy_train_2 | toy train | asset:toy train [4.929222621892557, 0.7754575282852321, 2.288025800690999] | trajectory |
| toy_train_3 | toy train | asset:toy train [3.778299962566821, 0.6072300602940963, 1.2136282714861235] | trajectory |

## Feedback loop
| round | score | note |
|---|---|---|
| 0 | 0.19028740450181947 |  |
| 0 | None | no VLM client: feedback loop skipped |

final score: 0.19028740450181947; binding slots: `{"player_spawn": [0.0003828631689807542, 1.2, -0.0006148157231922867], "goal_volume": {"pos": [15.550822422037138, 1.0, -12.428847865906032], "extent": [1.5, 2.0, 1.5]}, "walkable": "auto", "collectibles": [], "hazards": []}`

## Playtest
```json
{
 "reached_goal": true,
 "stuck": false,
 "fell": 0,
 "t_goal": 6.9999999999999805,
 "deaths": 0,
 "collected": 0,
 "collectibles_total": 0,
 "sim_seconds": 6.9999999999999805,
 "wall_ms": 1485,
 "sim_fps": 282.8282828282828,
 "playable": true
}
```

## Errors / fallbacks

- [perception/geometry_vipe_failed] RuntimeError('vipe not installed: ModuleNotFoundError("No module named \'vipe\'")') → next backend

## Files
- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`