# Run 20260919-145307-03dc — clip `handwritten_playground`

video: `data/clips/trimmed/handwritten_playground.mp4`  commit: `a415e3c`  node: rack05-09  started: 2026-09-19T14:53:07

## Perception
- geometry backend: gt (scale metric); segmentation: gt; fallbacks: none
- frames: 32 @ 4.0 fps, duration 7.8 s; keyframes: 12; timing: {'frames': 0.5, 'geometry_gt': 7.5, 'segmentation_gt': 0.4, 'evidence': 2.0, 'total': 28.3}
- alignment: {'n_static_points': 28676, 'n_object_points': 4301, 'plane_candidates': [{'idx': 0, 'inliers': 27680, 'angle_deg': 0.1, 'cam_height': 4.5032, 'frac_obj_above': 1.0, 'angle_to_view_deg': 76.1}, {'idx': 1, 'inliers': 735, 'angle_deg': 11.1, 'cam_height': 5.2803, 'frac_obj_above': 0.348, 'angle_to_view_deg': 67.7}], 'ground': 'candidate_0', 'ground_normal_angle_to_up_deg': 0.1, 'ground_inliers': 27680, 'camera0_height_before_scale': 4.5032, 'scale_factor': 1.0}

| evidence object | class guess | dynamic | motion guess | conf | frames |
|---|---|---|---|---|---|
| wall_back | wall | False | static (static class 'wall' (structure), centroid drift 3.685 m igno) | 0.80 | 32 |
| ledge | ledge | True | trajectory (free path, linear fraction 0.91, yaw range 18) | 0.50 | 32 |
| door_frame | frame | False | static (centroid drift correlated with camera motion (r=0.96), pos r) | 0.70 | 32 |
| lava_pit | lava | False | static (centroid drift correlated with camera motion (r=0.95), pos r) | 0.70 | 32 |
| stair_1 | step | False | static (centroid drift correlated with camera motion (r=0.97), pos r) | 0.70 | 32 |
| stair_2 | step | True | trajectory (linear fraction 0.99, non-monotone) | 0.50 | 32 |
| stair_3 | step | True | trajectory (linear fraction 0.99, non-monotone) | 0.50 | 32 |
| stair_4 | step | True | trajectory (linear fraction 0.98, non-monotone) | 0.50 | 32 |
| stair_5 | step | True | trajectory (linear fraction 0.98, non-monotone) | 0.50 | 32 |
| stair_6 | step | True | trajectory (linear fraction 0.98, non-monotone) | 0.50 | 32 |
| stair_7 | step | True | trajectory (linear fraction 0.98, non-monotone) | 0.50 | 32 |
| lift | platform | True | periodic_translate (autocorr peak 1.03 at lag 17; linear fraction 0.91) | 0.90 | 32 |
| door | door | True | periodic_rotate (in-place back-and-forth yaw 63 deg) | 0.55 | 20 |
| cart | minecart | True | trajectory (linear fraction 1.00, non-monotone) | 0.50 | 32 |
| coin_1 | coin | True | spin (in-place monotone yaw change 47 deg) | 0.60 | 12 |
| coin_2 | coin | False | static (centroid drift correlated with camera motion (r=0.97), pos r) | 0.70 | 7 |
| coin_4 | coin | False | static (centroid drift correlated with camera motion (r=0.92), pos r) | 0.70 | 10 |
| wall_left | wall | False | static (static class 'wall' (structure), centroid drift 7.357 m igno) | 0.80 | 29 |

## Program
- writer: {"seconds": 63.7, "stages": {"camera_static": {"ok": true, "tries": 1}, "objects": {"ok": true, "tries": 1}, "motion": {"ok": true, "tries": 1}}, "fallbacks": [], "final_validation": {"ok": true, "n_errors": 0, "warnings": []}}
- static: 3, objects: 16

| id | class | geom | motion |
|---|---|---|---|
| ledge | ledge | primitive:box [5.345454545454546, 0.3272727272727272, 3.9272727272727272] | static |
| door_frame | frame | primitive:box [0.3272727272727272, 2.4, 0.10909090909090909] | static |
| lava_pit | lava | primitive:box [3.163636363636363, 0.10909090909090909, 3.163636363636363] | static |
| stair_1 | step | primitive:box [2.0727272727272723, 0.3272727272727272, 0.43636363636363634] | static |
| stair_2 | step | primitive:box [1.8545454545454543, 0.7636363636363636, 0.3272727272727272] | static |
| stair_3 | step | primitive:box [1.9636363636363636, 1.0909090909090908, 0.43636363636363634] | static |
| stair_4 | step | primitive:box [2.0727272727272723, 1.5272727272727271, 0.43636363636363634] | static |
| stair_5 | step | primitive:box [1.9636363636363636, 1.9636363636363636, 0.43636363636363634] | static |
| stair_6 | step | primitive:box [1.9636363636363636, 2.4, 0.43636363636363634] | static |
| stair_7 | step | primitive:box [1.9636363636363636, 2.8363636363636364, 0.43636363636363634] | static |
| lift | platform | primitive:box [3.4909090909090907, 0.3272727272727272, 1.4181818181818182] | periodic_translate |
| door | door | primitive:box [1.2, 2.4, 0.021818181818181816] | revolute |
| cart | minecart | primitive:box [1.2, 0.7636363636363636, 0.7636363636363636] | trajectory |
| coin_1 | coin | asset:coin [0.5454545454545454, 0.43636363636363634, 0.3272727272727272] | spin |
| coin_2 | coin | asset:coin [0.5454545454545454, 0.43636363636363634, 0.3272727272727272] | static |
| coin_4 | coin | asset:coin [0.5454545454545454, 0.43636363636363634, 0.3272727272727272] | static |

## Feedback loop
| round | score | note |
|---|---|---|
| 0 | 0.5469488250412372 |  |
| 1 | 0.5469488250412372 | Adjusted wall_back pose, door_frame scale, lava_pit pose, and stair_1 scale to fix trajectory and mask IoU errors. |
| 2 | 0.5469488250412372 | Adjusted pose for wall_back and scale for door_frame, lava_pit, and stair_1 based on trajectory and mask errors. |

final score: 0.5469488250412372; binding slots: `{"player_spawn": [-8.4, 1.2, 5.727272727272727], "goal_volume": {"pos": [8.4, 1.0, -5.727272727272727], "extent": [1.5, 2.0, 1.5]}, "walkable": "auto", "collectibles": ["coin_1", "coin_2", "coin_4"], "hazards": ["lava_pit"]}`

## Playtest
```json
{
 "reached_goal": true,
 "stuck": false,
 "fell": 0,
 "t_goal": 7.033333333333314,
 "deaths": 0,
 "collected": 0,
 "collectibles_total": 3,
 "sim_seconds": 7.04999999999998,
 "wall_ms": 1627,
 "sim_fps": 259.9877074370006,
 "playable": true,
 "review": {
  "related_to_video": true,
  "playable": true,
  "defect": "The level has inconsistent object placement and geometry changes between frames, indicating poor generation quality."
 }
}
```

## Errors / fallbacks

- [feedback/no_improvement] round 1: best candidate 0.544 <= 0.547 → keep previous program
- [feedback/no_improvement] round 2: best candidate 0.544 <= 0.547 → keep previous program

## Files
- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`