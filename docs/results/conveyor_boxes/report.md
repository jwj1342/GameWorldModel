# Run 20260919-143631-ecfa — clip `conveyor_boxes`

video: `data/clips/trimmed/conveyor_boxes.mp4`  commit: `a415e3c`  node: rack06-05  started: 2026-09-19T14:36:31

## Perception
- geometry backend: vggt (scale relative); segmentation: grounded_sam2; fallbacks: ['geometry:vipe']
- frames: 64 @ 4.0 fps, duration 15.8 s; keyframes: 8; timing: {'frames': 0.6, 'geometry_vggt': 88.3, 'segmentation_grounded_sam2': 107.3, 'evidence': 2.4, 'total': 199.6}
- alignment: {'n_static_points': 38525, 'n_object_points': 2020, 'ground_mode': 'facing_camera', 'plane_candidates': [{'idx': 0, 'inliers': 19035, 'angle_deg': 83.9, 'cam_height': 0.9314, 'frac_obj_above': 0.805, 'angle_to_view_deg': 7.2}, {'idx': 1, 'inliers': 8211, 'angle_deg': 83.6, 'cam_height': 0.9429, 'frac_obj_above': 0.362, 'angle_to_view_deg': 7.6}, {'idx': 2, 'inliers': 6134, 'angle_deg': 83.6, 'cam_height': 0.9183, 'frac_obj_above': 0.957, 'angle_to_view_deg': 7.4}, {'idx': 3, 'inliers': 2553, 'angle_deg': 78.9, 'cam_height': -0.9171, 'frac_obj_above': 0.141, 'angle_to_view_deg': 13.4}], 'ground': 'candidate_0', 'ground_normal_angle_to_up_deg': 83.9, 'ground_inliers': 19035, 'camera0_height_before_scale': 0.9314, 'scale_factor': 1.2884}

| evidence object | class guess | dynamic | motion guess | conf | frames |
|---|---|---|---|---|---|
| conveyor_belt_1 | conveyor belt | False | static (static class 'conveyor belt' (structure), centroid drift 0.0) | 0.80 | 24 |
| metal_arm_2 | metal arm | False | static (pos range 0.010 m (< 0.160), yaw range 2.7 deg) | 0.90 | 24 |
| cardboard_box_3 | cardboard box | False | static (pos range 0.007 m (< 0.050), yaw range 1.4 deg) | 0.90 | 24 |

## Program
- writer: {"seconds": 0.0, "mode": "direct"}
- static: 2, objects: 2

| id | class | geom | motion |
|---|---|---|---|
| metal_arm_2 | metal arm | asset:metal arm [5.039424351449492, 0.4285714285714286, 0.9651322887523401] | static |
| cardboard_box_3 | cardboard box | asset:cardboard box [1.1150427030316346, 0.4285714285714286, 0.4406728816532303] | static |

## Feedback loop
| round | score | note |
|---|---|---|
| 0 | 0.49431059512639863 |  |
| 0 | None | no VLM client: feedback loop skipped |

final score: 0.49431059512639863; binding slots: `{"player_spawn": [0.0005441422845021545, 1.2, 0.0002785960224184559], "goal_volume": {"pos": [-9.693440951869503, 1.0, 9.094202547498478], "extent": [1.5, 2.0, 1.5]}, "walkable": "auto", "collectibles": [], "hazards": []}`

## Playtest
```json
{
 "reached_goal": true,
 "stuck": false,
 "fell": 0,
 "t_goal": 3.183333333333327,
 "deaths": 0,
 "collected": 0,
 "collectibles_total": 0,
 "sim_seconds": 3.1999999999999935,
 "wall_ms": 1051,
 "sim_fps": 182.68315889628926,
 "playable": true
}
```

## Errors / fallbacks

- [perception/geometry_vipe_failed] RuntimeError('vipe not installed: ModuleNotFoundError("No module named \'vipe\'")') → next backend

## Files
- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`