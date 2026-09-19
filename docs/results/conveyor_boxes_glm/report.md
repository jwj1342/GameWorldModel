# Run 20260919-145646-5479 — clip `conveyor_boxes`

video: `data/clips/trimmed/conveyor_boxes.mp4`  commit: `a415e3c`  node: rack12-06  started: 2026-09-19T14:56:46

## Perception
- geometry backend: vggt (scale relative); segmentation: grounded_sam2; fallbacks: ['geometry:vipe']
- frames: 64 @ 4.0 fps, duration 15.8 s; keyframes: 8; timing: {'frames': 0.7, 'geometry_vggt': 69.3, 'segmentation_grounded_sam2': 94.0, 'evidence': 1.1, 'total': 166.0}
- alignment: {'n_static_points': 38525, 'n_object_points': 2020, 'ground_mode': 'facing_camera', 'plane_candidates': [{'idx': 0, 'inliers': 19035, 'angle_deg': 83.9, 'cam_height': 0.9314, 'frac_obj_above': 0.805, 'angle_to_view_deg': 7.2}, {'idx': 1, 'inliers': 8211, 'angle_deg': 83.6, 'cam_height': 0.9429, 'frac_obj_above': 0.362, 'angle_to_view_deg': 7.6}, {'idx': 2, 'inliers': 6134, 'angle_deg': 83.6, 'cam_height': 0.9183, 'frac_obj_above': 0.957, 'angle_to_view_deg': 7.4}, {'idx': 3, 'inliers': 2553, 'angle_deg': 78.9, 'cam_height': -0.9171, 'frac_obj_above': 0.141, 'angle_to_view_deg': 13.4}], 'ground': 'candidate_0', 'ground_normal_angle_to_up_deg': 83.9, 'ground_inliers': 19035, 'camera0_height_before_scale': 0.9314, 'scale_factor': 1.2884}

| evidence object | class guess | dynamic | motion guess | conf | frames |
|---|---|---|---|---|---|
| conveyor_belt_1 | conveyor belt | False | static (static class 'conveyor belt' (structure), centroid drift 0.0) | 0.80 | 24 |
| metal_arm_2 | metal arm | False | static (pos range 0.010 m (< 0.160), yaw range 2.7 deg) | 0.90 | 24 |
| cardboard_box_3 | cardboard box | False | static (pos range 0.007 m (< 0.050), yaw range 1.4 deg) | 0.90 | 24 |

## Program
- writer: {"seconds": 27.5, "stages": {"camera_static": {"ok": true, "tries": 1}, "objects": {"ok": true, "tries": 1}, "motion": {"ok": true, "tries": 1}}, "fallbacks": [], "final_validation": {"ok": true, "n_errors": 0, "warnings": []}}
- static: 1, objects: 3

| id | class | geom | motion |
|---|---|---|---|
| conveyor_belt_1 | conveyor belt | primitive:box [18.912, 1.024, 1.312] | static |
| metal_arm_2 | metal arm | primitive:box [9.408, 0.752, 1.808] | static |
| cardboard_box_3 | cardboard box | primitive:box [2.08, 0.32, 0.816] | static |

## Feedback loop
| round | score | note |
|---|---|---|
| 0 | 0.5014534218365017 |  |
| 1 | 0.5014534218365017 | Two objects have issues: metal_arm_2 needs pose correction, cardboard_box_3 needs scale adjustment. |
| 2 | 0.5014534218365017 | Two objects have issues: metal_arm_2 has a pose error (corrected with its evidence centre) and cardboard_box_3 has a scale error (corrected with its evidence size). |

final score: 0.5014534218365017; binding slots: `{"player_spawn": [0.0010157322644040218, 1.2, 0.0005200459085144511], "goal_volume": {"pos": [-10.816, 1.0, 8.015999999999998], "extent": [1.5, 2.0, 1.5]}, "walkable": "auto", "collectibles": [], "hazards": []}`

## Playtest
```json
{
 "reached_goal": true,
 "stuck": false,
 "fell": 0,
 "t_goal": 4.716666666666655,
 "deaths": 0,
 "collected": 0,
 "collectibles_total": 0,
 "sim_seconds": 4.733333333333322,
 "wall_ms": 1320,
 "sim_fps": 215.15151515151513,
 "playable": true,
 "review": {
  "related_to_video": false,
  "playable": true,
  "defect": "The generated 3D platformer has minimal visual connection to the source industrial machinery video, featuring a simple blue capsule and basic geometric platforms."
 }
}
```

## Errors / fallbacks

- [perception/geometry_vipe_failed] RuntimeError('vipe not installed: ModuleNotFoundError("No module named \'vipe\'")') → next backend
- [feedback/no_improvement] round 1: best candidate 0.501 <= 0.501 → keep previous program
- [feedback/no_improvement] round 2: best candidate 0.501 <= 0.501 → keep previous program

## Files
- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`