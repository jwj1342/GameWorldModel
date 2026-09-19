# Run 20260919-142411-39ed — clip `handwritten_playground`

video: `data/clips/trimmed/handwritten_playground.mp4`  commit: `a415e3c`  node: rack02-13  started: 2026-09-19T14:24:11

## Perception
- geometry backend: gt (scale metric); segmentation: gt; fallbacks: none
- frames: 32 @ 4.0 fps, duration 7.8 s; keyframes: 12; timing: {'frames': 0.6, 'geometry_gt': 8.4, 'segmentation_gt': 0.4, 'evidence': 1.7, 'total': 11.8}
- alignment: {'n_static_points': 28676, 'n_object_points': 4301, 'plane_candidates': [{'idx': 0, 'inliers': 27680, 'angle_deg': 0.1, 'cam_height': 4.5032, 'frac_obj_above': 1.0}, {'idx': 1, 'inliers': 735, 'angle_deg': 11.1, 'cam_height': 5.2803, 'frac_obj_above': 0.348}], 'ground': 'candidate_0', 'ground_normal_angle_to_up_deg': 0.1, 'ground_inliers': 27680, 'camera0_height_before_scale': 4.5032, 'scale_factor': 1.0}

| evidence object | class guess | dynamic | motion guess | conf | frames |
|---|---|---|---|---|---|
| wall_back | wall | False | static (static class 'wall', pos range 3.685 m) | 0.80 | 32 |
| ledge | ledge | True | trajectory (free path, linear fraction 0.91, yaw range 18) | 0.50 | 32 |
| door_frame | frame | False | static (centroid drift correlated with camera motion (r=0.96), pos r) | 0.70 | 32 |
| lava_pit | lava | False | static (pos range 0.258 m (< 1.282), yaw range 0.0 deg) | 0.90 | 32 |
| stair_1 | step | True | trajectory (free path, linear fraction 0.80, yaw range 4) | 0.50 | 32 |
| stair_2 | step | False | static (centroid drift correlated with camera motion (r=0.99), pos r) | 0.70 | 32 |
| stair_3 | step | True | prismatic (linear fraction 0.98, monotone) | 0.70 | 32 |
| stair_4 | step | True | prismatic (linear fraction 1.00, monotone) | 0.70 | 32 |
| stair_5 | step | True | prismatic (linear fraction 1.00, monotone) | 0.70 | 32 |
| stair_6 | step | True | prismatic (linear fraction 0.99, monotone) | 0.70 | 32 |
| stair_7 | step | True | prismatic (linear fraction 1.00, monotone) | 0.70 | 32 |
| lift | platform | True | periodic_translate (autocorr peak 1.02 at lag 17; linear fraction 0.94) | 0.90 | 32 |
| door | door | True | periodic_rotate (in-place back-and-forth yaw 63 deg) | 0.55 | 20 |
| cart | minecart | True | trajectory (linear fraction 1.00, non-monotone) | 0.50 | 32 |
| coin_1 | coin | True | spin (in-place monotone yaw change 47 deg) | 0.60 | 12 |
| coin_2 | coin | False | static (centroid drift correlated with camera motion (r=0.97), pos r) | 0.70 | 7 |
| coin_4 | coin | False | static (centroid drift correlated with camera motion (r=0.92), pos r) | 0.70 | 10 |
| wall_left | wall | False | static (static class 'wall', pos range 7.357 m) | 0.80 | 29 |

## Program
- writer: {"seconds": 0.0, "mode": "direct"}
- static: 3, objects: 16

| id | class | geom | motion |
|---|---|---|---|
| ledge | ledge | asset:ledge [3.9680193969753956, 0.2834206146383358, 2.9701661374549415] | trajectory |
| door_frame | frame | asset:frame [0.2246033140964514, 1.7946191641644913, 0.08115980348267089] | static |
| lava_pit | lava | asset:lava [2.374100494527757, 0.04476990033569507, 2.334619478107501] | static |
| stair_1 | step | asset:step [1.5604790290501926, 0.2821648285589762, 0.293573784951094] | trajectory |
| stair_2 | step | asset:step [1.4137013713428372, 0.5558122072986966, 0.28337584843322994] | static |
| stair_3 | step | asset:step [1.4545527791513326, 0.8536087940836006, 0.30757635044518566] | trajectory |
| stair_4 | step | asset:step [1.5104365284161818, 1.1741186331700362, 0.3055828136601636] | trajectory |
| stair_5 | step | asset:step [1.4862071550944767, 1.472438465382045, 0.3158298125867162] | trajectory |
| stair_6 | step | asset:step [1.4499095963381705, 1.7831807720723745, 0.3049166427278825] | trajectory |
| stair_7 | step | asset:step [1.428584533683232, 2.0977769675264035, 0.3056095139223983] | trajectory |
| lift | platform | asset:platform [2.613321105835147, 0.20916919758296149, 1.0972403850612777] | periodic_translate |
| door | door | asset:door [0.9191637335606496, 1.8081155043457082, 0.040749206819734436] | periodic_rotate |
| cart | minecart | asset:minecart [0.908153361892637, 0.5841415237659945, 0.5773401114594998] | trajectory |
| coin_1 | coin | asset:coin [0.42416490225705883, 0.3101434503718625, 0.23711694291776786] | spin |
| coin_2 | coin | asset:coin [0.4018165730470543, 0.317035737082163, 0.187994109012186] | static |
| coin_4 | coin | asset:coin [0.4319704942237405, 0.2980714145866719, 0.22920786471806937] | static |

## Feedback loop
| round | score | note |
|---|---|---|
| 0 | 0.7143132760773923 |  |
| 0 | None | no VLM client: feedback loop skipped |

final score: 0.7143132760773923; binding slots: `{"player_spawn": [-7.334517222624918, 1.2, 6.39853248994004], "goal_volume": {"pos": [8.253349188898246, 1.0, -5.183560590508874], "extent": [1.5, 2.0, 1.5]}, "walkable": "auto", "collectibles": ["coin_1", "coin_2", "coin_4"], "hazards": ["lava_pit"]}`

## Playtest
```json
{
 "reached_goal": true,
 "stuck": false,
 "fell": 0,
 "t_goal": 4.733333333333322,
 "deaths": 0,
 "collected": 0,
 "collectibles_total": 3,
 "sim_seconds": 4.7499999999999885,
 "wall_ms": 1149,
 "sim_fps": 248.04177545691905,
 "playable": true
}
```

## Errors / fallbacks


## Files
- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`