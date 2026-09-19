# Run 20260919-143633-9ac6 — clip `handwritten_playground`

video: `data/clips/trimmed/handwritten_playground.mp4`  commit: `a415e3c`  node: rack07-12  started: 2026-09-19T14:36:33

## Perception
- geometry backend: vggt (scale relative); segmentation: grounded_sam2; fallbacks: ['geometry:vipe']
- frames: 32 @ 4.0 fps, duration 7.8 s; keyframes: 8; timing: {'frames': 0.5, 'geometry_vggt': 72.0, 'segmentation_grounded_sam2': 122.3, 'evidence': 1.0, 'total': 196.6}
- alignment: {'n_static_points': 32546, 'n_object_points': 11070, 'plane_candidates': [{'idx': 0, 'inliers': 3387, 'angle_deg': 61.6, 'cam_height': -1.0591, 'frac_obj_above': 0.0, 'angle_to_view_deg': 53.2}, {'idx': 1, 'inliers': 2548, 'angle_deg': 26.2, 'cam_height': 0.3558, 'frac_obj_above': 0.543, 'angle_to_view_deg': 67.3}, {'idx': 2, 'inliers': 2396, 'angle_deg': 20.1, 'cam_height': -0.6812, 'frac_obj_above': 0.0, 'angle_to_view_deg': 77.1}, {'idx': 3, 'inliers': 1989, 'angle_deg': 85.4, 'cam_height': 1.2434, 'frac_obj_above': 1.0, 'angle_to_view_deg': 35.0}], 'ground': 'candidate_1', 'ground_normal_angle_to_up_deg': 26.2, 'ground_inliers': 2548, 'camera0_height_before_scale': 0.3558, 'scale_factor': 3.3729}

| evidence object | class guess | dynamic | motion guess | conf | frames |
|---|---|---|---|---|---|
| lawn_1 | lawn | False | static (static class 'lawn' (structure), centroid drift 0.623 m igno) | 0.80 | 24 |
| gold_coin_2 | gold coin | True | trajectory (free path, linear fraction 0.90, yaw range 65) | 0.50 | 17 |
| gold_coin_3 | gold coin | False | static (centroid drift correlated with camera motion (r=0.99), pos r) | 0.70 | 16 |
| gold_coin_4 | gold coin | False | static (centroid drift correlated with camera motion (r=0.98), pos r) | 0.70 | 17 |
| door_5 | door | True | trajectory (free path, linear fraction 0.86, yaw range 96) | 0.50 | 24 |
| platform_6 | platform | False | static (pos range 0.240 m (< 0.285), yaw range 0.0 deg) | 0.90 | 24 |

## Program
- writer: {"seconds": 0.0, "mode": "direct"}
- static: 2, objects: 5

| id | class | geom | motion |
|---|---|---|---|
| gold_coin_2 | gold coin | asset:gold coin [2.0785224139094316, 0.9097732053359683, 0.32782992035438835] | trajectory |
| gold_coin_3 | gold coin | asset:gold coin [1.4206948163421764, 0.6894971445888604, 0.3015856823312606] | static |
| gold_coin_4 | gold coin | asset:gold coin [1.7265289556571797, 0.7076629354432497, 0.30315672781212616] | static |
| door_5 | door | asset:door [0.8103235591120677, 1.0748385271576737, 0.6762945614090823] | trajectory |
| platform_6 | platform | asset:platform [2.127171528848761, 0.4670304155292658, 1.7634430058837902] | static |

## Feedback loop
| round | score | note |
|---|---|---|
| 0 | 0.2334070773296775 |  |
| 0 | None | no VLM client: feedback loop skipped |

final score: 0.2334070773296775; binding slots: `{"player_spawn": [-0.001584671766786698, 1.2, -3.3096620592084496], "goal_volume": {"pos": [-10.150014487415689, 1.0, -15.757370454294616], "extent": [1.5, 2.0, 1.5]}, "walkable": "auto", "collectibles": ["gold_coin_2", "gold_coin_3", "gold_coin_4"], "hazards": []}`

## Playtest
```json
{
 "reached_goal": true,
 "stuck": false,
 "fell": 0,
 "t_goal": 3.866666666666658,
 "deaths": 0,
 "collected": 0,
 "collectibles_total": 3,
 "sim_seconds": 3.8833333333333244,
 "wall_ms": 1062,
 "sim_fps": 219.39736346516005,
 "playable": true
}
```

## Errors / fallbacks

- [perception/geometry_vipe_failed] RuntimeError('vipe not installed: ModuleNotFoundError("No module named \'vipe\'")') → next backend

## Files
- `program.json`, `game/index.html` (serve `game/` with any static server), `perception/overlay.mp4`, `feedback/` (rgb/depth/id renders), `playtest/frames/`