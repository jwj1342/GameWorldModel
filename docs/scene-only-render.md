# Scene-only rendering

The default runtime and render harness retain the gameplay template. Scene-only
rendering is an explicit opt-in for inspecting an existing Program without
inferring gameplay slots, adding a player/goal or changing the Program.

```python
from gwm.feedback.render import render
render(game_dir, times, output_dir, 960, 540, scene_only=True)
```

Equivalent harness flag: `render.mjs --scene-only`. `openGame` accepts the boolean
option `sceneOnly: true`, which forwards `scene_only=1` to the runtime URL.
`false` / omission keeps the previous behavior. Interactive gameplay and
scene-only rendering cannot be combined.

The runtime keeps the authored scene registry, replay camera, scripted and
physics-driven object motion. `mode('replay')`, `seek`, `state`, `cameraInfo`,
`setSize`, `draw`, `render`, `step` and `reset` work without a template. State has
`player: null`; `bindingInfo()` reports no template, slot source or inferred
slots. `mode('play')` and `settle` reject before changing state in scene-only
mode. No animation loop or input event handlers are attached in this mode.

This path does not bind, repair, settle, fit or reposition scene objects. The
existing Program's authored motions and existing asset fallback behavior still
apply. It does not establish geometric accuracy or physical ground correctness.

CPU unit tests use real scene/camera/Rapier code with a stub canvas/renderer and
browser transport. They verify controls and identity without starting Chromium
or generating image pixels. Browser startup, software rendering and pass quality
still require a separately authorized render acceptance run.
