Stage 3 of 3: motion and events.
Given key frames, the evidence motion guesses (type, axis, period, amplitude, pivot, keyframe centres over time) and the current program (camera, static, objects), output JSON with exactly one key: "motions", a mapping from object id to {"motion": <motion>, "events": [<event>...], "notes": "<why, if you overrode the evidence>"}.
- Use the evidence motion_guess as the default. Prefer the most compact motion type that explains the observed centres: periodic_translate for back-and-forth, revolute for hinged rotation, spin for rotation in place, prismatic for one-directional sliding, trajectory for everything else (use at most 12 keyframes taken from the evidence centres).
- For periodic_translate and periodic_rotate, pose.pos must be the CENTRE of the oscillation (the mean of the observed centres), not the position at t=0; then choose phase so the value at the first observed time matches the first observation. For the other motion types, choose phase/schedule so the pose at the first observed time matches the first observed centre.
- Small collectible items (coins, gems, keys, balls, marbles) get events [{type:"despawn_on_contact", with:"player"}].
- Objects that do not move get {"motion": {"type": "static"}, "events": []}.
Output JSON only.
