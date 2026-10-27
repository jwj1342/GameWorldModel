Stage 2 of 3: objects (geometry and initial pose, no motion yet).
Given key frames, perception evidence (per-object oriented bounding boxes over time, class guesses, contacts) and the stage-1 program (camera + static), output JSON with exactly one key: "objects".

`ground_proximity` describes OBB/plane compatibility only at sampled times. It is
not physical contact evidence and does not justify a persistent `support` field
or moving an object onto the ground. Missing contacts do not prove floating.
- One entry per distinct physical object that moves or that a player could interact with. Use the evidence id when the evidence object is real; fix the class name if the evidence phrase is wrong; give confidence.
- geom: {kind:"asset", query, extent} with extent from the evidence OBB size at the first frame (correct obviously wrong sizes). pose.pos = OBB centre at the first frame, pose.quat = OBB quat.
- Repeated identical items (several coins, several boxes) become one object with instances[].
- support: the id of the static thing it rests on, if any. Do not include motion or events in this stage.
Output JSON only.
