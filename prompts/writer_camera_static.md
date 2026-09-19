Stage 1 of 3: camera and static structure.
Given key frames of a video and perception evidence (camera intrinsics, camera poses, ground plane estimate), output JSON with exactly these keys: "style", "camera", "static".
- camera: copy the intrinsics fov_deg and aspect from the evidence; use the evidence camera poses as keyframes (you may subsample to at most 16 keyframes, keep t increasing).
- static: the ground as one box slightly larger than the ground extent hint (height 0.3, top face at y=0), plus walls, tables, tracks, belts, ledges, steps or other fixed structure you can see. Do not include the movable objects (they come in stage 2; the reserved ids are listed in the message). Give each a class and a material.
- style.background: a hex colour matching the scene mood.
Output JSON only.
