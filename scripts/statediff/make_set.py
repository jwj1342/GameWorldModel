#!/usr/bin/env python
"""造状态差分预实验的数据：每条一份真值程序、首末两帧（rgb 与 id 通道）、一句欠定的描述。

三个子集：train（训练）、holdout_in（同分布留出，可带「只换样式」的配对）、holdout_out（test 家族，
生成器没在训练里见过的场景家族）。只渲染两端的帧，不录整段视频，所以一条几秒钟。

用法:
  python scripts/statediff/make_set.py --out $GWM_SCRATCH/statediff/set1 \
      --train 300 --holdout-in 50 --pairs 15 --holdout-out 40 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.compiler.validate import validate_candidate                    # noqa: E402
from gwm.config import load_config                                       # noqa: E402
from gwm.dataset.behavior import TEST_FAMILIES, TRAIN_FAMILIES          # noqa: E402
from gwm.dataset.generator import SceneSpec, generate_scene             # noqa: E402
from scripts.statediff.common import HEIGHT, WIDTH, describe_scene, render_endpoints  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def specs_for(set_name: str, split: str, seed: int, count: int, pairs: int,
              duration: float, fps: int) -> list[SceneSpec]:
    """和 scripts/generate_dataset.build_specs 同一套抽法，只是片段名带上子集名，三个子集不撞名。"""
    families = TRAIN_FAMILIES if split == "train" else TEST_FAMILIES
    profile = "train" if split == "train" else "test"
    rng = random.Random(seed)
    specs = [SceneSpec(clip=f"{set_name}_{i:05d}", family=families[i % len(families)],
                       behavior_seed=rng.randrange(2 ** 31), style_seed=rng.randrange(2 ** 31),
                       camera_seed=rng.randrange(2 ** 31), profile=profile,
                       duration=duration, fps=fps, tags=("base",)) for i in range(count)]
    for i in range(min(pairs, len(specs))):      # 只换样式的配对：行为半边应该一字不差
        specs.append(specs[i].with_style(rng.randrange(2 ** 31), f"{specs[i].clip}_style"))
    return specs


def make_one(job: tuple) -> dict:
    set_name, spec, out_dir, width, height = job
    t0 = time.time()
    program, _facts = generate_scene(spec)
    report = validate_candidate(program)
    if not report["ok"]:
        return {"id": f"{set_name}/{spec.clip}", "error": "invalid:" + ",".join(e["code"] for e in report["errors"][:3])}
    clip_dir = Path(out_dir) / set_name / spec.clip
    clip_dir.mkdir(parents=True, exist_ok=True)
    program_path = clip_dir / "program.json"
    program_path.write_text(json.dumps(program, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    res = render_endpoints(program, clip_dir / "work", [0.0, spec.duration], load_config(), width, height)
    if not res["ok"]:
        return {"id": f"{set_name}/{spec.clip}", "error": res["stage"] + ":" + str(res.get("error") or res.get("validation", {}).get("errors", [""])[:1])[:300]}
    files = {}
    for t, key in ((0.0, "0"), (round(spec.duration, 3), "1")):
        for p in ("rgb", "id"):
            dst = clip_dir / f"{p}{key}.png"
            shutil.copyfile(res["files"][t][p], dst)
            files[f"{'img' if p == 'rgb' else 'id'}{key}"] = str(dst)
    shutil.rmtree(clip_dir / "work", ignore_errors=True)       # 编译产物和中间渲染不留，小文件太多
    return {"id": f"{set_name}/{spec.clip}", "set": set_name, "clip": spec.clip, "family": spec.family,
            "tags": list(spec.tags), "duration": spec.duration, "program": str(program_path), **files,
            "text": describe_scene(program), "n_objects": len(program["objects"]),
            "motion_types": sorted({(o.get("motion") or {}).get("type", "static") for o in program["objects"]}),
            "time_sampling": (res.get("time_sampling") or {}).get("mode"), "seconds": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--holdout-in", type=int, default=50)
    ap.add_argument("--pairs", type=int, default=15, help="holdout_in 里前 N 条各派生一条只换样式的配对")
    ap.add_argument("--holdout-out", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [("train", "train", a.seed + 1, a.train, 0),
            ("holdout_in", "train", a.seed + 2, a.holdout_in, a.pairs),
            ("holdout_out", "test", a.seed + 3, a.holdout_out, 0)]
    jobs = []
    for set_name, split, seed, count, pairs in plan:
        for spec in specs_for(set_name, split, seed, count, pairs, a.duration, a.fps):
            jobs.append((set_name, spec, str(out), a.width, a.height))
    print(f"{len(jobs)} clips to build with {a.workers} workers", flush=True)

    t0 = time.time()
    rows, failures = [], []
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(make_one, jobs), 1):
            (failures if "error" in row else rows).append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {len(failures)} failed, {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["id"])
    (out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {s: sum(r["set"] == s for r in rows) for s, *_ in plan}
    motion = {}
    for r in rows:
        for m in r["motion_types"]:
            motion[m] = motion.get(m, 0) + 1
    manifest = {"seed": a.seed, "duration_s": a.duration, "fps": a.fps, "width": a.width, "height": a.height,
                "git_commit": _git_commit(), "plan": plan, "written": len(rows), "failed": failures,
                "per_set": counts, "clips_with_motion_type": motion,
                "mean_seconds_per_clip": round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 2),
                "wall_s": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} samples to {out / 'samples.jsonl'}; per set {counts}; failed {len(failures)}")
    print("clips containing each motion type:", json.dumps(motion, sort_keys=True))
    if failures:
        print("FAILED:", json.dumps(failures[:3], ensure_ascii=False)[:800], file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
