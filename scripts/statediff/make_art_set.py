#!/usr/bin/env python
"""造铰接探针的数据：每条一个家具（柜门、抽屉、箱盖、烤箱门），首帧全关、末帧部分打开，
渲染两端的 rgb 与 id 通道，配一句欠定的描述。

子集：train（训练家族）、holdout_in（训练家族换种子，前 N 条各带一条只换样式的配对）、
holdout_out（训练没见过的铰接构型：往下翻的烤箱门、门和抽屉混在一起的柜子）。

用法:
  python scripts/statediff/make_art_set.py --out $GWM_SCRATCH/statediff/art1 \
      --train 600 --holdout-in 60 --pairs 20 --holdout-out 60 --workers 16
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

from gwm.compiler.validate import validate_candidate                                   # noqa: E402
from gwm.config import load_config                                                      # noqa: E402
from gwm.dataset.articulated import DURATION, TEST_FAMILIES, TRAIN_FAMILIES, generate_articulated  # noqa: E402
from scripts.statediff.common import HEIGHT, WIDTH, render_endpoints                    # noqa: E402


def _git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, timeout=10).stdout.strip()


def make_one(job: tuple) -> dict:
    set_name, clip, family, seed, style_seed, tags, out_dir = job
    t0 = time.time()
    program, facts = generate_articulated(clip, family, seed, style_seed)
    report = validate_candidate(program)
    if not report["ok"]:
        return {"id": f"{set_name}/{clip}", "error": "invalid:" + ",".join(e["code"] for e in report["errors"][:3])}
    clip_dir = Path(out_dir) / set_name / clip
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "program.json").write_text(json.dumps(program, ensure_ascii=False, indent=1), encoding="utf-8")
    res = render_endpoints(program, clip_dir / "work", [0.0, DURATION], load_config(), WIDTH, HEIGHT)
    if not res["ok"]:
        return {"id": f"{set_name}/{clip}", "error": res["stage"] + ":" + str(res.get("error") or res.get("validation", {}).get("errors", [""])[:1])[:300]}
    files = {}
    for t, key in ((0.0, "0"), (round(DURATION, 3), "1")):
        for p in ("rgb", "id"):
            shutil.copyfile(res["files"][t][p], clip_dir / f"{p}{key}.png")
            files[f"{'img' if p == 'rgb' else 'id'}{key}"] = f"{set_name}/{clip}/{p}{key}.png"
    shutil.rmtree(clip_dir / "work", ignore_errors=True)
    return {"id": f"{set_name}/{clip}", "set": set_name, "clip": clip, "task": "art", "family": family,
            "tags": tags, "duration": DURATION, "program": f"{set_name}/{clip}/program.json", **files,
            "text": facts["sentence"], "yaw": facts["yaw"], "n_parts": len(program["objects"]),
            "moving": sorted(k for k, j in facts["joints"].items() if j["moves"]),
            "seed": seed, "style_seed": style_seed, "seconds": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--holdout-in", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--holdout-out", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set-suffix", default="", help="子集名加后缀，比如造无标注的适配池时用 _pool，免得和正式留出集撞名")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for set_name, families, offset, count, pairs in (("train", TRAIN_FAMILIES, 1, a.train, 0),
                                                     ("holdout_in", TRAIN_FAMILIES, 2, a.holdout_in, a.pairs),
                                                     ("holdout_out", TEST_FAMILIES, 3, a.holdout_out, 0)):
        rng = random.Random(a.seed + offset)
        set_name += a.set_suffix
        base = []
        for i in range(count):
            base.append((set_name, f"{set_name}_{i:05d}", families[i % len(families)],
                         rng.randrange(2 ** 31), rng.randrange(2 ** 31), ["base"], str(out)))
        for b in base[:pairs]:          # 只换样式：几何、关节、部件颜色都不变，背景、光照、地面、柜体颜色换掉
            base.append((set_name, b[1] + "_style", b[2], b[3], rng.randrange(2 ** 31), ["style_variant", b[1]], str(out)))
        jobs += base
    print(f"{len(jobs)} clips with {a.workers} workers", flush=True)

    t0 = time.time()
    rows, failures = [], []
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(make_one, jobs), 1):
            (failures if "error" in row else rows).append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {len(failures)} failed, {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["id"])
    (out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    per_set = {}
    for r in rows:
        per_set.setdefault(r["set"], {}).setdefault(r["family"], 0)
        per_set[r["set"]][r["family"]] += 1
    manifest = {"seed": a.seed, "duration_s": DURATION, "width": WIDTH, "height": HEIGHT, "git_commit": _git_commit(),
                "written": len(rows), "failed": failures, "per_set_family": per_set, "wall_s": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"written": len(rows), "failed": len(failures), "per_set_family": per_set}, ensure_ascii=False))
    if failures:
        print("FAILED:", json.dumps(failures[:3], ensure_ascii=False)[:800], file=sys.stderr)
    return 0 if rows and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
