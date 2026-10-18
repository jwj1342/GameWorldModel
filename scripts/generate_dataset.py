#!/usr/bin/env python
"""造一批带真值程序的合成场景。只产出程序和清单，不渲染——渲染走 harness/record.mjs。

用法:
  python -m scripts.generate_dataset --split test --count 50 --out $GWM_SCRATCH/dataset/probe50
  python -m scripts.generate_dataset --split train --count 500 --pairs 60 --out ...

每条写一个目录：<out>/<clip>/program.json。清单 <out>/manifest.json 记下每条的家族、
相机档、三个种子、运动类型分布、材质写法分布，以及类名命中绑定词表的个数。
配对（同行为不同样式 / 同样式不同行为）写进 <out>/pairs.json。
"""
from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.compiler.validate import validate_candidate                    # noqa: E402
from gwm.dataset.behavior import TEST_FAMILIES, TRAIN_FAMILIES          # noqa: E402
from gwm.dataset.generator import SceneSpec, describe, generate_scene   # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        head = out.stdout.strip() if out.returncode == 0 else ""
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}" if head else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_specs(split: str, count: int, seed: int, duration: float, fps: int,
                pairs: int, stage_scale: float = 1.0) -> list[SceneSpec]:
    families = TRAIN_FAMILIES if split == "train" else TEST_FAMILIES
    profile = "train" if split == "train" else "test"
    rng = random.Random(seed)
    specs: list[SceneSpec] = []
    for i in range(count):
        specs.append(SceneSpec(clip=f"{split}_{i:05d}", family=families[i % len(families)],
                               behavior_seed=rng.randrange(2 ** 31), style_seed=rng.randrange(2 ** 31),
                               camera_seed=rng.randrange(2 ** 31), profile=profile,
                               duration=duration, fps=fps, tags=("base",),
                               stage_scale=stage_scale))
    # 配对从前 `pairs` 条派生：一条只换样式，一条只换行为。
    for i in range(min(pairs, len(specs))):
        base = specs[i]
        specs.append(base.with_style(rng.randrange(2 ** 31), f"{base.clip}_style"))
        specs.append(base.with_behavior(rng.randrange(2 ** 31), f"{base.clip}_behavior"))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--count", type=int, required=True, help="基础场景条数（配对在此之外另算）")
    ap.add_argument("--pairs", type=int, default=0, help="从前 N 条派生配对，每条派生两条")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stage-scale", type=float, default=1.0,
                    help="整体相似缩放。1.0 是现在的关卡尺度（走廊 4-7 x 18-30 米，相机"
                         "高度中位数 7.4 米）；链路把重建标定到米靠的是「相机离地 1.2 米」"
                         "这个假设，两者差六到十二倍，见 #68。0.15 左右把场景缩到房间/"
                         "桌面尺度，和目标输入（手持拍卧室、桌子）对上。缩放是相似变换，"
                         "场景内部的相对结构不变。")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = build_specs(a.split, a.count, a.seed, a.duration, a.fps, a.pairs, a.stage_scale)

    rows, failures = [], []
    for spec in specs:
        program, facts = generate_scene(spec)
        report = validate_candidate(program)
        if not report["ok"]:
            failures.append({"clip": spec.clip, "errors": report["errors"][:5]})
            continue
        clip_dir = out / spec.clip
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "program.json").write_text(
            json.dumps(program, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
        row = describe(program, spec, facts)
        row["warnings"] = [w["code"] for w in report["warnings"]]
        rows.append(row)

    pairs = [{"kind": "style_variant", "base": r["clip"][:-len("_style")], "variant": r["clip"]}
             for r in rows if r["clip"].endswith("_style")]
    pairs += [{"kind": "behavior_variant", "base": r["clip"][:-len("_behavior")], "variant": r["clip"]}
              for r in rows if r["clip"].endswith("_behavior")]

    # 三个种子加这个 commit 就足以把整批数据原样重造出来，所以产物放在会被清理的
    # $SCRATCH 上也不会真的丢。少了 commit 就不行——生成器一改，同样的种子是另一批场景。
    (out / "manifest.json").write_text(json.dumps({
        "split": a.split, "seed": a.seed, "duration_s": a.duration, "fps": a.fps,
        "git_commit": _git_commit(), "generator": "gwm.dataset.generator",
        "requested": len(specs), "written": len(rows), "failed": len(failures),
        "families": list(TRAIN_FAMILIES if a.split == "train" else TEST_FAMILIES),
        "clips": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")

    motion_counts: dict[str, int] = {}
    for row in rows:
        for m in row["motion_types"]:
            motion_counts[m] = motion_counts.get(m, 0) + 1
    print(f"wrote {len(rows)}/{len(specs)} programs to {out}")
    print("motion types:", json.dumps(motion_counts, sort_keys=True))
    print("binding-vocab class hits:", sum(r["binding_vocab_hits"] for r in rows),
          "of", sum(r["object_count"] for r in rows), "objects")
    if failures:
        print("FAILED:", json.dumps(failures[:3], ensure_ascii=False)[:1200], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
