#!/usr/bin/env python
"""在合成片段上跑一遍真实感知管线，逐条计时。

#5 有一条硬要求：合成数据也必须跑真实感知，不能把真值当证据喂进去，否则训练时用的是
干净证据、推理时用的是带噪证据，模型在真实视频上直接崩。这条要求让每条数据的成本
翻几倍，而这个倍数就是 #21 门二要的那个数。这个脚本只负责把它测出来。

用法:
  python -m scripts.probe_perception_cost --rendered $GWM_SCRATCH/dataset/probe50_rendered \
         --out $GWM_SCRATCH/dataset/probe50_perception --limit 5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.config import load_config, site_name                 # noqa: E402
from gwm.perception.run import run_perception                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered", required=True, help="render_dataset 的输出目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--config", action="append", default=[], help="额外配置，例如 configs/perception/cpu.yaml")
    a = ap.parse_args()

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *a.config]
    cfg = load_config([f for f in files if Path(f).is_file()])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    clips = sorted(p for p in Path(a.rendered).iterdir() if (p / "clip.mp4").is_file())[:a.limit]
    rows = []
    for i, clip_dir in enumerate(clips, 1):
        row = {"clip": clip_dir.name, "ok": False}
        t0 = time.time()
        try:
            evidence = run_perception(str(clip_dir / "clip.mp4"), clip_dir.name,
                                      out / clip_dir.name, cfg)
            row.update(ok=True, objects=len(evidence["objects"]),
                       geometry_backend=evidence["meta"]["geometry_backend"],
                       segmentation_backend=evidence["meta"]["segmentation_backend"],
                       fallbacks=evidence["meta"]["fallbacks"],
                       motion_unknown=sum(1 for o in evidence["objects"]
                                          if (o.get("motion_guess") or {}).get("type") == "unknown"))
        except Exception as error:                       # 一条挂了不该拖垮整轮计时
            row["error"] = f"{type(error).__name__}: {error}"[:400]
            traceback.print_exc()
        row["perception_s"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"[{i}/{len(clips)}] {row['clip']} {row['perception_s']}s "
              f"{'ok' if row['ok'] else row.get('error', '')[:70]}", flush=True)

    done = [r for r in rows if r["ok"]]
    summary = {"rendered": a.rendered, "clips": len(rows), "ok": len(done),
               "per_clip_perception_s": round(sum(r["perception_s"] for r in done) / len(done), 1) if done else None,
               "rows": rows}
    (out / "perception_cost.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
