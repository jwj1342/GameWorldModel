#!/usr/bin/env python
"""一段视频 -> 一个带真网格的可交互场景。

这个脚本把四步串起来，中间不调生成模型，所以跑出来的差别只来自几何和资产这两层：

    视频 -> 感知（证据、掩码、帧）
         -> 直译成场景程序（每个物体一个 asset 节点）
         -> 看画面挑资产，写进 geom.ref
         -> 编译成可运行的 game 目录

挑资产这一步是关键：只看类名的话，"toy train" 会命中 train 这个 kit 里的车厢连接件，
因为名字里都有 train；把视频里那个物体抠出来和资产缩略图比一比才分得出哪个像。

用法:
  python -m scripts.rebuild_with_assets --video data/clips/trimmed/plarail_train.mp4 \
      --clip plarail_train --out $GWM_SCRATCH/rebuild/plarail \
      --phrases "toy train,railroad track,wooden block,table,bookshelf,book,box,cup,chair,bed,lamp,plant"
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.assets import choose_assets                            # noqa: E402
from gwm.assets.embed import AssetIndex, Embedder               # noqa: E402
from gwm.assets.library import library_root, load_catalog       # noqa: E402
from gwm.compiler.compile import compile_program                # noqa: E402
from gwm.compiler.repair import repair_candidate                # noqa: E402
from gwm.synthesis.contact import settle_on_ground              # noqa: E402
from gwm.config import load_config, site_name                   # noqa: E402
from gwm.perception.run import load_masks, run_perception       # noqa: E402
from gwm.synthesis.direct import evidence_to_program            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrases", default="", help="逗号分隔的检测短语；留空走默认词表")
    ap.add_argument("--max-objects", type=int, default=0, help=">0 时覆盖配置里的物体数上限")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--no-image-retrieval", action="store_true",
                    help="只用文字检索，用来和带图像检索的结果做对照")
    a = ap.parse_args()

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *a.config]
    cfg = load_config([f for f in files if Path(f).is_file()])
    if a.max_objects > 0:
        cfg["perception"]["max_objects"] = a.max_objects
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    phrases = [p.strip() for p in a.phrases.split(",") if p.strip()] or None

    t0 = time.time()
    evidence = run_perception(a.video, a.clip, out / "perception", cfg, phrases=phrases)
    timings = {"perception_s": round(time.time() - t0, 1)}
    print(f"感知: {len(evidence['objects'])} 个物体, {timings['perception_s']}s", flush=True)

    program = evidence_to_program(evidence, a.clip, cfg)
    asset_nodes = sum(1 for o in program.get("objects", []) if o.get("geom", {}).get("kind") == "asset")
    print(f"直译: static {len(program.get('static', []))}, objects {len(program.get('objects', []))}"
          f"（其中 {asset_nodes} 个要挑资产）", flush=True)

    root = library_root(cfg)
    decisions = []
    if root and asset_nodes:
        t1 = time.time()
        catalog = load_catalog(root)
        index = embedder = None
        index_file = Path(root) / "dinov2_index.npz"
        if not a.no_image_retrieval and index_file.is_file():
            index = AssetIndex.load(index_file)
            embedder = Embedder(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])
        frames = json.loads((out / "perception" / "frames" / "frames.json").read_text())["frames"]
        masks = load_masks(out / "perception")
        settings = cfg.get("assets") or {}
        decisions = choose_assets(program, evidence, frames, masks, catalog, index, embedder,
                                  image_weight=float(settings.get("image_weight", 0.6)),
                                  min_score=float(settings.get("min_score", 0.45)))
        timings["retrieval_s"] = round(time.time() - t1, 1)
        for d in decisions:
            mark = "图像+文字" if d["image_used"] else "只有文字"
            print(f"  {d['id']:<18} {d['query']:<18} -> {d.get('chosen') or '（退回基本体）':<28} "
                  f"{mark} 候选={[c['asset_id'] for c in d['candidates'][:3]]}", flush=True)

    # 直译出来的位姿直接取自包围盒中心，地面估歪一点就会有物体陷进地里。
    # 模型那条路本来就会过一遍保守修复，直译这条也该过——它只在证据支持时才动位姿。
    moves = settle_on_ground(program)
    if moves:
        applied = [m for m in moves if m["applied"]]
        print(f"接触装配: 抬起 {len(applied)} 个物体，最大 {max((m['lift_m'] for m in applied), default=0):.3f} m"
              f"，放弃 {len(moves) - len(applied)} 个", flush=True)
    program, repair_report, unresolved = repair_candidate(program, evidence, None, cfg)
    timings["repairs"] = len(repair_report.get("repairs", []))
    timings["settled"] = sum(1 for m in moves if m["applied"])
    if unresolved:
        print(f"修复后仍有 {len(unresolved)} 条未解决: "
              f"{sorted({u['code'] for u in unresolved})}", flush=True)

    (out / "program.json").write_text(json.dumps(program, indent=1, ensure_ascii=False), encoding="utf-8")
    result = compile_program(program, out / "game", cfg)
    if not result["ok"]:
        print("编译失败:", json.dumps(result["validation"], ensure_ascii=False)[:400])
        return 2
    resolved = sum(1 for r in result["assets"]["requests"] if r["resolved"].startswith("library"))
    summary = {"clip": a.clip, "objects": len(program.get("objects", [])),
               "static": len(program.get("static", [])), "asset_nodes": asset_nodes,
               "assets_resolved": resolved, "image_retrieval": bool(index) if root else False,
               "timings": timings, "decisions": decisions, "game_dir": result["game_dir"]}
    (out / "rebuild.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "decisions"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
