#!/usr/bin/env python
"""颜色占了多少保真度：把重建的材质颜色换成真值颜色，重渲染相似度能涨多少。

#55 提议用渲染反馈闭环把材质颜色拟合到和视频一致，理由是实测量回来的亮度只有写进去
的 0.41 倍。那条 issue 值不值得做，取决于**颜色在保真度里占多大比重**——如果换成
完美的颜色，重渲染相似度也只涨一点点，那闭环就不该先做。

做法是给一个上界：拿评测已经配好的对，把重建程序里每个物体的材质颜色直接换成真值
程序里的那个，其余一个字段不动，重渲染、算相似度。这不是修法（真实视频里没有真值
颜色），是**天花板**。

用法:
  python -m scripts.probe_colour_headroom --report $GWM_SCRATCH/dataset/ab015_report \
      --gt-root $GWM_SCRATCH/dataset/ab015 --eval $GWM_SCRATCH/dataset/ab015_eval_tol.json \
      --work $SLURM_TMPDIR/ch --limit 4
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _current_colour(node: dict) -> str:
    material = node.get("material")
    if isinstance(material, dict):
        base = material.get("base_color")
        if isinstance(base, str) and base.startswith("#"):
            return base
    if isinstance(material, str) and material.startswith("#"):
        return material
    return "#9a9a9a"


def _opposite(colour: str) -> str:
    """色相旋转半圈，亮度大致保持。只改颜色，不改明暗，免得把亮度和色彩混在一起。"""
    import colorsys
    value = colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb((h + 0.5) % 1.0, l, max(s, 0.6))
    return "#%02x%02x%02x" % (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def gt_colour(node: dict) -> str | None:
    material = node.get("material")
    if isinstance(material, dict):
        base = material.get("base_color")
        return base if isinstance(base, str) else None
    return material if isinstance(material, str) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--corrupt", action="store_true",
                    help="换个方向问：把所有物体的颜色刻意改错，相似度掉多少。"
                         "掉得很少就说明这个指标看不见颜色，那颜色相关的改动就没法用它验收。")
    a = ap.parse_args()
    from gwm.compiler.compile import compile_program
    from gwm.config import load_config, site_name
    from gwm.feedback.likeness import likeness

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"]
    cfg = load_config([f for f in files if Path(f).is_file()])
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    harness = str(REPO / "scripts/node_harness.sh")
    weights = Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"]

    pairs_by_clip = {}
    for row in json.loads(Path(a.eval).read_text(encoding="utf-8"))["rows"]:
        pairs = ((row.get("metrics") or {}).get("matching") or {}).get("pairs") or {}
        pairs_by_clip[row["clip"]] = {p: g for g, p in pairs.items() if p}

    rows = []
    clips = [p for p in sorted(Path(a.report).iterdir())
             if (p / "game" / "program.json").is_file()][:a.limit]
    for clip_dir in clips:
        frames_json = clip_dir / "perception" / "frames" / "frames.json"
        if not frames_json.is_file():
            continue
        frames = json.loads(frames_json.read_text(encoding="utf-8"))["frames"]
        gt_file = Path(a.gt_root) / clip_dir.name / "program.json"
        if not gt_file.is_file():
            continue
        gt = {n["id"]: n for n in json.loads(gt_file.read_text(encoding="utf-8")).get("objects", [])}
        pairs = pairs_by_clip.get(clip_dir.name, {})
        base = json.loads((clip_dir / "game" / "program.json").read_text(encoding="utf-8"))

        swapped = json.loads(json.dumps(base))
        changed = 0
        if a.corrupt:
            # 把每个物体的颜色整体旋转色相半圈：形状、摆位、光照一个不动，只有颜色错。
            # 这是指标敏感度的直接测法——如果相似度几乎不掉，它就看不见颜色。
            for node in swapped.get("objects", []):
                node["material"] = _opposite(_current_colour(node))
                changed += 1
            label = "颜色全改错"
        else:
            for node in swapped.get("objects", []):
                truth = gt.get(pairs.get(node["id"], ""))
                colour = gt_colour(truth) if truth else None
                if colour:
                    node["material"] = colour
                    changed += 1
            label = "换成真值颜色"
        out = {}
        for tag, program in (("原样", base), (label, swapped)):
            game = work / clip_dir.name / tag
            result = compile_program(program, game, cfg)
            if not result["ok"]:
                out[tag] = None
                continue
            like = likeness(Path(result["game_dir"]), frames, work / clip_dir.name / f"lk_{tag}",
                            harness, weights)
            out[tag] = like.get("median") if like.get("ok") else None
        rows.append((clip_dir.name, changed, out.get("原样"), out.get(label)))
        print(f"  {clip_dir.name}: 换了 {changed} 个物体的颜色，"
              f"相似度 {out.get('原样')} -> {out.get(label)}", flush=True)

    usable = [(b, s) for _, _, b, s in rows if isinstance(b, (int, float)) and isinstance(s, (int, float))]
    if usable:
        print(f"\n{len(usable)} 条片段：相似度中位数 "
              f"{statistics.median(b for b, _ in usable):.4f} -> "
              f"{statistics.median(s for _, s in usable):.4f}")
        print("  这是天花板，不是修法——真实视频里没有真值颜色。"
              "涨得少就说明颜色不是保真度的主要缺口，#55 的闭环不该先做。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
