#!/usr/bin/env python
"""用**纯图像检索**给物体命名，比检测器的短语准不准。

背景（#56）：物体的 `class` 现在直接来自检测器给的短语标签，而实测即使把场景里确切
存在的类名喂给检测器，它在**正确定位**的物体上也只有 12.5% 贴对了标签。类名决定
检索哪个资产，资产决定场景长什么样，所以这是「复原得像不像」最直接的一环。

资产检索救不了它：`choose_assets` 用短语做文字候选集、图像只负责在候选里重排，
短语错了候选集里根本没有正确的资产。

但 `index.scores(vector)` 返回的是**整个资产库**的相似度。所以还有一条没试过的路：
完全绕开短语，拿物体截图在全库里找最近的资产，用那个资产的名字当类名。

> 和之前排除的那条不是一回事。被排除的是「图像检索分能不能**判真假**」（不能——
> 背景碎片和某个资产的相似度，跟真物体与它最像的那个资产的相似度一样）。这里问的
> 是「给一个**已知为真**的物体命名」，是另一个问题。

只在评测判定配对上的物体上算——那些是确认为真的，命名问题才谈得上。

用法:
  python -m scripts.probe_image_naming --eval <eval.json> --gt-root <真值根>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.assets.annotate import best_observation, bbox_of, crop_object   # noqa: E402
from gwm.assets.embed import AssetIndex, Embedder                        # noqa: E402
from gwm.assets.library import library_root, load_catalog                # noqa: E402
from gwm.config import load_config, site_name                            # noqa: E402
from gwm.perception.run import load_masks                                # noqa: E402
from gwm.taxonomy import slug                                            # noqa: E402


def tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--top", type=int, default=3, help="图像最近的前几个都看一眼")
    a = ap.parse_args(argv)

    from PIL import Image

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"]
    cfg = load_config([str(f) for f in files if f.is_file()])
    root = library_root(cfg)
    if root is None:
        print("没有资产库"); return 1
    index = AssetIndex.load(Path(root) / "dinov2_index.npz")
    embedder = Embedder(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])
    catalog = {asset.id: asset for asset in load_catalog(root)}

    rows = json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", [])
    n = phrase_ok = image_ok = image_topk_ok = 0
    samples = []
    for row in rows:
        run = Path(row["run"])
        evp, prog = run / "perception" / "evidence.json", run / "program.json"
        gtp = Path(a.gt_root) / row["clip"] / "program.json"
        if not (evp.is_file() and prog.is_file() and gtp.is_file()):
            continue
        pairs = ((row.get("metrics") or {}).get("matching") or {}).get("pairs") or {}
        gt_cls = {o["id"]: str(o.get("class", "")).lower()
                  for o in json.loads(gtp.read_text(encoding="utf-8")).get("objects", [])}
        pred = {o["id"]: o for o in json.loads(prog.read_text(encoding="utf-8")).get("objects", [])}
        evidence = json.loads(evp.read_text(encoding="utf-8"))
        by_id = {}
        for obj in evidence.get("objects", []):
            by_id[obj["id"]] = obj
            by_id.setdefault(slug(obj["id"]), obj)
        frames = json.loads((run / "perception" / "frames" / "frames.json").read_text())["frames"]
        frame_by_index = {f.get("index"): f for f in frames}
        masks = load_masks(run / "perception")

        for g, q in pairs.items():
            if not q or q not in pred or g not in gt_cls:
                continue
            truth = gt_cls[g]
            obj = by_id.get(q)
            observation = best_observation(obj) if obj else None
            if observation is None:
                continue
            frame = frame_by_index.get(observation.get("frame_index"))
            mask = (masks.get(obj["id"]) or {}).get(observation.get("frame_index"))
            if not frame or mask is None:
                continue
            rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
            box, box_size = bbox_of(observation)
            if box_size and box_size[0]:
                k = rgb.shape[1] / float(box_size[0])
                box = [v * k for v in box]
            crop = crop_object(rgb, np.asarray(mask), box)
            if crop is None:
                continue
            vector = embedder.embed_images([Image.fromarray(crop)])[0]
            scores = index.scores(vector)                     # 全库，不经过文字
            best = sorted(scores.items(), key=lambda kv: -kv[1])[:a.top]

            n += 1
            phrase = str(pred[q].get("class", "")).lower()
            truth_tok = tokens(truth)
            phrase_hit = bool(truth_tok & tokens(phrase))
            top1 = tokens(catalog[best[0][0]].name) | tokens(catalog[best[0][0]].kit) if best and best[0][0] in catalog else set()
            topk = set()
            for aid, _ in best:
                if aid in catalog:
                    topk |= tokens(catalog[aid].name) | tokens(catalog[aid].kit)
            phrase_ok += phrase_hit
            image_ok += bool(truth_tok & top1)
            image_topk_ok += bool(truth_tok & topk)
            if len(samples) < 12:
                samples.append((row["clip"], truth, phrase,
                                best[0][0] if best else "—", bool(truth_tok & top1)))

    if not n:
        print("没有可用样本"); return 1
    print(f"评测判定配对上、且能取到截图的物体 {n} 个\n")
    print(f"  检测器短语当类名      判对 {phrase_ok:3d}（{phrase_ok/n:.1%}）  <- 现在的做法")
    print(f"  纯图像最近的那个资产   判对 {image_ok:3d}（{image_ok/n:.1%}）")
    print(f"  纯图像前 {a.top} 个里有一个对 判对 {image_topk_ok:3d}（{image_topk_ok/n:.1%}）")
    print("\n逐个看（真值 / 短语 / 图像最近的资产 / 图像对不对）：")
    for clip, truth, phrase, asset, ok in samples:
        print(f"  {clip:<12}{truth:<14}{phrase:<16}{asset:<34}{'对' if ok else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
