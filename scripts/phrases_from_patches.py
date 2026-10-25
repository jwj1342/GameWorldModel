"""不经过检测，直接从关键帧的网格 patch 拿**具体**的资产名当检测短语。

**为什么要它。** 今天量到：真值类名当检测短语，命中 14.9%、同套件 55.3%（分割和
几何全是神经的），而写死的默认词 0/69、资产库高频词 0/77、闭环检索名 0/76~2/112。
**管用的是「逐物体的具体名字」，不是「领域对口的词表」**——检出一个 "tile" 之后
库里有 496 个 tile，检索照样分不开。

闭环（用检索出来的候选名当短语）失败的原因是它**依赖第一遍的检测**：检测错了，
裁剪图就错，名字也错，环自己解不开。

这里绕开那个环：把关键帧切成网格，每块 patch 直接做图像检索，取 top-1 资产名。
名字来自**原始图像**而不是检测结果。全程不碰真值——资产库和图像都是推理时就有的。

短语数量要压住：实测给检测器 40 个短语时产出物体从 69 暴涨到 112、命中反而掉，
而真值类名那一档每条片段只有 5~8 个。少而具体才管用。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.assets.embed import AssetIndex, Embedder        # noqa: E402
from gwm.assets.library import library_root             # noqa: E402
from gwm.config import load_config, site_name           # noqa: E402


def asset_phrase(asset_id: str) -> str:
    """资产 id 转短语：驼峰拆词 + 去掉纯数字片段。和数据集生成器同一套规则。"""
    raw = asset_id.split("/")[-1].replace("-", " ").replace("_", " ")
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    toks = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", raw)
            if t and not t.isdigit()]
    return " ".join(toks[:3])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True, help="片段目录（里面有 clip.mp4）")
    ap.add_argument("--frames", type=int, default=3, help="取几帧")
    ap.add_argument("--grid", type=int, default=4, help="每帧切成 grid×grid 块")
    ap.add_argument("--max-phrases", type=int, default=8)
    a = ap.parse_args(argv)

    cfg = load_config([f for f in (REPO / "configs/default.yaml",
                                   REPO / f"configs/{site_name()}.yaml") if f.is_file()])
    root = library_root(cfg)
    if root is None:
        print(""); return 0
    index_file = Path(root) / "dinov2_index.npz"
    if not index_file.is_file():
        print(""); return 0
    index = AssetIndex.load(index_file)
    embedder = Embedder(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])

    import cv2
    cap = cv2.VideoCapture(str(Path(a.video_dir) / "clip.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    picks = [int(total * (i + 0.5) / a.frames) for i in range(a.frames)]
    votes: Counter = Counter()
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = np.asarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        for gy in range(a.grid):
            for gx in range(a.grid):
                patch = rgb[gy * h // a.grid:(gy + 1) * h // a.grid,
                            gx * w // a.grid:(gx + 1) * w // a.grid]
                if patch.size < 64:
                    continue
                try:
                    v = embedder.embed_images([Image.fromarray(patch)])[0]
                    scores = index.scores(v)
                except (ValueError, RuntimeError):
                    continue
                if not scores:
                    continue
                best = max(scores.items(), key=lambda kv: kv[1])[0]
                p = asset_phrase(best)
                if p:
                    votes[p] += 1
    cap.release()
    # 按出现次数取前几个：一个资产在多块 patch 上都赢，说明它真的在画面里。
    print(",".join(p for p, _n in votes.most_common(a.max_phrases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
