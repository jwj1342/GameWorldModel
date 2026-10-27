"""把**真实查询图**和**库里那一项**并排存出来，直接看它们差在哪。

为什么到这一步。覆盖率 0.284 的原因，三条假设都被量化证伪了：

  嵌入本身   留一视角自匹配 0.990
  颜色       渲染侧只有 37.6% 的物体真被染色
  分辨率     缩略图降到 16px，留一视角仍 0.958；按裁剪尺寸分布预测 0.965

间接判据用完了。再从扶手椅上提第四个假设不如**直接看图**。

每个物体存一张拼图：左边是管线真正送进嵌入的那张裁剪图（`crop_object` 的输出，
已经抠过背景、合到缩略图底色上），右边是**真值资产**的四个缩略图。
两边看起来像不像，一眼就知道。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.assets.annotate import bbox_of, best_observation, crop_object   # noqa: E402
from gwm.perception.run import load_masks                                # noqa: E402

THUMBS = Path("/scratch/jwj/gwm/asset_thumbs")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    made = 0
    for clip_dir in sorted(Path(a.report).glob("*test_*")):
        if made >= a.n:
            break
        clip = "test_" + clip_dir.name.split("test_")[-1]
        gp = Path(a.gt_root) / clip / "program.json"
        ev_p = clip_dir / "perception" / "evidence.json"
        fr_p = clip_dir / "perception" / "frames" / "frames.json"
        if not (gp.is_file() and ev_p.is_file() and fr_p.is_file()):
            continue
        gprog = json.loads(gp.read_text(encoding="utf-8"))
        refs = [((n.get("geom") or {}).get("ref") or "")
                for n in (gprog.get("objects") or []) + (gprog.get("static") or [])]
        refs = [r for r in refs if r]
        if not refs:
            continue
        evidence = json.loads(ev_p.read_text(encoding="utf-8"))
        frames = json.loads(fr_p.read_text(encoding="utf-8"))["frames"]
        by_idx = {f.get("index"): f for f in frames}
        masks = load_masks(clip_dir / "perception")

        for obj in (evidence.get("objects") or []):
            if made >= a.n:
                break
            obs = best_observation(obj)
            if obs is None:
                continue
            frame = by_idx.get(obs.get("frame_index"))
            mask = (masks.get(obj["id"]) or {}).get(obs.get("frame_index"))
            if not frame or mask is None:
                continue
            rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
            box, box_size = bbox_of(obs)
            if box_size and box_size[0]:
                k = rgb.shape[1] / float(box_size[0])
                box = [v * k for v in box]
            crop = crop_object(rgb, np.asarray(mask), box)
            if crop is None:
                continue
            q = Image.fromarray(crop)
            ref = refs[made % len(refs)]
            tiles = []
            for v in range(4):
                f = THUMBS / f"{ref.replace('/', '-')}_{v}.png"
                if f.is_file():
                    tiles.append(Image.open(f).convert("RGB"))
            H = 160
            qs = q.resize((max(1, int(q.width * H / max(1, q.height))), H), Image.NEAREST)
            W = qs.width + sum(t.width for t in tiles) + 8 * (1 + len(tiles))
            canvas = Image.new("RGB", (W, H + 4), (255, 255, 255))
            x = 4
            canvas.paste(qs, (x, 2)); x += qs.width + 8
            for t in tiles:
                canvas.paste(t.resize((H, H)), (x, 2)); x += H + 8
            name = f"{made:02d}_{clip}_{obj['id']}_{q.width}x{q.height}_{ref.replace('/', '-')}.png"
            canvas.save(out / name)
            made += 1
    print(f"存了 {made} 张对比图到 {out}")
    print("左边=管线真正送进嵌入的查询图，右边=**同片段里某个**真值资产的 4 个缩略图。")
    print("右边那个不是左边的配对（配对条件含被测量，这坑踩过）——这张图要看的是")
    print("**两侧的域差**：真实查询和库里的条目长得像不像，而不是某一对匹不匹配。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
