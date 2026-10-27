"""建一个**灰度**资产索引，用来验「0.284 的锅在颜色」。

留一视角自匹配（纯渲染对纯渲染、同材质、只差视角）前 64 名覆盖 0.990，
而视频裁剪图对缩略图只有 0.284。嵌入跨视角没问题，差的是域。

而域里最大的一项基本可以确定是颜色：这个数据集**给每个物体重新上色**，
`material` 给的是任意的 base_color / roughness / metalness / emissive
（比如 food/chinese 被渲染成 #4fce98 的青色），库里的缩略图却是资产原生外观。

两侧都转灰度再嵌入，就把颜色这一项去掉了。覆盖率涨 → 颜色确实是主因，
而且修法便宜（不用换模型）；不涨 → 锅在光照/粗糙度/自发光或别的，再往下查。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.assets.embed import build_index          # noqa: E402
from gwm.config import load_config, site_name     # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs", default="/scratch/jwj/gwm/asset_thumbs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--rgb", action="store_true", help="建彩色的（对照用）")
    ap.add_argument("--downsample", type=int, default=0,
                    help="先把缩略图降到短边 N 像素再放回原尺寸，模拟视频里物体的真实分辨率")
    ap.add_argument("--silhouette", action="store_true",
                    help="建**轮廓**索引：黑底白形，两侧都只剩形状")
    a = ap.parse_args(argv)
    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *a.config]
    cfg = load_config([f for f in files if Path(f).is_file()])
    weights = Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"]
    info = build_index(a.thumbs, weights, a.out,
                       gray=(not a.rgb and not a.silhouette),
                       silhouette=a.silhouette, downsample=a.downsample)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
