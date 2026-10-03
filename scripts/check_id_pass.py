#!/usr/bin/env python3
"""检查 ID 通道是不是干净的。

ID 和深度两个通道把数值编码进像素，任何后处理（抗锯齿、色彩空间变换）都会在物体边缘
把相邻编码混成不存在的颜色，Python 侧精确匹配会整片丢掉这些像素，掩码 IoU 被系统性压低。
这个坑踩过两次，所以固化成一个检查。

    python3 scripts/check_id_pass.py <game目录> <渲染目录>
"""
import glob, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwm.compiler.ids import id_to_color, registry_order


def main(game_dir: str, render_dir: str) -> int:
    program = json.loads((Path(game_dir) / "program.json").read_text())
    legit = {id_to_color(r["idx"]) for r in registry_order(program)} | {(0, 0, 0)}
    stray_total = 0
    for f in sorted(glob.glob(f"{render_dir}/id_*.png")):
        px = np.asarray(Image.open(f).convert("RGB")).reshape(-1, 3)
        cols, counts = np.unique(px, axis=0, return_counts=True)
        stray = int(sum(c for col, c in zip(map(tuple, cols), counts) if col not in legit))
        print(f"  {Path(f).name}: {len(cols)} 种颜色, 注册表 {len(legit) - 1} 项, 越界像素 {stray}")
        stray_total += stray
    print("  ID pass 干净" if not stray_total else f"  !! 越界像素共 {stray_total} 个，掩码 IoU 会被压低")
    return 0 if not stray_total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
