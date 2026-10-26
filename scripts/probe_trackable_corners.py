"""物体掩码里到底有多少个可跟踪的角点——「合成渲染纹理太少」是推断还是事实。

起因。把运动估计换成跨帧特征点对应（光流 + Kabsch）之后，真转 114° 的 spin 物体
解出 **0.000°**。那个零看起来像「光流一次都没跟成功」，但那是**推断**：代码在跟不到
的时候会沿用上一个值，所以 0.000 也可能来自别的分支。

这个脚本直接数：每个物体的掩码内，`goodFeaturesToTrack` 能找出几个角点。
数接近零，「纹理太少」就是事实而不是猜测；数不低而光流仍然失败，那是别的原因，
得重新查。

顺带报一个无量纲的量：**角点数 ÷ 掩码像素数**。物体在画面里小的时候角点自然少，
不除掉面积就会把「物体小」误读成「没纹理」。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.perception.masks import resize_mask                     # noqa: E402
from gwm.perception.run import load_geometry                      # noqa: E402


def main(argv: list[str]) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", action="append", required=True)
    ap.add_argument("--quality", type=float, default=0.01)
    a = ap.parse_args(argv)

    per_object: list[int] = []
    density: list[float] = []
    areas: list[int] = []
    zero = n = 0
    for d in a.report_dir:
        for clip_dir in sorted(Path(d).glob("*test_*")):
            per = clip_dir / "perception"
            gz, mz = per / "geometry.npz", per / "masks.npz"
            idx_path = per / "masks_index.json"
            frame_dir = per / "geom_frames"
            if not (gz.is_file() and mz.is_file() and idx_path.is_file() and frame_dir.is_dir()):
                continue
            files = sorted(frame_dir.glob("g_*.jpg"))
            geom = load_geometry(gz)
            M = np.load(mz)
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
            order = {f: i for i, f in enumerate(geom.frame_indices)}
            for oid, meta in (idx.get("objects") or {}).items():
                for fi in (meta.get("frames") or [])[:3]:        # 每个物体看前三帧够了
                    k = order.get(fi)
                    key = f"{oid}__{fi}"
                    if k is None or k >= len(files) or key not in M.files:
                        continue
                    img = cv2.imread(str(files[k]), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    m = np.asarray(M[key]).astype(np.uint8)
                    if m.shape != img.shape:
                        m = resize_mask(m.astype(bool), (img.shape[1], img.shape[0])).astype(np.uint8)
                    area = int(m.sum())
                    if area < 50:
                        continue
                    pts = cv2.goodFeaturesToTrack(img, maxCorners=400, qualityLevel=a.quality,
                                                  minDistance=4, mask=m)
                    c = 0 if pts is None else len(pts)
                    n += 1
                    zero += (c == 0)
                    per_object.append(c); areas.append(area)
                    density.append(c / area * 1000.0)            # 每千像素几个角点
    print("=" * 66)
    if not n:
        print("一个样本都没算成"); return 1
    print(f"物体-帧样本 {n} 个（质量阈值 {a.quality}）")
    print(f"  角点数：中位 {statistics.median(per_object):.1f}，"
          f"四分位 {np.percentile(per_object, 25):.1f} / {np.percentile(per_object, 75):.1f}，"
          f"最大 {max(per_object)}")
    print(f"  **一个角点都没有的占 {zero}/{n} = {zero / n:.3f}**")
    print(f"  掩码面积：中位 {statistics.median(areas):.0f} 像素")
    print(f"  角点密度（每千掩码像素）：中位 {statistics.median(density):.2f}")
    print("""
怎么读：
  「一个都没有」的比例高，或者密度接近零 → 纹理太少是**事实**，
  合成基准确实测不了基于对应关系的方法，要换有纹理的数据。
  角点不少而光流仍然失败 → 原因在别处（跟踪窗口、运动太大、遮挡），得重新查。
提醒：角点数要配着掩码面积看。物体在画面里小的时候角点自然少，
      不除面积会把「物体小」误读成「没纹理」。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
