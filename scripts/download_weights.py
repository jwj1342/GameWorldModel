#!/usr/bin/env python3
"""把感知用到的模型权重下载到配置里的 paths.weights 目录。

    python3 scripts/download_weights.py              # 全部下载，约 9 GB
    python3 scripts/download_weights.py --list       # 只看要下哪些

VGGT 的 5 GB 权重走 HuggingFace 的 CDN，有些机房的代理会拦，那种情况下脚本会给出直接用 curl 下载的命令。
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwm.config import load_config

# 仓库名, 本地目录名(configs 里的 key), 只要这些文件, 大概多大
MODELS = [
    ("facebook/VGGT-1B", "vggt", ["*.safetensors", "*.json", "*.md"], "5 GB", "相机位姿和深度"),
    ("facebook/sam2.1-hiera-large", "sam2", None, "1.8 GB", "视频分割和跨帧跟踪"),
    ("IDEA-Research/grounding-dino-tiny", "grounding_dino", None, "1.4 GB", "按名词短语找物体"),
    ("facebook/dinov2-base", "dinov2", None, "0.7 GB", "渲染画面和视频的整体相似度"),
    ("depth-anything/Depth-Anything-V2-Small-hf", "depth_anything", None, "0.1 GB", "VGGT 失败时的兜底深度"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只打印清单，不下载")
    ap.add_argument("--only", default=None, help="只下载某一个，填 configs 里的 key，比如 vggt")
    a = ap.parse_args()
    cfg = load_config()
    root = Path(cfg["paths"]["weights"]); root.mkdir(parents=True, exist_ok=True)
    print(f"权重目录 {root}（site={cfg['site']}）\n")
    for repo, key, allow, size, what in MODELS:
        if a.only and a.only != key: continue
        name = cfg["weights"][key]; dst = root / name
        mark = "已有" if (dst / ".complete").exists() else "要下"
        print(f"  [{mark}] {name:28s} {size:>7s}  {what}  ({repo})")
        if a.list or mark == "已有": continue
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(repo_id=repo, local_dir=str(dst), allow_patterns=allow)
            (dst / ".complete").touch()
            print(f"         下载完成")
        except Exception as e:
            print(f"         失败: {type(e).__name__}: {str(e)[:160]}")
            if key == "vggt":
                print(f"         可以手动下: curl -L -o {dst}/model.safetensors \\\n"
                      f"           https://huggingface.co/facebook/VGGT-1B/resolve/main/model.safetensors")
    if a.list: print("\n加 --only <key> 可以只下其中一个。")

if __name__ == "__main__":
    main()
