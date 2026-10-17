"""资产库的图像索引：把每个资产的缩略图嵌成向量，供「这个物体长得像哪个资产」检索。

为什么要图像这一路：光靠文件名检索会挑错东西。查 "toy train" 时，train 这个 kit 里
`train-connector`（车厢之间的连接件）和 `train-locomotive-a`（机车）名字里都有 train，
文字分数分不出高下，而画面一比就知道哪个像视频里那辆车。

用 DINOv2 而不是 CLIP：集群上已经有这份权重（评测里算渲染图与原帧相似度就用它），
不用再下一份；而且这里要的是「看起来像不像」这种纯视觉相似度，正是它擅长的。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Embedder:
    """DINOv2 图像嵌入。向量都做了 L2 归一化，所以点积就是余弦相似度。"""

    def __init__(self, weights_dir: str | Path, batch: int = 32):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(str(weights_dir))
        self.model = AutoModel.from_pretrained(str(weights_dir)).to(self.device).eval()
        self.batch = batch

    def embed_images(self, images: list) -> np.ndarray:
        """images 可以是 PIL 图或 numpy RGB 数组。返回 (N, D)。"""
        out = []
        for start in range(0, len(images), self.batch):
            chunk = images[start:start + self.batch]
            inputs = self.proc(images=chunk, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                result = self.model(**inputs)
            vectors = (result.pooler_output if getattr(result, "pooler_output", None) is not None
                       else result.last_hidden_state[:, 0])
            out.append(vectors.float().cpu().numpy())
        stacked = np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)
        return stacked / (np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-9)


class AssetIndex:
    """资产 id -> 若干视角的向量。查询时对视角取最大值：资产在库里是正着摆的，
    视频里的物体朝向任意，只要有一个视角对得上就算像。"""

    def __init__(self, asset_ids: np.ndarray, vectors: np.ndarray):
        self.asset_ids = asset_ids
        self.vectors = vectors

    @classmethod
    def load(cls, path: str | Path) -> "AssetIndex":
        data = np.load(path, allow_pickle=False)
        return cls(data["asset_ids"], data["vectors"])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, asset_ids=self.asset_ids, vectors=self.vectors)
        return path

    def scores(self, query: np.ndarray) -> dict[str, float]:
        """返回 资产id -> [0,1] 的相似度（余弦相似度裁到非负）。"""
        sims = self.vectors @ (query / (np.linalg.norm(query) + 1e-9))
        best: dict[str, float] = {}
        for asset_id, sim in zip(self.asset_ids, sims):
            value = float(max(sim, 0.0))
            key = str(asset_id)
            if value > best.get(key, -1.0):
                best[key] = round(value, 4)
        return best


def build_index(thumbs_dir: str | Path, weights_dir: str | Path, out_path: str | Path,
                batch: int = 32) -> dict:
    """读缩略图目录（thumbnails.mjs 的产物），嵌入后存成一个 npz。"""
    from PIL import Image
    thumbs_dir = Path(thumbs_dir)
    manifest = json.loads((thumbs_dir / "thumbs.json").read_text(encoding="utf-8"))
    embedder = Embedder(weights_dir, batch=batch)
    ids, vectors = [], []
    pending_ids, pending_images = [], []

    def flush():
        if not pending_images:
            return
        vectors.append(embedder.embed_images(pending_images))
        ids.extend(pending_ids)
        pending_ids.clear(); pending_images.clear()

    for asset in manifest["assets"]:
        for name in asset["files"]:
            file = thumbs_dir / name
            if not file.is_file():
                continue
            pending_images.append(Image.open(file).convert("RGB"))
            pending_ids.append(asset["id"])
            if len(pending_images) >= batch * 8:
                flush()
    flush()
    index = AssetIndex(np.array(ids, dtype=object).astype("U"),
                       np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 1), dtype=np.float32))
    index.save(out_path)
    return {"assets": len(manifest["assets"]), "views_embedded": len(ids),
            "dim": int(index.vectors.shape[1]) if index.vectors.size else 0,
            "out": str(out_path)}
