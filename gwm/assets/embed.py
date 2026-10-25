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


def _to_silhouette(img):
    """缩略图 -> 黑底白形的二值轮廓图。

    **不能用 alpha 通道抠**：这批缩略图的 alpha 全是 255（有 alpha 通道不等于
    用了它）。背景是均匀浅灰 (242,242,242)，按离背景色的距离抠前景。
    这个坑踩过一次，当时造出了「74/74 全对」的假结果。
    """
    import numpy as _np
    from PIL import Image as _Image
    a = _np.asarray(img.convert("RGB"), dtype=_np.int16)
    bg = _np.median(_np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]), axis=0)
    fg = (_np.abs(a - bg[None, None, :]).max(axis=2) > 24)
    out = _np.where(fg[:, :, None], 255, 0).astype(_np.uint8).repeat(3, axis=2)
    return _Image.fromarray(out)


def build_index(thumbs_dir: str | Path, weights_dir: str | Path, out_path: str | Path,
                batch: int = 32, gray: bool = False, silhouette: bool = False,
                downsample: int = 0) -> dict:
    """读缩略图目录（thumbnails.mjs 的产物），嵌入后存成一个 npz。

    `gray=True` 时两侧都转灰度再嵌入。为什么要这个开关：这个数据集**给每个物体
    重新上色**（`material` 字段给的是任意 base_color / roughness / metalness /
    emissive），而库里的缩略图是资产的原生外观。DINOv2 对外观很敏感，于是查询
    （一个青色的物体）和库里那一项（原生贴图的同一个网格）看起来像两个东西。

    这不是猜的：留一视角自匹配（纯渲染对纯渲染、同材质、只差视角）前 64 名
    覆盖 **0.990**，而视频裁剪图对缩略图只有 0.284。嵌入跨视角没问题，
    差的是域，而域里最大的一项就是颜色。
    """
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
            img = Image.open(file).convert("RGB")
            if downsample > 0:
                # **把缩略图降到视频里物体的真实尺寸再嵌入。**
                # 实测视频里物体的裁剪图短边中位只有 16px，而缩略图是 160px。
                # 留一视角自匹配拿 160px 对 160px 得到 0.990，它**刻意回避了
                # 分辨率这个变量**。降采样之后再做留一视角，如果塌到 0.28 附近，
                # 那就说明覆盖率 0.284 基本全是分辨率造成的，和颜色关系不大。
                w, h = img.size
                k = downsample / float(min(w, h))
                img = img.resize((max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR)
                img = img.resize((w, h), Image.BICUBIC)
            if silhouette:
                # 两侧都只剩形状：数据集给物体重新上色、还改 roughness/metalness/
                # 自发光，外观这个通道在这个数据集上天然失真。轮廓没有外观。
                img = _to_silhouette(img)
            elif gray:
                img = img.convert("L").convert("RGB")
            pending_images.append(img)
            pending_ids.append(asset["id"])
            if len(pending_images) >= batch * 8:
                flush()
    flush()
    index = AssetIndex(np.array(ids, dtype=object).astype("U"),
                       np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 1), dtype=np.float32))
    index.save(out_path)
    return {"assets": len(manifest["assets"]), "views_embedded": len(ids), "gray": gray, "silhouette": silhouette, "downsample": downsample,
            "dim": int(index.vectors.shape[1]) if index.vectors.size else 0,
            "out": str(out_path)}
