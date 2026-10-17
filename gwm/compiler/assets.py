"""把程序里的 asset 几何解析成真正的 GLB 文件。

程序里写 `{"kind": "asset", "query": "toy train", "extent": [...]}` 的时候，它只说了
「这里有一个火车那么大的东西」。以前这句话到内核就退化成一个盒子，所以重建出来的场景
永远是一堆方块。这里负责把它换成库里一个真网格。

要紧的是这一步**只决定长什么样**。碰撞体仍然由 extent 生成（见 kernel/scene.js 的
buildNode），所以换不换网格都不影响这个物体能不能被拖动、能不能被踩上去、物理怎么算。
场景的表征和交互是两半，这条边界就在这里。

命中分低于阈值时保持退回基本体：塞一个不相干的网格进去比放一个盒子更糟，它会让
「重建得像不像」这个指标变得没法解释。
"""
from __future__ import annotations

import re
from pathlib import Path

from ..assets import library_root, load_catalog, retrieve

DEFAULT_MIN_SCORE = 0.45
_SAFE = re.compile(r"[^a-z0-9_-]+")


def _safe_name(node_id: str) -> str:
    return _SAFE.sub("-", node_id.lower()).strip("-") or "asset"


def resolve_assets(program: dict, cfg: dict | None = None) -> dict:
    """返回资产清单。每条要么指向一个真 GLB，要么标明退回了基本体。"""
    settings = (cfg or {}).get("assets") or {}
    root = None if settings.get("enabled") is False else library_root(cfg)
    catalog = load_catalog(root) if root else []
    min_score = float(settings.get("min_score", DEFAULT_MIN_SCORE))
    top_k = int(settings.get("candidates", 3))
    by_id = {a.id: a for a in catalog}

    requests = []
    for node in program.get("objects", []):
        geom = node.get("geom", {})
        if geom.get("kind") not in ("asset", "generated"):
            continue
        query = geom.get("query") or node.get("class") or ""
        request = {"id": node["id"], "class": node.get("class"), "query": query,
                   "extent": geom.get("extent"), "resolved": "primitive_fallback"}
        # 程序里写了 ref 就按 ref 来。那是 gwm/assets/annotate.py 看过画面之后挑的，
        # 比这里只看文字的检索靠谱；这里的文字检索是没有 ref 时的兜底。
        chosen = by_id.get(geom.get("ref")) if geom.get("ref") else None
        if chosen is not None:
            asset, score, how = chosen, 1.0, "library_ref"
        else:
            hits = retrieve(query, catalog, k=top_k) if catalog else []
            if hits:
                request["candidates"] = [{"asset_id": a.id, "score": s} for a, s in hits]
            if geom.get("ref"):
                request["ref_miss"] = geom["ref"]
            asset, score, how = (hits[0][0], hits[0][1], "library") if hits and hits[0][1] >= min_score else (None, 0.0, None)
        if asset is not None:
            request.update(resolved=how, asset_id=asset.id, kit=asset.kit,
                           score=score, license="CC0",
                           source=str(Path(root) / asset.path),
                           file=f"assets/{_safe_name(node['id'])}.glb")
        requests.append(request)

    strategy = ["library", "primitive"] if catalog else ["primitive"]
    return {"strategy": strategy, "library": str(root) if root else None,
            "min_score": min_score, "requests": requests}
