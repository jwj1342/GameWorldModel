"""资产目录：把一个 CC0 GLB 库扫成一张可检索的表。

库的布局按 Kenney 镜像来：`3d/<kit>/<名字>.glb`，加一份 `3d/kits.tsv` 给出 kit 的中文可读标题。
目录只记名字和出处，不记包围盒——包围盒在内核加载 GLB 时现算（three.js 的 Box3），
那里拿到的是真几何，不必在 Python 这边再写一个 GLB 解析器。
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

CATALOG_NAME = "catalog.json"


@dataclass(frozen=True)
class Asset:
    id: str                  # kit/名字，全库唯一
    path: str                # 相对库根的路径
    kit: str                 # kit 的 slug
    kit_title: str           # kit 的可读标题
    name: str                # 文件名去掉后缀


def library_root(cfg: dict | None = None) -> Path | None:
    """库根目录。配置优先，其次环境变量 GWM_ASSETS。没有就返回 None——
    没有库不是错误，只是退回基本体，整条管线照常跑。"""
    configured = ((cfg or {}).get("assets") or {}).get("library")
    for candidate in (configured, os.environ.get("GWM_ASSETS")):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def _kit_titles(root: Path) -> dict[str, str]:
    table = root / "3d" / "kits.tsv"
    if not table.is_file():
        return {}
    with table.open(encoding="utf-8", newline="") as handle:
        return {row["slug"]: row.get("title", row["slug"])
                for row in csv.DictReader(handle, delimiter="\t") if row.get("slug")}


def build_catalog(root: str | Path) -> list[Asset]:
    """扫一遍库。按 id 排序，所以同一个库每次扫出来的顺序一样。"""
    root = Path(root)
    titles = _kit_titles(root)
    assets = []
    for glb in sorted((root / "3d").rglob("*.glb")):
        kit = glb.relative_to(root / "3d").parts[0]
        assets.append(Asset(id=f"{kit}/{glb.stem}", path=str(glb.relative_to(root)),
                            kit=kit, kit_title=titles.get(kit, kit), name=glb.stem))
    return assets


def write_catalog(root: str | Path, assets: list[Asset]) -> Path:
    out = Path(root) / CATALOG_NAME
    out.write_text(json.dumps({"count": len(assets), "assets": [asdict(a) for a in assets]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_catalog(root: str | Path) -> list[Asset]:
    """读目录文件；没有就现扫一遍。现扫一次 5000 个文件在登录节点上是秒级的。"""
    root = Path(root)
    cached = root / CATALOG_NAME
    if cached.is_file():
        return [Asset(**row) for row in json.loads(cached.read_text(encoding="utf-8"))["assets"]]
    return build_catalog(root)
