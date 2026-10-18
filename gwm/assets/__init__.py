"""CC0 资产库：目录、检索、对齐。

场景的「表征」那一半从这里来：视频里检到一个物体，拿类名和裁剪帧去库里找一个真网格，
按证据给的包围盒缩放摆进去。碰撞体不走这条路——它仍然由包围盒生成，所以换不换资产
都不影响这个物体能不能被拖动、能不能被踩上去。两半分开是有意的，见 docs/design-notes.md。
"""
from .library import Asset, build_catalog, load_catalog, library_root
from .retrieve import retrieve, score_asset, tokenize
from .annotate import choose_assets, crop_object

__all__ = ["Asset", "build_catalog", "load_catalog", "library_root",
           "retrieve", "score_asset", "tokenize", "choose_assets", "crop_object"]
