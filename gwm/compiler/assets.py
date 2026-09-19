"""AssetResolver: strategy chain. MVP: everything resolves to primitive fallbacks handled by the kernel (scene.js fallbackForClass).
The compiler only records what would be needed so later stages can plug in a library or a generator."""
from __future__ import annotations

def resolve_assets(program: dict) -> dict:
    """Return a manifest of asset requests. The kernel renders asset/generated nodes as primitives from `extent` + `class`."""
    requests = []
    for o in program.get("objects", []):
        g = o.get("geom", {})
        if g.get("kind") in ("asset", "generated"):
            requests.append({"id": o["id"], "class": o.get("class"), "query": g.get("query") or g.get("ref"), "extent": g.get("extent"), "resolved": "primitive_fallback"})
    return {"strategy": ["primitive"], "requests": requests}
