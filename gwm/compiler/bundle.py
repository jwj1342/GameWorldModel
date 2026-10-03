"""Bundle a validated program into a self-contained game/ directory (buildless ESM)."""
from __future__ import annotations
import json, os, re, shutil, time

from .ids import id_to_color, registry_order
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "kernel"

def _node_modules() -> Path:
    root = os.environ.get("GWM_NODE_ROOT") or (os.environ.get("SLURM_TMPDIR") and str(Path(os.environ["SLURM_TMPDIR"]) / "node"))
    for cand in [os.environ.get("GWM_NODE_MODULES"), root and Path(root) / "node_modules", REPO / "node_modules"]:
        if cand and Path(cand).exists():
            return Path(cand)
    # extract the staged bundle on demand (same layout node_harness.sh uses)
    tar = Path(os.environ.get("GWM_DEPS", "/project/aip-zhouyang/jwj/GameWorldModel/deps")) / "node-playwright-three.tar"
    if root and tar.exists():
        import subprocess
        Path(root).mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(tar), "-C", root], check=True)
        if (Path(root) / "node_modules").exists(): return Path(root) / "node_modules"
    raise FileNotFoundError(
        "找不到 node_modules。在仓库根目录执行 npm install 就行；"
        "在没有外网的集群节点上，用 scripts/stage_node_deps.sh 打好的 tar，或设 GWM_NODE_ROOT / GWM_NODE_MODULES。")

_IMPORT_RE = re.compile(r"""from\s+['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]""")

def _imports(path: Path) -> list[str]:
    return [a or b for a, b in _IMPORT_RE.findall(path.read_text())]

def _addon_imports(root: Path) -> list[str]:
    """Paths under three/examples/jsm imported by any kernel module, e.g. controls/OrbitControls.js."""
    out = []
    for f in root.rglob("*.js"):
        out += [spec[len("three/addons/"):] for spec in _imports(f) if spec.startswith("three/addons/")]
    return out

def _relative_imports(src: Path, rel: str) -> list[str]:
    """Relative imports inside an addon file, as paths relative to the jsm root."""
    return [os.path.normpath(str(Path(rel).parent / spec)) for spec in _imports(src) if spec.startswith(".")]

def _copy_vendor(dst: Path) -> None:
    nm = _node_modules()
    three_dst = dst / "vendor" / "three"
    (three_dst / "build").mkdir(parents=True, exist_ok=True)
    for f in ("three.module.js", "three.core.js"):
        src = nm / "three" / "build" / f
        if src.exists(): shutil.copy2(src, three_dst / "build" / f)
    # three/addons: copy only the files the kernel actually imports (and what they import in turn),
    # otherwise every game carries ~10 MB of examples it never loads
    jsm_src, jsm_dst = nm / "three" / "examples" / "jsm", three_dst / "examples" / "jsm"
    wanted, seen = set(_addon_imports(KERNEL)), set()
    while wanted:
        rel = wanted.pop()
        if rel in seen: continue
        seen.add(rel)
        src = jsm_src / rel
        if not src.exists(): continue
        out = jsm_dst / rel; out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, out)
        for nxt in _relative_imports(src, rel): wanted.add(nxt)
    rapier_dst = dst / "vendor" / "rapier"; rapier_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nm / "@dimforge" / "rapier3d-compat" / "dist" / "rapier.mjs", rapier_dst / "rapier.mjs")

def kernel_config(program: dict, cfg: dict | None = None) -> dict:
    """内核要用、但不属于场景本身的数据。写成文件是为了让 Python 这边当唯一来源，
    JS 不再自己算一遍物体 ID 颜色、也不再自己抄一份收集物和危险物的类别表。"""
    b = (cfg or {}).get("binding", {})
    return {
        "registry": [{**r, "color": list(id_to_color(r["idx"]))} for r in registry_order(program)],
        "collectible_classes": b.get("collectible_classes", ["coin", "gem", "key", "star", "pickup", "ball", "marble"]),
        "hazard_classes": b.get("hazard_classes", ["lava", "spike", "water", "fire", "acid"]),
    }

def bundle(program: dict, out_dir: str | Path, asset_manifest: dict | None = None, validation: dict | None = None, cfg: dict | None = None) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    kdst = out / "kernel"
    if kdst.exists(): shutil.rmtree(kdst)
    shutil.copytree(KERNEL, kdst, ignore=shutil.ignore_patterns("index.html", "package.json"))
    shutil.copy2(KERNEL / "index.html", out / "index.html")
    _copy_vendor(out)
    (out / "program.json").write_text(json.dumps(program, indent=2, ensure_ascii=False))
    (out / "kernel_config.json").write_text(json.dumps(kernel_config(program, cfg), indent=1))
    report = {"bundled_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kernel_files": sorted(p.name for p in KERNEL.glob("*.js")),
              "assets": asset_manifest or {}, "validation": validation or {}, "objects": len(program.get("objects", [])), "static": len(program.get("static", []))}
    (out / "compile_report.json").write_text(json.dumps(report, indent=2))
    return out
