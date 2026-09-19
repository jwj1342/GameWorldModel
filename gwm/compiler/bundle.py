"""Bundle a validated program into a self-contained game/ directory (buildless ESM)."""
from __future__ import annotations
import json, os, shutil, time
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
    raise FileNotFoundError("node_modules not found; set GWM_NODE_ROOT/GWM_NODE_MODULES or untar deps/node-playwright-three.tar")

def _copy_vendor(dst: Path) -> None:
    nm = _node_modules()
    three_dst = dst / "vendor" / "three"
    (three_dst / "build").mkdir(parents=True, exist_ok=True)
    for f in ("three.module.js", "three.core.js"):
        src = nm / "three" / "build" / f
        if src.exists(): shutil.copy2(src, three_dst / "build" / f)
    jsm_src, jsm_dst = nm / "three" / "examples" / "jsm", three_dst / "examples" / "jsm"
    if not jsm_dst.exists():
        shutil.copytree(jsm_src, jsm_dst, ignore=shutil.ignore_patterns("*.d.ts", "libs", "loaders", "nodes", "tsl", "renderers", "postprocessing", "physics"))
    rapier_dst = dst / "vendor" / "rapier"; rapier_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nm / "@dimforge" / "rapier3d-compat" / "dist" / "rapier.mjs", rapier_dst / "rapier.mjs")

def bundle(program: dict, out_dir: str | Path, asset_manifest: dict | None = None, validation: dict | None = None) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    kdst = out / "kernel"
    if kdst.exists(): shutil.rmtree(kdst)
    shutil.copytree(KERNEL, kdst, ignore=shutil.ignore_patterns("index.html", "package.json"))
    shutil.copy2(KERNEL / "index.html", out / "index.html")
    _copy_vendor(out)
    (out / "program.json").write_text(json.dumps(program, indent=2, ensure_ascii=False))
    report = {"bundled_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kernel_files": sorted(p.name for p in KERNEL.glob("*.js")),
              "assets": asset_manifest or {}, "validation": validation or {}, "objects": len(program.get("objects", [])), "static": len(program.get("static", []))}
    (out / "compile_report.json").write_text(json.dumps(report, indent=2))
    return out
