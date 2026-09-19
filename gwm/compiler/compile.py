"""CLI: validate + resolve assets + bundle.  python -m gwm.compiler.compile program.json out/game"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from .validate import validate, format_errors
from .assets import resolve_assets
from .bundle import bundle

def compile_program(program: dict, out_dir: str | Path) -> dict:
    report = validate(program)
    if not report["ok"]:
        return {"ok": False, "validation": report}
    manifest = resolve_assets(program)
    game_dir = bundle(program, out_dir, manifest, report)
    return {"ok": True, "validation": report, "assets": manifest, "game_dir": str(game_dir)}

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("program"); ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    program = json.loads(Path(a.program).read_text())
    res = compile_program(program, a.out_dir)
    if not res["ok"]:
        print("VALIDATION FAILED\n" + format_errors(res["validation"])); sys.exit(2)
    print(json.dumps({"ok": True, "game_dir": res["game_dir"], "warnings": res["validation"]["warnings"], "asset_requests": len(res["assets"]["requests"])}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
