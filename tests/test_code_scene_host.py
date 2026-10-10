"""CPU host integration with authored fixtures, never model-generated observations."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from gwm.config import REPO


def test_three_rapier_code_host(tmp_path):
    node = os.environ.get("GWM_NODE_BINARY") or shutil.which("node")
    dependencies = Path(os.environ.get("GWM_NODE_MODULES", str(REPO / "node_modules")))
    if not node or not dependencies.is_dir():
        pytest.skip("CPU integration requires Node and existing GWM_NODE_MODULES; no automatic download")
    sys.path.insert(0, str(REPO / "scripts"))
    from run_direct_code import build_game_dir
    stub = {
        "meta": {"clip": "synthetic_host_test", "duration": 2, "fps": 30, "units": "m", "up": "y"},
        "camera": {"keyframes": [{"t": 0, "pos": [8, 6, 8], "look_at": [0, 0, 0]}]},
        "static": [], "objects": [],
        "binding": {"template": "platformer_3p", "slots": {"walkable": "auto"}},
    }
    game = tmp_path / "game"
    build_game_dir("export function describe(THREE) { return globalThis.__hostFixture(THREE); }", stub, game)
    (game / "kernel/handwritten_fixture.js").write_text(
        (REPO / "examples/direct_code/model_scene.js").read_text(encoding="utf-8"), encoding="utf-8")
    # Node has no browser import map. Only resolve imports in this temporary bundle;
    # production Three/Rapier/host algorithms remain unchanged.
    (game / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    for module in (game / "kernel").glob("*.js"):
        text = module.read_text(encoding="utf-8")
        text = text.replace("from 'three'", "from '../vendor/three/build/three.module.js'")
        text = text.replace("from '@dimforge/rapier3d-compat'", "from '../vendor/rapier/rapier.mjs'")
        module.write_text(text, encoding="utf-8")
    env = {**os.environ, "GWM_TEST_CODE_GAME": str(game)}
    run = subprocess.run([node, "--test", str(REPO / "tests/test_code_scene_host.mjs")],
                         env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
    print(run.stdout)
    assert run.returncode == 0, run.stdout + run.stderr
