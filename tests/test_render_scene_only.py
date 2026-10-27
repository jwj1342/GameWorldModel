"""Python -> Node argument wiring, without a browser or actual rendering."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest
import gwm.feedback.render as renderer

@pytest.mark.parametrize('scene_only',[False,True])
def test_scene_only_is_explicit_keyword_opt_in(monkeypatch,tmp_path,scene_only):
    captured=[]
    def harness(script,args):
        captured.append((script,args))
        (tmp_path/'index.json').write_text(json.dumps({'frames':[]}))
        return subprocess.CompletedProcess([],0,'','')
    monkeypatch.setattr(renderer,'run_harness',harness)
    renderer.render('synthetic-game',[0.1],tmp_path,960,540,scene_only=scene_only)
    assert ('--scene-only' in captured[0][1])==scene_only
    assert captured[0][0]=='render.mjs'

def test_existing_call_defaults_to_legacy(monkeypatch,tmp_path):
    def harness(script,args):
        assert '--scene-only' not in args
        (tmp_path/'index.json').write_text('{}')
        return subprocess.CompletedProcess([],0,'','')
    monkeypatch.setattr(renderer,'run_harness',harness)
    renderer.render('synthetic',[0],tmp_path,320,180)

def test_scene_only_does_not_coerce_strings(tmp_path):
    with pytest.raises(TypeError,match='bool'):
        renderer.render('synthetic',[0],tmp_path,320,180,scene_only='false')


def test_scene_only_node_controls():
    """Run CPU runtime/wiring fixtures as part of pytest; never download dependencies."""
    node = os.environ.get("GWM_NODE_BINARY") or shutil.which("node")
    repo = Path(renderer.REPO)
    if not node or not (repo / "node_modules" / "three").is_dir() or not (
        repo / "node_modules" / "@dimforge" / "rapier3d-compat"
    ).is_dir():
        pytest.skip("scene-only CPU controls need Node, three and Rapier installed locally")
    result = subprocess.run(
        [node, "--experimental-default-type=module", "--test",
         str(repo / "tests/test_scene_only_harness.mjs"),
         str(repo / "tests/test_scene_only_runtime.mjs")],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
