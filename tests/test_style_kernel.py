"""把上面那个 node 测试挂进 pytest，有 node_modules 的时候就真跑。"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gwm.config import REPO


def test_style_reaches_the_runtime():
    node = os.environ.get("GWM_NODE_BINARY") or shutil.which("node")
    modules = Path(os.environ.get("GWM_NODE_MODULES", str(REPO / "node_modules")))
    if not node or not modules.is_dir():
        pytest.skip("要 node 和 GWM_NODE_MODULES（three.js）；不自动下载")
    link = REPO / "node_modules"
    created = False
    if not link.exists():
        link.symlink_to(modules); created = True
    try:
        run = subprocess.run([node, "--experimental-default-type=module", "--test",
                              str(REPO / "tests/test_style_kernel.mjs")],
                             capture_output=True, text=True, encoding="utf-8", timeout=120, cwd=REPO)
    finally:
        if created: link.unlink()
    print(run.stdout)
    assert run.returncode == 0, run.stdout + run.stderr
