"""智能体工作区：该给的都给到，不该给的（留出视角、交互清单）一样都不能进去。"""
import json
import os

import pytest
from PIL import Image

from gwm.bench import agent, run


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    pack = tmp_path / "out" / "toy" / "pack"
    (pack / "agent" / "views").mkdir(parents=True)
    (pack / "eval" / "gt").mkdir(parents=True)
    Image.new("RGB", (8, 6), (10, 20, 30)).save(pack / "agent" / "views" / "train_001.jpg")
    Image.new("RGB", (8, 6), (90, 80, 70)).save(pack / "eval" / "gt" / "heldout_000.png")
    view = {"name": "train_001.JPG", "file": "views/train_001.jpg", "pos": [0, 1, 2], "quat": [0, 0, 0, 1],
            "fov_deg": 50, "aspect": 1.33, "width": 8, "height": 6}
    (pack / "agent" / "cameras.json").write_text(json.dumps({"views": [view]}), encoding="utf-8")
    (pack / "agent" / "points.json").write_text(json.dumps({"xyz": [[0, 0, 0]], "rgb": [[1, 2, 3]]}), encoding="utf-8")
    (pack / "agent" / "frame.json").write_text("{}", encoding="utf-8")
    (pack / "eval" / "cameras.json").write_text(json.dumps({"views": [{**view, "name": "heldout_000.JPG"}]}), encoding="utf-8")
    monkeypatch.setattr(agent, "scene_out", lambda cfg: tmp_path / "out" / "toy")
    return {"scene": "toy", "paths": {"out": "out"}, "agent": {"model": "m", "max_budget_usd": 1, "preview_views": 2},
            "manifest": "configs/bench/room.manifest.json", "checklist": "configs/bench/room.checklist.json"}


def test_workspace_has_inputs_task_and_tools(cfg, tmp_path):
    ws = agent.prepare(cfg, tmp_path / "ws")
    task = (ws / "TASK.md").read_text(encoding="utf-8")
    assert "{{" not in task                                   # 占位符都填上了
    manifest = json.loads((ws / "inputs" / "manifest.json").read_text(encoding="utf-8"))
    assert all(f"`{o['id']}`" in task for o in manifest)
    assert (ws / "inputs" / "views" / "train_001.jpg").is_file()
    for tool in ("preview.sh", "physics.sh"):
        assert os.access(ws / "tools" / tool, os.X_OK)
    assert "train_001" in (ws / "tools" / "preview.sh").read_text(encoding="utf-8")


def test_workspace_leaks_neither_held_out_views_nor_checklist(cfg, tmp_path):
    ws = agent.prepare(cfg, tmp_path / "ws")
    checklist = json.loads((run.REPO / cfg["checklist"]).read_text(encoding="utf-8"))
    names = [p.name for p in ws.rglob("*")]
    assert not any("heldout" in n or "checklist" in n for n in names)
    for path in ws.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".json", ".sh"}:
            text = path.read_text(encoding="utf-8")
            assert "heldout_000" not in text and "checklist" not in text, path
            assert checklist["note"] not in text
    # 名单只说是什么，不说能不能动
    manifest_text = (ws / "inputs" / "manifest.json").read_text(encoding="utf-8")
    assert not any(word in manifest_text for word in ("movable", "fixed", "rests", "hinge"))


def test_prepare_refuses_a_used_workspace(cfg, tmp_path):
    agent.prepare(cfg, tmp_path / "ws")
    with pytest.raises(FileExistsError):
        agent.prepare(cfg, tmp_path / "ws")


def test_no_manifest_leaves_granularity_to_the_agent(cfg, tmp_path):
    ws = agent.prepare(cfg, tmp_path / "ws", manifest=False)
    task = (ws / "TASK.md").read_text(encoding="utf-8")
    assert not (ws / "inputs" / "manifest.json").exists()
    assert "{{" not in task
    assert "armchair_cushion" not in task
