"""状态差分预实验里真有可能出错的两处：语言模板器的分组与欠定性，掩码配对。"""
import re

import numpy as np

from scripts.statediff.common import describe_scene, match_masks


def _obj(i, cls, mat, kind):
    return {"id": f"obj_{i}", "class": cls, "material": mat, "motion": {"type": kind, "axis": [1, 0, 0]}}


def test_模板器_同名分组且不泄露参数():
    program = {"objects": [_obj(1, "crate", "red", "prismatic"), _obj(2, "crate", "#c82828", "static"),
                           _obj(3, "drum", {"base_color": "#3250d2"}, "spin"), _obj(4, "box", "wood", "trajectory")]}
    text = describe_scene(program)
    assert "two red crates" in text and "Of the two red crates" in text
    assert "blue drum spins in place" in text and "wooden box travels" in text
    assert not re.search(r"\d", text.replace("four objects", "")), text     # 不写任何数字参数
    assert "axis" not in text and "[" not in text


def test_模板器_全同运动用both():
    program = {"objects": [_obj(1, "pod", "metal", "spin"), _obj(2, "pod", "metal", "spin")]}
    assert "Both metal pods spin in place" in describe_scene(program)


def test_掩码配对_挑最大重叠且不重复():
    a = np.zeros((10, 10), bool); a[0:5, 0:5] = True
    b = np.zeros((10, 10), bool); b[5:10, 5:10] = True
    p1 = np.zeros((10, 10), bool); p1[0:5, 0:4] = True          # 和 a 重叠 0.8
    p2 = np.zeros((10, 10), bool); p2[4:10, 4:10] = True         # 和 b 重叠大，和 a 只碰一个像素
    match = match_masks({"a": [a], "b": [b]}, {"p1": [p1], "p2": [p2]})
    assert match["a"][0] == "p1" and abs(match["a"][1] - 0.8) < 1e-6
    assert match["b"][0] == "p2" and match["b"][1] > 0.6
    assert match_masks({"a": [a]}, {"p": [np.zeros((10, 10), bool)]})["a"] == (None, 0.0)


def test_掩码配对_只在末帧入画的物体也配得上():
    empty = np.zeros((10, 10), bool)
    late = np.zeros((10, 10), bool); late[2:6, 2:6] = True
    match = match_masks({"g": [empty, late]}, {"p": [empty, late.copy()]})
    assert match["g"][0] == "p"


def test_画面外的物体不进分母_真值原样送进来是满分(tmp_path, monkeypatch):
    from scripts.statediff import evaluate
    seen = np.zeros((4, 4), bool); seen[1:3, 1:3] = True
    gone = np.zeros((4, 4), bool)
    program = {"meta": {"clip": "c"}, "static": [], "objects": [
        {"id": "inside", "class": "box", "geom": {"kind": "primitive", "shape": "box", "extent": [1, 1, 1]},
         "pose": {"pos": [0, 0.5, 0]}, "motion": {"type": "static"}},
        {"id": "outside", "class": "box", "geom": {"kind": "primitive", "shape": "box", "extent": [1, 1, 1]},
         "pose": {"pos": [9, 0.5, 0]}, "motion": {"type": "static"}}]}
    path = tmp_path / "program.json"; path.write_text(__import__("json").dumps(program))
    masks = {"inside": seen, "outside": gone}
    monkeypatch.setattr(evaluate, "object_masks", lambda p, _: {o["id"]: masks[o["id"]] for o in p["objects"]})
    monkeypatch.setattr(evaluate, "render_endpoints", lambda *a: {"ok": True, "files": {0.0: {"id": "x"}, 8.0: {"id": "y"}}})
    s = {"id": "s", "set": "h", "duration": 8, "program": str(path), "id0": "a", "id1": "b"}
    row = evaluate.eval_one(("oracle_gt", s, evaluate.target_objects(program), str(tmp_path / "work")))
    assert (row["iou_t0"], row["iou_tT"], row["recall50"], row["prec50"], row["motion_acc"]) == (1.0, 1.0, 1.0, 1.0, 1.0)
