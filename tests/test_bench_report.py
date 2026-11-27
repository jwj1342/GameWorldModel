"""汇总网页：用一个很小的假场景包和两次假运行，查页面写得出来、图都在、不可信文字都转义了。"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from gwm.bench.report import _pick_views, build_report

EVIL = "<script>alert('x')</script>"
VIEWS = [f"V{i:02d}.JPG" for i in range(8)]
CHECKLIST = [("floor", "fixed"), ("door", "hinge"), ("mug_cup", "movable"), ("mug_cup", "rests"), ("drawer", "slide")]


def _png(path: Path, colour, size=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def make_pack(root: Path) -> Path:
    pack = root / "pack"
    views = []
    for i, name in enumerate(VIEWS):
        stem = Path(name).stem
        _png(pack / "eval" / "gt" / f"{stem}.png", (30 * i, 80, 120), size=(1000, 750))
        views.append({"name": name, "file": f"gt/{stem}.png", "width": 1000, "height": 750})
    _dump(pack / "eval" / "cameras.json", {"views": views})
    _dump(pack / "frame.json", {
        "scene": "toy", "image_dir": "images_4", "split": "每 8 张留出 1 张",
        "counts": {"all": 64, "train": 56, "held_out": 8, "shown_to_agent": 4},
        "normalisation": {"scale": 0.41, "camera_height_m_assumed": 1.4,
                          "camera_height_spread_m": [1.2, 1.4, 1.6], "up_vs_camera_up_deg": 4.9}})
    return pack


def make_run(root: Path, run_id: str, label: str, modes: list[str], *, lpips: bool, passed: int,
             textures: bool = False, agent: dict | None = None) -> Path:
    run = root / "runs" / run_id
    items = []
    for i, (target, expect) in enumerate(CHECKLIST):
        ok = i < passed
        items.append({"target": target, "expect": expect, "pass": ok,
                      "reason": None if ok else ("missing_object" if i % 2 else "declared_static"), "detail": None})
    summary = {"passed": passed, "total": len(CHECKLIST), "pass_rate": round(passed / len(CHECKLIST), 3)}
    _dump(run / "interact.json", {"scene": "toy", "summary": summary, "objects": [], "items": items})
    points = []
    for m, mode in enumerate(modes):
        rows, frames = [], []
        for i, name in enumerate(VIEWS):
            stem = Path(name).stem
            _png(run / "renders" / mode / f"{stem}.png", (200, 40 * m + 20 * i, 60))
            frames.append({"name": name, "file": f"{stem}.png"})
            row = {"name": name, "psnr": 10.0 + i, "ssim": 0.4}
            if lpips:
                row["lpips"] = round(0.7 - 0.1 * m - 0.01 * i, 4)
            rows.append(row)
        _dump(run / "renders" / mode / "index.json", {"mode": mode, "frames": frames})
        mean = {"psnr": 13.5, "ssim": 0.4}
        if lpips:
            mean["lpips"] = round(0.665 - 0.1 * m, 4)
        _dump(run / f"fidelity_{mode}.json", {"views": len(rows), "mean": mean, "per_view": rows})
        points.append({"name": f"{label}-{mode}", "label": label, "mode": mode, "fidelity": mean, "interact": summary})
    materials = {"oak_floor": {"prompt": "oak planks", "seed": 7}, "evil_wall": {"prompt": EVIL, "seed": 11}}
    _dump(run / "candidate" / "materials.json", materials)
    (run / "candidate" / "scene.js").write_text("export function build() {}\n", encoding="utf-8")
    if textures:
        records = []
        for name, entry in materials.items():
            _png(run / "candidate" / "textures" / f"{name}.png", (150, 110, 70), size=(512, 512))
            records.append({"name": name, "file": f"{name}.png", **entry})
        _dump(run / "candidate" / "textures" / "textures.json", {"model": "fake/texture-model", "textures": records})
    _dump(run / "results.json", {
        "run_id": run_id, "label": label, "scene": "toy", "held_out_views": len(VIEWS), "image_dir": "images_4",
        "code_size": {"lines": 123, "bytes": 4567}, "materials": len(materials), "objects": [], "points": points,
        "texture_model": "fake/texture-model" if textures else None})
    _dump(run / "run.json", {"run_id": run_id, "label": label, "scene": "toy", "modes": modes,
                             "git_commit": "abc123+dirty", "started": "2026-11-25T10:00:00", "agent": agent})
    return run


def make_cfg(root: Path, references: list | None = None) -> dict:
    checklist = root / "checklist.json"
    _dump(checklist, {"scene": "toy", "note": "清单是对着照片写的 <b>待复核</b>", "annotated_by": "初版待人工复核",
                      "items": [{"target": t, "expect": e} for t, e in CHECKLIST]})
    # 绝对路径和仓库根目录拼出来还是它自己，所以测试里可以指到 tmp_path 下面
    return {"scene": "toy", "data": {"image_dir": "images_4"}, "checklist": str(checklist),
            "references": references or []}


def test_report_with_two_runs_and_a_reference(tmp_path):
    pack = make_pack(tmp_path)
    alpha = make_run(tmp_path, "20261125-100000-alpha", "alpha", ["flat", "textured"], lpips=True, passed=3,
                     textures=True, agent={"command": ["claude", "-p", "x", "--model", "some-model"], "seconds": 600,
                                           "result": {"total_cost_usd": 3.21, "num_turns": 42}})
    beta = make_run(tmp_path, "20261125-110000-beta<img src=x>", "beta", ["flat"], lpips=False, passed=1)
    cfg = make_cfg(tmp_path, [
        {"name": "3DGS-ref", "fidelity": {"psnr": 31.0, "ssim": 0.92, "lpips": 0.2}, "interact_pass_rate": 0.0,
         "note": "论文里的数 <i>不是本机算的</i>", "source": "https://example.org/paper"},
        {"name": "odd-ref", "fidelity": {"psnr": 20.0}, "interact_pass_rate": 0.0,
         "note": "只有 PSNR", "source": "javascript:alert(1)"}])
    out = tmp_path / "report"

    index = build_report(cfg, pack, [alpha, beta], out)

    assert index == out / "index.html" and index.is_file()
    page = index.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and 'lang="zh-CN"' in page

    # 每个点、每条清单目标都出现在页面上
    for name in ("alpha-flat", "alpha-textured", "beta-flat", "3DGS-ref", "odd-ref"):
        assert name in page
    for target, expect in CHECKLIST:
        assert f"<code>{target}</code>" in page and f"<code>{expect}</code>" in page
    for reason in ("missing_object", "declared_static"):
        assert reason in page
    assert "待人工复核" in page and "引用" in page and "非本机测得" in page

    # 不可信的字符串一律转义
    assert "<script>alert" not in page and "&lt;script&gt;alert" in page
    assert "<img src=x>" not in page and "beta&lt;img src=x&gt;" in page
    assert "<b>待复核</b>" not in page and "<i>不是本机算的</i>" not in page
    assert 'href="https://example.org/paper"' in page
    assert 'href="javascript:' not in page

    # 引用的图都在，都是缩过的 JPEG；页面不引用任何外部资源
    sources = re.findall(r'<img src="([^"]+)"', page)
    assert len(sources) == 6 * (1 + 3) + 2          # 6 个视角 ×（照片 + 三个点），外加两张贴图
    for src in set(sources):
        assert src.startswith("img/") and src.endswith(".jpg")
        with Image.open(out / src) as image:
            assert image.format == "JPEG" and image.width <= 640
    assert "<script src" not in page and "<link" not in page and "url(" not in page and "@import" not in page

    # 没有 LPIPS 的点不画进图里，但要点名；表里写「未计算」
    assert page.count('class="hit"') == 2 * 3       # 宽、窄两张图，各画 alpha 的两个点和一个引用点
    assert "未计算" in page and "没有画进图里：beta-flat" in page
    assert page.count('class="mk-cited"') >= 2 and "mk-textured" in page
    # 贴图和来历
    assert "fake/texture-model" in page and "oak planks" in page and "种子 7" in page
    assert "abc123+dirty" in page and "some-model" in page and "$3.21" in page and "42 轮" in page
    assert "尺度是先验" in page


def test_report_with_a_single_flat_run_and_nothing_else(tmp_path):
    pack = make_pack(tmp_path)
    run = make_run(tmp_path, "20261125-120000-solo", "solo", ["flat"], lpips=True, passed=0)
    cfg = make_cfg(tmp_path)
    cfg["checklist"] = str(tmp_path / "no-such-checklist.json")    # 清单文件读不到时退回到 interact.json 里的条目
    index = build_report(cfg, pack, [run], tmp_path / "out")
    page = index.read_text(encoding="utf-8")
    assert "solo-flat" in page and page.count('class="hit"') == 2
    assert "<code>mug_cup</code>" in page
    assert 'id="textures"' not in page and 'class="mk-cited"' not in page
    assert len(list((tmp_path / "out" / "img").glob("*.jpg"))) == 6 * 2


def test_report_without_any_lpips_still_writes_a_page(tmp_path):
    pack = make_pack(tmp_path)
    run = make_run(tmp_path, "20261125-130000-nolp", "nolp", ["flat"], lpips=False, passed=2)
    page = build_report(make_cfg(tmp_path), pack, [run], tmp_path / "out").read_text(encoding="utf-8")
    assert "<svg class=\"chart" not in page and "这张图画不出来" in page and "nolp-flat" in page


def test_pick_views_is_evenly_spaced_and_keeps_both_ends():
    views = [{"name": str(i)} for i in range(39)]
    picked = [int(v["name"]) for v in _pick_views(views, 6)]
    assert picked[0] == 0 and picked[-1] == 38 and len(picked) == 6
    assert picked == sorted(picked) and max(b - a for a, b in zip(picked, picked[1:])) <= 8
    assert _pick_views(views[:4], 6) == views[:4]
