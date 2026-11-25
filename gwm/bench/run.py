"""两轴评测 demo 的命令行入口。

    python -m gwm.bench.run pack   --config configs/bench/room.yaml
    python -m gwm.bench.run eval   --config configs/bench/room.yaml --cand DIR --label NAME [--modes flat,textured]
    python -m gwm.bench.run report --config configs/bench/room.yaml [--runs ID,ID]

pack    把数据集整理成场景包（坐标归一化、训练/留出划分），只需要跑一次。
eval    对一个候选场景（scene.js + materials.json）算两轴：留出视角的保真度、交互清单的通过率。
        同一份场景代码按 flat（纯色）和 textured（生成贴图）各渲染一遍，得到两个点，
        两个点的交互分数必然相同——交互只看几何，这是设计，不是巧合。
report  把若干次 eval 的结果拼成一个静态网页。

每次 eval 的产物在 out/bench/<scene>/runs/<run_id>/ 下面，永不覆盖；run.json 记下
git 提交、配置快照和时间，候选场景原样拷一份进去，之后任何一个数都能追到它是怎么来的。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MODES = ("flat", "textured")
CANDIDATE_FILES = ("scene.js", "materials.json")


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    return cfg


def scene_out(cfg: dict) -> Path:
    return REPO / cfg["paths"]["out"] / cfg["scene"]


def _git_commit() -> str | None:
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    if done.returncode != 0:
        return None
    return done.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def _node(script: str, *args: str) -> None:
    done = subprocess.run(["node", str(REPO / "harness" / script), *args], cwd=REPO, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{script} 失败：\n{(done.stderr or done.stdout)[-2000:]}")
    if done.stdout.strip():
        print(done.stdout.strip())


def code_size(scene_js: Path) -> dict:
    """代码量：去掉空行和整行注释之后的行数，以及字节数。"""
    text = scene_js.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("//")]
    return {"lines": len(lines), "bytes": len(text.encode("utf-8"))}


def cmd_pack(cfg: dict) -> dict:
    from .data import build_pack
    d = cfg["data"]
    frame = build_pack(REPO / d["root"], scene_out(cfg) / "pack", image_dir=d["image_dir"],
                       camera_height_m=d["camera_height_m"], agent_views=d["agent_views"],
                       agent_view_width=d["agent_view_width"], max_points=d["max_points"], seed=d.get("seed", 0))
    print(json.dumps(frame, ensure_ascii=False, indent=1))
    return frame


def cmd_eval(cfg: dict, cand: Path, label: str, modes: list[str], *, skip_lpips: bool = False) -> Path:
    if not re.fullmatch(r"[a-z0-9_-]+", label):
        raise ValueError("label 只能用小写字母、数字、下划线和连字符")
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"不认识的模式 {mode}")
    pack = scene_out(cfg) / "pack"
    cameras_file = pack / "eval" / "cameras.json"
    if not cameras_file.is_file():
        raise FileNotFoundError(f"没有场景包，先跑 pack：{cameras_file}")
    if not (cand / "scene.js").is_file():
        raise FileNotFoundError(f"候选目录里没有 scene.js：{cand}")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = scene_out(cfg) / "runs" / f"{stamp}-{label}"
    cand_dir = run_dir / "candidate"
    cand_dir.mkdir(parents=True)
    for name in CANDIDATE_FILES:
        if (cand / name).is_file():
            shutil.copy2(cand / name, cand_dir / name)
    (cand_dir / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
    materials = json.loads((cand_dir / "materials.json").read_text(encoding="utf-8")) if (cand_dir / "materials.json").is_file() else {}
    run_meta = {
        "run_id": run_dir.name, "label": label, "scene": cfg["scene"], "modes": modes,
        "source_candidate": str(cand.resolve()), "git_commit": _git_commit(),
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "agent": json.loads((cand / "agent_run.json").read_text(encoding="utf-8")) if (cand / "agent_run.json").is_file() else None,
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=1), encoding="utf-8")

    views = json.loads(cameras_file.read_text(encoding="utf-8"))["views"]
    weights = REPO / cfg["paths"]["weights"]
    lpips_fn = None
    if not skip_lpips:
        from .fidelity import Lpips
        lpips_fn = Lpips(weights, net=cfg["fidelity"]["lpips_net"])
    from . import fidelity

    texture_log = None
    points = []
    for mode in modes:
        if mode == "textured":
            if not materials:
                raise ValueError("textured 模式需要 materials.json")
            from .appearance import bind_textures
            from .texture import DEFAULT_MODEL, generate_textures
            # 生成的原图留在 textures_raw/；渲染用的是绑过底色的那一份（见 appearance.py）
            raw_dir, tex_dir = cand_dir / "textures_raw", cand_dir / "textures"
            texture_log = generate_textures(materials, raw_dir,
                                            model=cfg["texture"].get("model") or DEFAULT_MODEL,
                                            size=cfg["texture"].get("size", 512))
            if cfg["texture"].get("bind_base_color", True):
                strength = float(cfg["texture"].get("detail_strength", 1.0))
                texture_log["colour_bound"] = bind_textures(materials, raw_dir, tex_dir, strength)
                texture_log["detail_strength"] = strength
            else:
                shutil.copytree(raw_dir, tex_dir, dirs_exist_ok=True)
            (tex_dir / "textures.json").write_text(json.dumps(texture_log, ensure_ascii=False, indent=1), encoding="utf-8")
        render_dir = run_dir / "renders" / mode
        _node("bench_render.mjs", "--cand", str(cand_dir), "--cameras", str(cameras_file),
              "--out", str(render_dir), "--mode", mode)
        result = fidelity.evaluate(pack / "eval" / "gt", render_dir, views, lpips_fn)
        fidelity.write(result, run_dir / f"fidelity_{mode}.json")
        points.append({"name": f"{label}-{mode}", "label": label, "mode": mode, "fidelity": result["mean"]})

    interact_file = run_dir / "interact.json"
    anchors = ["--anchors", str(REPO / cfg["anchors"])] if cfg.get("anchors") else []
    _node("bench_interact.mjs", "--cand", str(cand_dir), "--checklist", str(REPO / cfg["checklist"]),
          "--out", str(interact_file), *anchors)
    interact = json.loads(interact_file.read_text(encoding="utf-8"))
    # 连续交互：同一个世界里按顺序做一串动作。两种条件都跑，见 harness/bench_sequence.mjs
    sequence = {}
    if cfg.get("sequence"):
        for condition, flags in (("lazy", []), ("settled", ["--settle"])):
            out = run_dir / f"sequence_{condition}.json"
            _node("bench_sequence.mjs", "--cand", str(cand_dir), "--sequence", str(REPO / cfg["sequence"]),
                  "--out", str(out), *anchors, *flags)
            sequence[condition] = json.loads(out.read_text(encoding="utf-8"))["summary"]
    for point in points:
        point["interact"] = interact["summary"]
        point["sequence"] = sequence
    results = {
        "run_id": run_dir.name, "label": label, "scene": cfg["scene"],
        "held_out_views": len(views), "image_dir": cfg["data"]["image_dir"],
        "code_size": code_size(cand_dir / "scene.js"), "materials": len(materials),
        "objects": interact["objects"], "points": points,
        "texture_model": (texture_log or {}).get("model"),
        "finished": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for point in points:
        seq = " ".join(f"sequence_{k}={v['passed']}/{v['total']}" for k, v in sequence.items())
        print(f"{point['name']}: fidelity={point['fidelity']} interact={interact['summary']['passed']}/{interact['summary']['total']} {seq}")
    print(f"run dir: {run_dir}")
    return run_dir


def cmd_report(cfg: dict, run_ids: list[str] | None) -> Path:
    from .report import build_report
    runs_dir = scene_out(cfg) / "runs"
    if run_ids:
        run_dirs = [runs_dir / r for r in run_ids]
    else:
        # 没指定就取每个 label 最新的一次
        latest: dict[str, Path] = {}
        for d in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
            if (d / "results.json").is_file():
                latest[json.loads((d / "results.json").read_text(encoding="utf-8"))["label"]] = d
        run_dirs = list(latest.values())
    if not run_dirs:
        raise FileNotFoundError("没有可以汇总的运行结果，先跑 eval")
    out = build_report(cfg, scene_out(cfg) / "pack", run_dirs, scene_out(cfg) / "report")
    print(f"report: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pack", "eval", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "eval":
            p.add_argument("--cand", required=True)
            p.add_argument("--label", required=True)
            p.add_argument("--modes", default="flat,textured")
            p.add_argument("--skip-lpips", action="store_true", help="只在调试管线时用：不加载 LPIPS 权重")
        if name == "report":
            p.add_argument("--runs", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "pack":
        cmd_pack(cfg)
    elif args.command == "eval":
        cmd_eval(cfg, Path(args.cand), args.label, args.modes.split(","), skip_lpips=args.skip_lpips)
    else:
        cmd_report(cfg, args.runs.split(",") if args.runs else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
