"""两轴评测 demo 的命令行入口。

    python -m gwm.bench.run pack   --config configs/bench/room.yaml
    python -m gwm.bench.run eval   --config configs/bench/room.yaml --cand DIR --label NAME [--modes flat,textured]
    python -m gwm.bench.run trajectory --config configs/bench/room.yaml --workspace DIR --label NAME
    python -m gwm.bench.run interact --config configs/bench/room.yaml --cand DIR --match id|anchors   （只重算交互，不渲染）
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
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MODES = ("flat", "textured")


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
                       camera_height_m=d["camera_height_m"], floor_band=d["floor_band"], agent_views=d["agent_views"],
                       agent_view_width=d["agent_view_width"], max_points=d["max_points"], seed=d.get("seed", 0))
    print(json.dumps(frame, ensure_ascii=False, indent=1))
    return frame


def score_interaction(cfg: dict, cand_dir: Path, out_dir: Path, match: str) -> tuple[dict, dict]:
    """交互清单和连续交互，写进 out_dir 的 interact.json 和 sequence_*.json。

    match 是 'id' 或 'anchors'。给过物体名单的运行按 id 找物体：锚点取自第一次运行的场景，拿它来判
    别的运行，等于把「东西摆得离那份场景多近」算进交互分。锚点命中了几个单独记在 interact.json 的
    anchors 里，只当摆放位置的诊断。没给名单的运行 id 是智能体自己起的，只能按锚点找。"""
    anchors = ["--anchors", str(REPO / cfg["anchors"])] if cfg.get("anchors") else []
    interact_file = out_dir / "interact.json"
    _node("bench_interact.mjs", "--cand", str(cand_dir), "--checklist", str(REPO / cfg["checklist"]),
          "--out", str(interact_file), "--match", match, *anchors)
    interact = json.loads(interact_file.read_text(encoding="utf-8"))
    # 连续交互：同一个世界里按顺序做一串动作。两种条件都跑，见 harness/bench_sequence.mjs
    sequence = {}
    if cfg.get("sequence"):
        for condition, flags in (("lazy", []), ("settled", ["--settle"])):
            out = out_dir / f"sequence_{condition}.json"
            _node("bench_sequence.mjs", "--cand", str(cand_dir), "--sequence", str(REPO / cfg["sequence"]),
                  "--out", str(out), "--match", match, *anchors, *flags)
            sequence[condition] = json.loads(out.read_text(encoding="utf-8"))["summary"]
    return interact, sequence


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
    # scene.js 可以从同目录别的 .js 里导入（比如把一份的几何和另一份的灯光拼起来），所以 .js 都带上
    for path in [*sorted(cand.glob("*.js")), cand / "materials.json"]:
        if path.is_file():
            shutil.copy2(path, cand_dir / path.name)
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

    # 几何：留出视角的深度图和 COLMAP 在这些照片里实际看到的点比（geometry.py）
    from . import geometry
    depth_dir = run_dir / "renders" / "depth"
    _node("bench_render.mjs", "--cand", str(cand_dir), "--cameras", str(cameras_file), "--out", str(depth_dir), "--mode", "depth")
    geo = geometry.evaluate(pack / "eval" / "observed.json", depth_dir, views)
    (run_dir / "geometry.json").write_text(json.dumps(geo, indent=1), encoding="utf-8")

    match = "id" if (cand / "inputs" / "manifest.json").is_file() else "anchors"
    interact, sequence = score_interaction(cfg, cand_dir, run_dir, match)
    for point in points:
        point["interact"] = interact["summary"]
        point["sequence"] = sequence
    results = {
        "run_id": run_dir.name, "label": label, "scene": cfg["scene"],
        "held_out_views": len(views), "image_dir": cfg["data"]["image_dir"],
        "code_size": code_size(cand_dir / "scene.js"), "materials": len(materials),
        "objects": interact["objects"], "points": points, "geometry": geo["mean"],
        "match": match, "anchors_found": interact["anchors"] and {k: interact["anchors"][k] for k in ("found", "total")},
        "texture_model": (texture_log or {}).get("model"),
        "finished": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for point in points:
        seq = " ".join(f"sequence_{k}={v['passed']}/{v['total']}" for k, v in sequence.items())
        print(f"{point['name']}: fidelity={point['fidelity']} geometry={geo['mean']} interact={interact['summary']['passed']}/{interact['summary']['total']} {seq}")
    print(f"run dir: {run_dir}")
    return run_dir


def cmd_trajectory(cfg: dict, workspace: Path, label: str) -> Path:
    """把智能体每一轮预览时存下的快照都评一遍：留出视角和训练视角的保真度、几何、代码量。

    两件事要靠它回答。一是代码写到多少行以后保真度不再涨，也就是代码该把细节交给生成器的那个点。
    二是智能体对着训练视角越改越好的时候，留出视角是不是也在变好；第一次运行的最终版就不如中途快照。
    只渲染纯色模式，贴图对 LPIPS 的影响只有 0.01，不值得每个快照都生成一遍。"""
    from . import fidelity, geometry
    pack = scene_out(cfg) / "pack"
    held_views = json.loads((pack / "eval" / "cameras.json").read_text(encoding="utf-8"))["views"]
    train_views = json.loads((pack / "agent" / "cameras.json").read_text(encoding="utf-8"))["views"]
    snapshots = sorted(p for p in (workspace / "snapshots").iterdir() if (p / "scene.js").is_file())
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = scene_out(cfg) / "trajectories" / f"{stamp}-{label}"
    # 训练视角的照片在场景包里是 jpg，fidelity.evaluate 按 png 找，先转一份
    train_gt = out / "train_gt"
    train_gt.mkdir(parents=True)
    for view in train_views:
        Image.open(pack / "agent" / view["file"]).convert("RGB").save(train_gt / f"{Path(view['name']).stem}.png")
    lpips_fn = fidelity.Lpips(REPO / cfg["paths"]["weights"], net=cfg["fidelity"]["lpips_net"])
    rows = []
    for snap in [*snapshots, workspace]:
        name = "final" if snap == workspace else snap.name
        cand = out / "candidates" / name
        cand.mkdir(parents=True)
        for file in ("scene.js", "materials.json"):
            shutil.copy2(snap / file, cand / file)
        (cand / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
        renders = out / "renders" / name
        _node("bench_render.mjs", "--cand", str(cand), "--cameras", str(pack / "eval" / "cameras.json"),
              "--out", str(renders / "held_flat"), "--mode", "flat")
        _node("bench_render.mjs", "--cand", str(cand), "--cameras", str(pack / "eval" / "cameras.json"),
              "--out", str(renders / "held_depth"), "--mode", "depth")
        _node("bench_render.mjs", "--cand", str(cand), "--cameras", str(pack / "agent" / "cameras.json"),
              "--out", str(renders / "train_flat"), "--mode", "flat")
        rows.append({
            "snapshot": name, "code": code_size(cand / "scene.js"),
            "held_out": fidelity.evaluate(pack / "eval" / "gt", renders / "held_flat", held_views, lpips_fn)["mean"],
            "train": fidelity.evaluate(train_gt, renders / "train_flat", train_views, lpips_fn)["mean"],
            "geometry": geometry.evaluate(pack / "eval" / "observed.json", renders / "held_depth", held_views)["mean"],
        })
        print(json.dumps(rows[-1]))
    (out / "trajectory.json").write_text(json.dumps({
        "label": label, "workspace": str(workspace.resolve()), "scene": cfg["scene"], "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"trajectory: {out}")
    return out


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
    for name in ("pack", "eval", "trajectory", "interact", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "interact":
            p.add_argument("--cand", required=True, help="目录里要有 scene.js（和 materials.json）；结果写回这个目录")
            p.add_argument("--match", choices=("id", "anchors"), required=True)
        if name == "trajectory":
            p.add_argument("--workspace", required=True)
            p.add_argument("--label", required=True)
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
    elif args.command == "interact":
        interact, sequence = score_interaction(cfg, Path(args.cand), Path(args.cand), args.match)
        a = interact["anchors"]
        print(f"interact ({args.match}) {interact['summary']['passed']}/{interact['summary']['total']}"
              + "".join(f" sequence_{k}={v['passed']}/{v['total']}" for k, v in sequence.items())
              + (f" anchors_found={a['found']}/{a['total']}" if a else ""))
    elif args.command == "trajectory":
        cmd_trajectory(cfg, Path(args.workspace), args.label)
    else:
        cmd_report(cfg, args.runs.split(",") if args.runs else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
