"""让一个无头的编码智能体写场景：准备工作区、调起 `claude -p`、收回产物。

    python -m gwm.bench.agent prepare --config configs/bench/room.yaml --workspace DIR
    python -m gwm.bench.agent run     --config configs/bench/room.yaml --workspace DIR
    python -m gwm.bench.agent sidebyside --workspace DIR [--names a,b]

工作区里只有场景包的 `agent/` 那一半（训练视角的照片、相机、稀疏点）、物体名单、
任务说明和两个自检工具。留出视角的照片和交互清单不放进去。

要说清楚的一点：这里没有沙箱。智能体被允许运行 node，原则上能读到磁盘上别的文件，
包括留出视角的照片。demo 阶段靠的是「不告诉它在哪」加上事后查运行记录，
正式做 benchmark 时这一步要放进容器里。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from PIL import Image

from .run import REPO, load_config, scene_out

PROMPT = REPO / "prompts" / "bench_scene_writer.md"
# 允许的工具：读写文件，加上两个自检脚本和 node。不给联网。
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep",
                 "Bash(./tools/preview.sh:*)", "Bash(./tools/physics.sh:*)", "Bash(node:*)", "Bash(ls:*)"]
DISALLOWED_TOOLS = ["WebSearch", "WebFetch"]


def _script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# 给不给物体名单，提示词里有四处不一样。不给名单时，场景里有哪些物体、每个物体拆成多细都由
# 智能体自己定，评测按位置把清单对到它写的物体上（configs/bench/*.anchors.json）。
WITH_MANIFEST = {
    "MANIFEST_INPUT": "inputs/manifest.json    物体名单：场景里必须出现这些 id\n",
    "ID_RULE": "- `id`：名单里的物体必须用名单给的 id。名单之外的物体可以自己起名，不要重复。",
    "MANIFEST_DONE": "- 名单里的每个 id 都在场景里。",
}
WITHOUT_MANIFEST = {
    "MANIFEST_INPUT": "",
    "ID_RULE": ("- `id`：自己起名，不要重复。评测不看名字，按位置找物体：清单里每一条对应房间里的一个位置，"
                "评测时取占着那个位置的物体。所以现实里能单独拿起、推开、打开的东西要各自写成一个物体；"
                "现实里不会分开的部件放进同一个物体。"),
    "MANIFEST_DONE": "- 照片里认得出来的家具和物件都在场景里，现实里能分开动的东西各自是一个物体。",
    "MANIFEST_SECTION": "## 物体\n\n这次没有物体名单。房间里有哪些物体、每个物体拆成多细，由你对着照片自己决定。",
}


def prepare(cfg: dict, workspace: Path, *, manifest: bool = True) -> Path:
    pack_agent = scene_out(cfg) / "pack" / "agent"
    if not pack_agent.is_dir():
        raise FileNotFoundError(f"没有场景包，先跑 python -m gwm.bench.run pack：{pack_agent}")
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"工作区已经有内容，换一个目录，免得盖掉上一次的产物：{workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pack_agent, workspace / "inputs")
    if manifest:
        objects = json.loads((REPO / cfg["manifest"]).read_text(encoding="utf-8"))["objects"]
        (workspace / "inputs" / "manifest.json").write_text(
            json.dumps(objects, ensure_ascii=False, indent=1), encoding="utf-8")
        listing = "\n".join(f"- `{o['id']}`：{o['what']}" for o in objects)
        fill = {**WITH_MANIFEST, "MANIFEST_SECTION": f"## 物体名单\n\n{listing}"}
    else:
        fill = WITHOUT_MANIFEST
    (workspace / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")

    views = json.loads((workspace / "inputs" / "cameras.json").read_text(encoding="utf-8"))["views"]
    task = PROMPT.read_text(encoding="utf-8").replace("{{VIEW_COUNT}}", str(len(views)))
    for key, text in fill.items():
        task = task.replace("{{" + key + "}}", text)
    (workspace / "TASK.md").write_text(task, encoding="utf-8")

    count = int(cfg["agent"].get("preview_views", 6))
    step = max(1, len(views) // count)
    default_names = ",".join(Path(v["name"]).stem for v in views[::step][:count])
    tools = workspace / "tools"
    tools.mkdir()
    python = REPO / "venv" / "bin" / "python"
    _script(tools / "preview.sh", f"""#!/bin/bash
# 渲染当前场景（纯色模式），和训练视角的照片并排，输出到 preview/。
set -euo pipefail
cd "$(dirname "$0")/.."
NAMES="${{1:-{default_names}}}"
rm -rf preview/render
node "{REPO}/harness/bench_render.mjs" --cand . --cameras inputs/cameras.json --out preview/render --mode flat --names "$NAMES"
PYTHONPATH="{REPO}" "{python}" -m gwm.bench.agent sidebyside --workspace . --names "$NAMES"
""")
    _script(tools / "physics.sh", f"""#!/bin/bash
# 把场景里每个声明成能动的物体抓一遍、放一遍，报告执行结果。
set -euo pipefail
cd "$(dirname "$0")/.."
node "{REPO}/harness/bench_interact.mjs" --cand . --selfcheck --out preview/physics.json
""")
    return workspace


def sidebyside(workspace: Path, names: list[str] | None) -> list[Path]:
    """左边照片、右边渲染，拼成一张图，方便用一次 Read 看完。"""
    views = json.loads((workspace / "inputs" / "cameras.json").read_text(encoding="utf-8"))["views"]
    out = []
    for view in views:
        stem = Path(view["name"]).stem
        if names and stem not in names:
            continue
        render = workspace / "preview" / "render" / f"{stem}.png"
        if not render.is_file():
            continue
        photo = Image.open(workspace / "inputs" / view["file"]).convert("RGB")
        shot = Image.open(render).convert("RGB").resize(photo.size, Image.LANCZOS)
        sheet = Image.new("RGB", (photo.width * 2, photo.height))
        sheet.paste(photo, (0, 0))
        sheet.paste(shot, (photo.width, 0))
        sheet.thumbnail((1600, 1600))
        path = workspace / "preview" / f"{stem}.jpg"
        sheet.save(path, quality=85)
        out.append(path)
    return out


def run(cfg: dict, workspace: Path) -> dict:
    """调起 claude -p。花费上限、模型都从配置里读；原始输出存成 agent_run.json。"""
    if not (workspace / "TASK.md").is_file():
        raise FileNotFoundError(f"工作区还没准备好，先跑 prepare：{workspace}")
    agent = cfg["agent"]
    cmd = ["claude", "-p", "按 TASK.md 完成任务。先读 TASK.md。",
           "--model", agent["model"], "--output-format", "json",
           "--permission-mode", "acceptEdits",
           "--max-budget-usd", str(agent["max_budget_usd"]),
           "--allowedTools", *ALLOWED_TOOLS, "--disallowedTools", *DISALLOWED_TOOLS]
    started = _dt.datetime.now()
    done = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
    record: dict = {
        "command": cmd, "started": started.isoformat(timespec="seconds"),
        "seconds": round((_dt.datetime.now() - started).total_seconds(), 1),
        "returncode": done.returncode,
        "cli_version": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip(),
    }
    try:
        record["result"] = json.loads(done.stdout)
    except json.JSONDecodeError:
        record["stdout_tail"] = done.stdout[-2000:]
    if done.stderr.strip():
        record["stderr_tail"] = done.stderr[-2000:]
    (workspace / "agent_run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(f"claude -p 退出码 {done.returncode}：{(done.stderr or done.stdout)[-500:]}")
    if not (workspace / "scene.js").is_file():
        raise RuntimeError("智能体结束了但没有写出 scene.js，详见 agent_run.json")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "sidebyside"):
        p = sub.add_parser(name)
        p.add_argument("--workspace", required=True)
        if name != "sidebyside":
            p.add_argument("--config", required=True)
        if name == "prepare":
            p.add_argument("--no-manifest", action="store_true", help="不给智能体物体名单，粒度由它自己定")
        else:
            p.add_argument("--names", default=None)
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    if args.command == "sidebyside":
        for path in sidebyside(workspace, args.names.split(",") if args.names else None):
            print(path)
    elif args.command == "prepare":
        print(prepare(load_config(args.config), workspace, manifest=not args.no_manifest))
    else:
        record = run(load_config(args.config), workspace)
        result = record.get("result", {})
        print(f"done in {record['seconds']} s, cost ${result.get('total_cost_usd')}, turns {result.get('num_turns')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
