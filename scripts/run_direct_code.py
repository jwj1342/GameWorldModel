#!/usr/bin/env python3
"""让模型直接写 three.js 代码，不经过中间的 JSON 表示（issue #11）。

论文里这是消融表最重要的一行，回答「为什么要中间那层 DSL」。现在支撑这个
设计选择的只有相关工作里别人的数字，自己一个实验都没有。

公平性是这个实验的全部价值所在，所以两条路径的条件必须一致：同样的关键帧
和证据摘要、同样的模型和温度、同样的运行时（three.js、物理、渲染通道、
玩法模板、相机）。唯一的差别是模型输出的形态——声明式的 JSON，还是命令式的 JS。

管道部分（物体编号、ID 颜色、碰撞体）两边都由宿主补齐，不让模型写：
DSL 那边这些是编译器生成的，让直接出代码这边自己写会把对比变得不公平。
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

from gwm.config import REPO, load_config, site_name

REQUIRED = "export function describe"


def strip_fences(text: str) -> str:
    """模型有时还是会套 markdown 围栏，去掉。"""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def build_game_dir(code: str, program_stub: dict, out: Path) -> None:
    """复用 DSL 路径那套打包，再把 scene.js 换成走模型代码的宿主。"""
    from gwm.compiler.compile import compile_program
    r = compile_program(program_stub, out, load_config([REPO / "configs/default.yaml",
                                                        REPO / f"configs/{site_name()}.yaml"]))
    if not r["ok"]: raise RuntimeError(f"打包失败：{r}")
    kernel = out / "kernel"
    (kernel / "model_scene.js").write_text(code, encoding="utf-8")
    # main.js 从 scene.js 取 buildScene，所以这里把 scene.js 改成转发给宿主，
    # 其余内核文件一个字不动。
    orig = kernel / "scene.js"
    orig.rename(kernel / "scene_dsl.js")
    orig.write_text(
        "// 直接出代码消融：buildScene 转给宿主，其余导出仍然来自原文件。\n"
        "export * from './scene_dsl.js';\n"
        "import { buildSceneFromCode } from './scene_from_code.js';\n"
        "export const buildScene = buildSceneFromCode;\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True, help="感知产出的 evidence.json")
    ap.add_argument("--keyframes", required=True, help="关键帧目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--attempts", type=int, default=3, help="语法不过时重试几次")
    a = ap.parse_args(argv)

    cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml",
                       *[REPO / c for c in a.config]])
    from gwm.synthesis.vlm import make_client
    from gwm.synthesis.writer import evidence_summary
    client = make_client(cfg)

    ev = json.loads(Path(a.evidence).read_text())
    frames = sorted(Path(a.keyframes).glob("*.jpg"))[: cfg["perception"].get("n_keyframes", 8)]
    system = (REPO / "prompts" / "writer_direct_code.md").read_text()
    user = f"证据摘要（和 DSL 那条路径给的是同一份）：\n{evidence_summary(ev)}"

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stub = {"meta": {"clip": ev.get("clip", "clip"), "fps": 30,
                     "duration": float(ev.get("duration") or 8.0), "units": "m", "up": "y"},
            "style": {"background": "#dddde3"},
            "camera": {"intrinsics": {"fov_deg": 50, "aspect": 1.7778, "far": 80},
                       "keyframes": [{"t": 0.0, "pos": [0, 2, 6], "quat": [0, 0, 0, 1]}], "interp": "linear"},
            "static": [], "objects": [],
            "binding": {"template": "platformer_3p", "slots": {"walkable": "auto"}}}

    t0, rows = time.time(), []
    for attempt in range(1, a.attempts + 1):
        r = client.chat(system, user, images=[str(f) for f in frames], temperature=0.0)
        code = strip_fences(r.get("text", ""))
        row = {"attempt": attempt, "chars": len(code), "has_describe": REQUIRED in code}
        if REQUIRED not in code:
            row["error"] = "没有导出 describe"; rows.append(row); continue
        (out / "model_scene.js").write_text(code, encoding="utf-8")
        syn = subprocess.run(["node", "--check", str(out / "model_scene.js")], capture_output=True, text=True)
        row["syntax_ok"] = syn.returncode == 0
        if syn.returncode != 0:
            row["error"] = syn.stderr.strip()[:300]
            user += f"\n\n上一版有语法错误，请修正后重写：\n{syn.stderr.strip()[:400]}"
            rows.append(row); continue
        build_game_dir(code, stub, out / "game")
        row["game_dir"] = str(out / "game"); rows.append(row)
        break

    report = {"model": client.model, "calls": client.calls, "seconds": round(time.time() - t0, 1),
              "attempts": rows, "ok": bool(rows and rows[-1].get("syntax_ok"))}
    (out / "direct_code_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
