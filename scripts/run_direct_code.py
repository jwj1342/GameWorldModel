#!/usr/bin/env python3
"""让模型直接写 three.js 代码，不经过中间的 JSON 表示（issue #11）。

论文里这是消融表最重要的一行，回答「为什么要中间那层 DSL」。现在支撑这个
设计选择的只有相关工作里别人的数字，自己一个实验都没有。

这是直接代码路径的生成/打包入口，不执行浏览器。共享运行时不代表相机、输入或
调用预算已与DSL严格配平；结果记录实际条件，未验证项不报告成功。

管道部分（物体编号、ID 颜色、碰撞体）两边都由宿主补齐，不让模型写：
DSL 那边这些是编译器生成的，让直接出代码这边自己写会把对比变得不公平。
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path

from gwm.config import REPO, load_config, site_name

REQUIRED = "export function describe"


def declared_video_hash(meta: dict) -> str:
    """Read existing source claims; this does not verify the video bytes."""
    claims = [meta[k] for k in ("source_video_sha256", "video_sha256")
              if meta.get(k) not in (None, "unknown")]
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdefABCDEF" for c in v)
           for v in claims):
        raise ValueError("视频来源SHA-256声明格式非法")
    normalized = {v.lower() for v in claims}
    if len(normalized) > 1:
        raise ValueError("视频来源SHA-256声明相互冲突")
    return next(iter(normalized), "unknown")


def prepare_direct_inputs(ev: dict, directory: Path, *, allow_relocated: bool = False) -> tuple[dict, list[dict], dict]:
    """校验并固定元数据、声明的关键帧和相机；不从目录猜测采样计划。"""
    from gwm.perception.contract import validate_evidence, format_evidence_errors
    from gwm.compiler.validate import validate
    from PIL import Image
    checked = validate_evidence(ev)
    # 某些旧生产者已经带正式frames注册表；适配器重建时不能丢弃这些现有引用。
    # 仅保留确实提供的注册表，并重新走统一验证，不根据图片名制造frame。
    if checked["adapted"] and isinstance(ev.get("frames"), list):
        normalized = checked["evidence"]
        normalized["frames"] = copy.deepcopy(ev["frames"])
        checked = validate_evidence(normalized, adapt_legacy=False)
    if not checked["ok"]:
        raise ValueError("Evidence 不合法：" + format_evidence_errors(checked))
    evidence = checked["evidence"]
    meta = evidence["meta"]
    declared_video_hash(meta)
    duration = meta["duration"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("meta.duration 必须为有限正数，不能默认伪造8秒")
    if not meta["clip"].strip():
        raise ValueError("meta.clip 必须非空")
    declarations = evidence.get("keyframes") or []
    if not declarations:
        raise ValueError("Evidence 未声明关键帧，不能从目录文件名猜测对应关系")
    root = directory.resolve()
    selected, seen, previous = [], set(), -1.0
    for declaration in declarations:
        t = declaration.get("t")
        if type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= duration or t <= previous:
            raise ValueError("关键帧时间须严格递增且位于视频时长内")
        previous = t
        reference = declaration.get("file_small") or declaration.get("file")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("关键帧缺少明确图片引用")
        declared = Path(reference)
        path = declared if declared.is_absolute() else root / declared
        resolution = "declared_path"
        if not declared.is_absolute():
            # 旧生产者存的是相对运行目录路径，新输入也可使用相对图片目录路径。
            # 两种解析均受同一目录边界约束；不能在两个不同文件之间猜测。
            candidates = {}
            for candidate, label in ((path, "declared_path"), (declared, "declared_cwd_path")):
                resolved = candidate.resolve()
                if resolved.is_relative_to(root) and resolved.is_file():
                    candidates.setdefault(resolved, label)
            if len(candidates) > 1:
                raise ValueError("关键帧相对引用有多个合法文件，需明确路径")
            if candidates:
                path, resolution = next(iter(candidates.items()))
        if allow_relocated and (not path.is_file() or not path.resolve().is_relative_to(root)):
            path = root / Path(reference).name
            resolution = "relocated_basename_unverified"
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("关键帧引用不存在或不在指定目录；迁移路径需显式允许")
        if path in seen:
            raise ValueError("关键帧重复引用同一图片")
        seen.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # 原图hash不能用于校验重新编码的file_small。
        claimed = declaration.get("file_small_sha256") if declaration.get("file_small") else declaration.get("image_sha256")
        if claimed not in (None, "unknown") and digest != claimed:
            raise ValueError("关键帧图片与声明SHA-256不匹配")
        with Image.open(path) as image:
            image.verify()
        selected.append({"file": str(path), "t": t,
                         "frame_index": declaration.get("frame_index", declaration.get("index", "unknown")),
                         "sha256": digest, "reference": reference, "resolution": resolution,
                         "declared_hash_verified": claimed not in (None, "unknown")})
    cam = evidence["camera"]
    if not cam.get("poses"):
        raise ValueError("相机位姿缺失；不提供虚构默认相机")
    intrinsics = cam["intrinsics"]
    width, height, fov = intrinsics.get("width"), intrinsics.get("height"), intrinsics.get("fov_deg")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (width, height, fov)):
        raise ValueError("相机width/height/fov_deg需要实有的有限正值")
    stub = {"meta": {"clip": meta["clip"], "fps": 30, "duration": duration, "units": "m", "up": "y"},
            "style": {"background": "#dddde3"},
            "camera": {"intrinsics": {"fov_deg": fov, "aspect": width / height, "far": 80},
                       "keyframes": [{k: copy.deepcopy(p[k]) for k in ("t", "pos", "quat")} for p in cam["poses"]],
                       "interp": "linear"},
            "static": [], "objects": [],
            "binding": {"template": "platformer_3p", "slots": {"walkable": "auto"}}}
    if not validate(stub)["ok"]:
        raise ValueError("Evidence 相机/元数据不能生成合法宿主Program")
    return evidence, selected, stub


def strip_fences(text: str) -> str:
    """模型有时还是会套 markdown 围栏，去掉。"""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def build_game_dir(code: str, program_stub: dict, out: Path, cfg: dict | None = None) -> None:
    """复用 DSL 路径那套打包，再把 scene.js 换成走模型代码的宿主。"""
    from gwm.compiler.compile import compile_program
    effective = cfg if cfg is not None else load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"])
    r = compile_program(program_stub, out, effective)
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
    ap.add_argument("--allow-relocated-keyframes", action="store_true", help="显式允许图片引用迁移到给定目录；无声明哈希时不证明来源对应")
    a = ap.parse_args(argv)
    if a.attempts < 1:
        ap.error("--attempts 必须为正整数")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        ap.error("输出目录非空，请使用新的目录，不覆盖旧实验")
    out.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    report = {"model": None, "calls": 0, "attempts": rows, "ok": None,
              "generation_ok": False, "runtime_ok": None, "runtime_status": "not_checked",
              "status": "input_failed", "diagnostics": [], "provenance": {}}

    def finish(status, error=""):
        report.update(status=status, seconds=round(time.time() - t0, 1))
        if error:
            report["diagnostics"].append({"stage": "direct_code", "severity": "error", "code": status,
                                          "path": "/", "hint": error})
        (out / "direct_code_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        # 退出码仅代表生成/打包流程；runtime_ok/ok仍未知，不冒充浏览器运行通过。
        return 0 if report["generation_ok"] else 1

    try:
        raw = Path(a.evidence).read_bytes()
        ev, frames, stub = prepare_direct_inputs(json.loads(raw), Path(a.keyframes), allow_relocated=a.allow_relocated_keyframes)
        cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml",
                           *[REPO / c for c in a.config]])
        from gwm.synthesis.writer import evidence_summary
        system = (REPO / "prompts" / "writer_direct_code.md").read_text(encoding="utf-8")
        user = ("关键帧按Evidence时间顺序附上，t = " + ", ".join(str(f["t"]) for f in frames)
                + " 秒。\nEVIDENCE:\n" + evidence_summary(ev))
        report["provenance"] = {"evidence_sha256": hashlib.sha256(raw).hexdigest(), "keyframes": frames,
                                "clip": ev["meta"]["clip"], "duration": ev["meta"]["duration"],
                                "video_sha256": declared_video_hash(ev["meta"]),
                                "video_source_verification": "not_checked",
                                "timestamp_basis": "evidence_references_not_redecoded",
                                "camera_policy": "evidence_poses_linear", "camera": stub["camera"],
                                "temperature": 0.0, "max_attempts": a.attempts,
                                "max_tokens": cfg.get("vlm", {}).get("max_tokens"),
                                "seed": cfg.get("vlm", {}).get("seed"),
                                "fairness_status": "not_established"}
    except Exception as e:
        reason = str(e) if isinstance(e, ValueError) else type(e).__name__
        return finish("input_failed", reason)
    try:
        from gwm.synthesis.vlm import make_client
        client = make_client(cfg)
        report["model"] = client.model
    except Exception as e:
        return finish("client_failed", type(e).__name__)

    for attempt in range(1, a.attempts + 1):
        row = {"attempt": attempt, "syntax_ok": None, "packaged": False, "runtime_ok": None,
               "prompt_sha256": hashlib.sha256(json.dumps([system, user], ensure_ascii=False).encode("utf-8")).hexdigest()}
        rows.append(row)
        try:
            r = client.chat(system, user, images=[f["file"] for f in frames], temperature=0.0)
        except Exception as e:
            row.update(status="call_failed", error=type(e).__name__)
            report["calls"] = client.calls
            continue
        report["calls"] = client.calls
        if not isinstance(r, dict) or not isinstance(r.get("text"), str):
            row.update(status="invalid_response", error="缺少代码文本")
            continue
        code = strip_fences(r["text"])
        row.update(chars=len(code), has_describe=REQUIRED in code,
                   code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest())
        if REQUIRED not in code:
            row.update(status="invalid_response", error="没有导出 describe"); continue
        (out / "model_scene.js").write_text(code, encoding="utf-8")
        # node --check 把 .js 当成 CommonJS，碰到 export 会报语法错，
        # 但模型写的本来就是 ES 模块、浏览器里也是 import 加载的。
        # 所以校验时复制成 .mjs，否则每一份有效输出都会被误判成失败。
        probe = out / "model_scene.mjs"; probe.write_text(code, encoding="utf-8")
        try:
            syn = subprocess.run([os.environ.get("GWM_NODE_BINARY") or "node", "--check", str(probe)],
                                 capture_output=True, text=True, encoding="utf-8", timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            row.update(status="syntax_check_failed", error=type(e).__name__)
            continue
        finally:
            probe.unlink(missing_ok=True)
        row["syntax_ok"] = syn.returncode == 0
        if syn.returncode != 0:
            row.update(status="syntax_failed", error=syn.stderr.strip()[:300])
            user += f"\n\n上一版有语法错误，请修正后重写：\n{syn.stderr.strip()[:400]}"
            continue
        try:
            build_game_dir(code, stub, out / "game", cfg=cfg)
        except Exception as e:
            row.update(status="packaging_failed", error=type(e).__name__)
            continue
        row.update(status="packaged_unverified", packaged=True, game_dir=str(out / "game"))
        report["generation_ok"] = True
        break
    return finish("packaged_unverified" if report["generation_ok"] else "generation_failed")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
