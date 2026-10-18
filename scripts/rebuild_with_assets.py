#!/usr/bin/env python
"""一段视频 -> 一个带真网格的可交互场景。

这个脚本把四步串起来，中间不调生成模型，所以跑出来的差别只来自几何和资产这两层：

    视频 -> 感知（证据、掩码、帧）
         -> 直译成场景程序（每个物体一个 asset 节点）
         -> 看画面挑资产，写进 geom.ref
         -> 编译成可运行的 game 目录

挑资产这一步是关键：只看类名的话，"toy train" 会命中 train 这个 kit 里的车厢连接件，
因为名字里都有 train；把视频里那个物体抠出来和资产缩略图比一比才分得出哪个像。

用法:
  python -m scripts.rebuild_with_assets --video data/clips/trimmed/plarail_train.mp4 \
      --clip plarail_train --out $GWM_SCRATCH/rebuild/plarail \
      --phrases "toy train,railroad track,wooden block,table,bookshelf,book,box,cup,chair,bed,lamp,plant"
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.assets import choose_assets                            # noqa: E402
from gwm.assets.embed import AssetIndex, Embedder               # noqa: E402
from gwm.assets.library import library_root, load_catalog       # noqa: E402
from gwm.compiler.compile import compile_program                # noqa: E402
from gwm.compiler.repair import repair_candidate                # noqa: E402
from gwm.feedback.colour import colour_residual, fit_colours, fit_lighting  # noqa: E402
from gwm.feedback.completeness import assess, format_table      # noqa: E402
from gwm.feedback.interaction_probe import measure as probe_interaction, merge as merge_probe  # noqa: E402
from gwm.feedback.likeness import likeness                      # noqa: E402
from gwm.binding.platformer import scale_program                # noqa: E402
from gwm.synthesis.contact import settle_on_ground              # noqa: E402
from gwm.synthesis.dedup import merge_duplicates, merge_fragment_chains   # noqa: E402
from gwm.synthesis.scale import estimate_metric_scale, regularize_object_sizes  # noqa: E402
from gwm.synthesis.stabilise import apply_settled                # noqa: E402
from gwm.config import load_config, site_name                   # noqa: E402
from gwm.perception.run import load_masks, run_perception       # noqa: E402
from gwm.synthesis.direct import evidence_to_program            # noqa: E402


def _settle(game_dir: Path, out: Path, seconds: float, separate_seconds: float = 0.0) -> dict:
    """调 harness 跑一段仿真。跑不起来就当没稳定过，不让它挡住整条链路。"""
    import subprocess
    target = out / "settle.json"
    try:
        subprocess.run([str(REPO / "scripts/node_harness.sh"), "settle.mjs",
                        "--game", str(game_dir), "--out", str(target),
                        "--seconds", str(seconds),
                        "--separate-seconds", str(separate_seconds)],
                       check=True, capture_output=True, timeout=600)
        return json.loads(target.read_text(encoding="utf-8"))
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        print(f"物理稳定跳过: {type(error).__name__}", flush=True)
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrases", default="", help="逗号分隔的检测短语；留空走默认词表")
    ap.add_argument("--max-objects", type=int, default=0, help=">0 时覆盖配置里的物体数上限")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--likeness", action="store_true",
                    help="把重建的场景按输入相机轨迹重渲染，和原视频比相似度")
    ap.add_argument("--likeness-reference", default=None,
                    help="当下限用的那个 game 目录。做消融时必须固定，否则各变体的"
                         "下限不一样，归一化没法比")
    ap.add_argument("--no-scale", action="store_true",
                    help="跳过度量尺度与尺寸回收，用于消融")
    ap.add_argument("--no-settle", action="store_true", help="跳过物理稳定这一步")
    ap.add_argument("--probe-interaction", action="store_true",
                    help="真的去抓一遍、拖一遍，量出可交互程度（而不是只数程序声明了什么）")
    ap.add_argument("--fit-colours", type=int, default=None, metavar="N",
                    help="按闭环残差把材质颜色调 N 轮（#55）。渲出来比视频暗，因为把像素"
                         "当反照率用、渲染时又乘了一遍光照。两轮实测：合成片段每通道误差"
                         "16.8 -> 5.3、亮度比 0.833 -> 0.972；真实视频 27.9 -> 6.2、"
                         "0.627 -> 0.897")
    ap.add_argument("--colour-residual", action="store_true",
                    help="样式那一半的验收指标：渲出来的颜色和视频里量到的差多少。"
                         "重渲染相似度看不见颜色（色相全改错只动 0.007），这一项不能靠它代替")
    ap.add_argument("--style-swap", action="store_true",
                    help="顺带量样式-行为解耦：换一套外形网格，轨迹应该一丝不差")
    ap.add_argument("--behaviour-swap", action="store_true",
                    help="解耦的另一半：改运动和交互，样式应该一个字段都不变")
    ap.add_argument("--settle-seconds", type=float, default=1.5)
    ap.add_argument("--separate-seconds", type=float, default=None,
                    help="重力稳定之前先跑多久的纯分离（不开重力，只把互相插着的顶开）。"
                         "不给就用配置里的 synthesis.separate_seconds")
    ap.add_argument("--no-image-retrieval", action="store_true",
                    help="只用文字检索，用来和带图像检索的结果做对照")
    a = ap.parse_args()

    files = [REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml", *a.config]
    cfg = load_config([f for f in files if Path(f).is_file()])
    if a.max_objects > 0:
        cfg["perception"]["max_objects"] = a.max_objects
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    phrases = [p.strip() for p in a.phrases.split(",") if p.strip()] or None

    t0 = time.time()
    evidence = run_perception(a.video, a.clip, out / "perception", cfg, phrases=phrases)
    timings = {"perception_s": round(time.time() - t0, 1)}
    print(f"感知: {len(evidence['objects'])} 个物体, {timings['perception_s']}s", flush=True)

    program = evidence_to_program(evidence, a.clip, cfg)
    asset_nodes = sum(1 for o in program.get("objects", []) if o.get("geom", {}).get("kind") == "asset")
    print(f"直译: static {len(program.get('static', []))}, objects {len(program.get('objects', []))}"
          f"（其中 {asset_nodes} 个要挑资产）", flush=True)

    root = library_root(cfg)
    decisions = []
    # 在 if 外面初始化。下面汇总那一行是 `bool(index) if root else False`——它只守了
    # root，没守 asset_nodes，于是一条**一个物体都没检出来**的片段（asset_nodes 为 0）
    # 会在所有工作都做完之后、写汇总时崩在 UnboundLocalError 上，产物全丢。
    # 实测十二条片段里有两条是这样（test_00000、test_00011）。
    index = embedder = None
    if root and asset_nodes:
        t1 = time.time()
        catalog = load_catalog(root)
        index_file = Path(root) / "dinov2_index.npz"
        if not a.no_image_retrieval and index_file.is_file():
            index = AssetIndex.load(index_file)
            embedder = Embedder(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])
        frames = json.loads((out / "perception" / "frames" / "frames.json").read_text())["frames"]
        masks = load_masks(out / "perception")
        settings = cfg.get("assets") or {}
        decisions = choose_assets(program, evidence, frames, masks, catalog, index, embedder,
                                  image_weight=float(settings.get("image_weight", 0.6)),
                                  min_score=float(settings.get("min_score", 0.45)))
        timings["retrieval_s"] = round(time.time() - t1, 1)
        for d in decisions:
            mark = "图像+文字" if d["image_used"] else "只有文字"
            print(f"  {d['id']:<18} {d['query']:<18} -> {d.get('chosen') or '（退回基本体）':<28} "
                  f"{mark} 候选={[c['asset_id'] for c in d['candidates'][:3]]}", flush=True)

    # 直译出来的位姿直接取自包围盒中心，地面估歪一点就会有物体陷进地里。
    # 模型那条路本来就会过一遍保守修复，直译这条也该过——它只在证据支持时才动位姿。
    # 度量尺度：现在的锚是「相机离地 1.2 米」，俯拍桌面时会整体放大两三倍。
    # 改用物体尺寸当锚——命名那一步问过的 size_m 就存在 namer.json 里，一直没人用。
    namer = out / "perception" / "namer.json"
    vlm_sizes = {}
    if namer.is_file():
        try:
            for item in (json.loads(namer.read_text(encoding="utf-8")) or {}).get("objects") or []:
                if isinstance(item, dict) and item.get("phrase") and item.get("size_m"):
                    vlm_sizes[str(item["phrase"]).lower()] = item["size_m"]
        except (OSError, ValueError):
            pass
    factor, scale_report = ((1.0, {"factor": 1.0, "samples": [], "source": "skipped"})
                            if a.no_scale else estimate_metric_scale(program, vlm_sizes))
    if abs(factor - 1.0) > 1e-6:
        program = scale_program(program, factor)
        program.setdefault("meta", {})["metric_scale"] = round(factor, 4)
        print(f"度量尺度: x{factor:.3f}（依据 {len(scale_report['samples'])} 个物体，"
              f"来源 {scale_report['source']}，离散度 {scale_report.get('spread')}）", flush=True)
    else:
        print(f"度量尺度: 不改（{scale_report.get('reason', '无依据')}）", flush=True)
    timings["metric_scale"] = round(factor, 4)

    # 全局尺度之后再单独处理个别离谱的物体：小物体的包围盒被深度噪声撑大，
    # 实测同一段视频里玩具火车会被量成一米一，而桌子只差百分之十几。
    # 尺寸回收默认关掉。三条真实视频上做过消融，用的是唯一直接衡量「像不像」的指标：
    #   plarail  开 0.281 / 关 0.473   传送带 开 0.098 / 关 0.098   牛顿摆 开 0.017 / 关 0.022
    # plarail 那条差了 0.19，是整张消融表里最大的单项效应。原因大概是：把物体按先验
    # 缩回去的同时，相机距离并没有跟着改，于是取景整个变了——度量上更对，画面上更不像。
    # 要重新打开的话，得先把相机和物体尺寸一起解，而不是只动物体。
    resized = ([] if (a.no_scale or not (cfg.get("synthesis") or {}).get("size_regularisation", False))
               else regularize_object_sizes(program, vlm_sizes))
    if resized:
        print(f"尺寸回收: {len(resized)} 个物体 "
              f"{[(r['id'], r['observed_m'], '->', r['prior_m']) for r in resized]}", flush=True)
    timings["resized"] = len(resized)

    # 先把**排成一条线的同类小碎块**合成一个长条，再去重。顺序不能反：碎片合成
    # 长条之后，重复检测用的那个「相对对角线」的尺度就变了。
    #
    # 为什么需要这一步（#56）：真值里一根轨道是 6.4 比 1 的长条，而掩码只覆盖它的
    # 一小段，于是一根轨道被切成好几块、每块都成了独立「物体」。实测十一条片段上，
    # 落在真值 static 上的 25 个检出只对应 **11 个**真值结构件（平均 2.27 次，
    # 一个被切成 6 块），碎片尺寸中位只有真值的 0.208。
    #
    # 这一条解释了为什么二十条「把假的滤掉」的信号全部失败：**那些东西不是假的**，
    # 是真实结构件的碎片，有深度、有掩码、有一致的跨帧跟踪。
    syn = cfg.get("synthesis") or {}
    if syn.get("merge_fragment_chains", True):
        chains = merge_fragment_chains(
            program,
            max_gap_diagonals=float(syn.get("fragment_max_gap_diagonals", 2.5)),
            min_chain=int(syn.get("fragment_min_chain", 3)),
            min_elongation=float(syn.get("fragment_min_elongation", 3.0)))
        if chains:
            print(f"碎片连链: 合掉 {sum(len(c['absorbed']) for c in chains)} 个，"
                  f"成链 {len(chains)} 条 "
                  f"{[(c['kept'], c['class'], c['elongation']) for c in chains[:4]]}", flush=True)
        timings["fragment_chains"] = len(chains)

    # 同一个物体被检出两遍的合掉。放在接触装配之前：重复的节点会互相穿插，
    # 让后面的稳定步做无用功，而且它们本来就不该存在。
    dedup_threshold = float((cfg.get("synthesis") or {}).get("dedup_relative_distance", 0.0))
    duplicates = merge_duplicates(program, dedup_threshold, evidence=evidence)
    if duplicates:
        print(f"重复检出: 合掉 {len(duplicates)} 个 "
              f"{[(d['dropped'], '->', d['kept']) for d in duplicates[:5]]}", flush=True)
    timings["deduped"] = len(duplicates)

    moves = settle_on_ground(program)
    if moves:
        applied = [m for m in moves if m["applied"]]
        print(f"接触装配: 抬起 {len(applied)} 个物体，最大 {max((m['lift_m'] for m in applied), default=0):.3f} m"
              f"，放弃 {len(moves) - len(applied)} 个", flush=True)
    program, repair_report, unresolved = repair_candidate(program, evidence, None, cfg)
    timings["repairs"] = len(repair_report.get("repairs", []))
    timings["settled"] = sum(1 for m in moves if m["applied"])
    if unresolved:
        print(f"修复后仍有 {len(unresolved)} 条未解决: "
              f"{sorted({u['code'] for u in unresolved})}", flush=True)

    (out / "program.json").write_text(json.dumps(program, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "program_before_settle.json").write_text(
        json.dumps(program, indent=1, ensure_ascii=False), encoding="utf-8")
    result = compile_program(program, out / "game", cfg)
    rows = []
    if result["ok"] and not a.no_settle:
        # 编译一次先把场景放进物理里稳一稳，再把采纳的位姿写回去重编译。
        # 感知给的是按可见表面拟合的包围盒中心，物体常常半悬空或互相穿插。
        separate = (a.separate_seconds if a.separate_seconds is not None
                    else float((cfg.get("synthesis") or {}).get("separate_seconds", 0.0)))
        settled = _settle(Path(result["game_dir"]), out, a.settle_seconds, separate)
        rows = apply_settled(program, settled)
        taken = [r for r in rows if r["applied"]]
        timings["stabilised"] = len(taken)
        if taken:
            # 仿真允许极小的穿透来保持接触稳定，而校验器按几何判定，两者容差对不上。
            # 所以稳定之后再过一次接触装配，把贴着地面的物体精确推到表面上。
            after = [m for m in settle_on_ground(program) if m["applied"]]
            print(f"物理稳定: {len(taken)} 个物体落位，最大位移 "
                  f"{max(r['moved_m'] for r in taken):.3f} m，"
                  f"放弃 {len(rows) - len(taken)} 个；稳定后再贴合 {len(after)} 个", flush=True)
            (out / "program.json").write_text(json.dumps(program, indent=1, ensure_ascii=False), encoding="utf-8")
            result = compile_program(program, out / "game", cfg)
            if not result["ok"]:
                # 仍然不过就回到稳定之前那一版：稳定是锦上添花，不该把整条链路挡死
                print("物理稳定后校验未过，回退到稳定前的位姿", flush=True)
                program = json.loads((out / "program_before_settle.json").read_text(encoding="utf-8"))
                result = compile_program(program, out / "game", cfg)
                timings["stabilised"] = 0
    if not result["ok"]:
        print("编译失败:", json.dumps(result["validation"], ensure_ascii=False)[:400])
        return 2
    completeness = assess(program, evidence, result["assets"],
                          load_masks(out / "perception"))
    if a.probe_interaction:
        probe = probe_interaction(Path(result["game_dir"]), out,
                                  str(REPO / "scripts/node_harness.sh"), a.style_swap,
                                  behaviour_swap=a.behaviour_swap)
        completeness["interactivity"] = merge_probe(completeness["interactivity"], probe)
    if a.likeness:
        # 唯一直接回答「像不像」的数：按输入的相机轨迹重渲染，和原帧比
        frames_json = out / "perception" / "frames" / "frames.json"
        frames_list = json.loads(frames_json.read_text(encoding="utf-8"))["frames"] \
            if frames_json.is_file() else []
        # 下限那条标尺要一段**别的**视频的帧。没有就只报上限。
        unrelated = [str(p) for p in sorted(
            (REPO / "data/clips/trimmed").glob("*.mp4"))][:1]
        others: list[str] = []
        for other in sorted((out.parent).glob("*/perception/frames/frames.json")):
            if other.parts[-4] == out.name:
                continue
            try:
                rows = json.loads(other.read_text(encoding="utf-8"))["frames"]
            except (OSError, ValueError):
                continue
            others = [r["file"] for r in rows[::max(1, len(rows) // 8)] if r.get("file")][:8]
            if others:
                break
        # 干净的下限：另一条片段重建出来的场景。同样是低多边形渲染，唯一的区别是场景不对。
        # 做消融时要用 --likeness-reference 把它固定住：自动挑的话，随着各变体陆续产出，
        # 挑到的参照会变，下限跟着变，归一化就不可比了（踩过）。
        if a.likeness_reference:
            other_game = Path(a.likeness_reference)
        else:
            other_game = next((g for g in sorted(out.parent.glob("*/game"))
                               if g.parts[-2] != out.name and (g / "program.json").is_file()), None)
        like = likeness(Path(result["game_dir"]), frames_list, out / "likeness",
                        str(REPO / "scripts/node_harness.sh"),
                        Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"],
                        unrelated_frames=others, unrelated_game=other_game)
        completeness["likeness"] = like
        if like.get("ok"):
            scale = ""
            if like.get("reference_self") is not None:
                scale = f"；标尺 上限 {like['reference_self']}"
                floor_key = ("reference_other_render" if "reference_other_render" in like
                             else "reference_unrelated")
                if like.get(floor_key) is not None:
                    scale += (f" 下限 {like[floor_key]}（{like.get('normalised_against')}）"
                              f"，归一化 {like.get('normalised')}")
            print(f"重渲染相似度: 中位数 {like['median']}（{like['n']} 帧，"
                  f"范围 {like['min']}~{like['max']}）{scale}", flush=True)
        else:
            print(f"重渲染相似度: 没算成（{like.get('reason')}）", flush=True)
    fit_rounds = (a.fit_colours if a.fit_colours is not None
                  else int((cfg.get("synthesis") or {}).get("colour_fit_rounds", 0)))
    if fit_rounds > 0:
        # #55 的闭环：量残差、按残差调颜色、重编译、再量。阻尼 0.7 不一轮到位——
        # 增益是从一次带噪声的测量估出来的，到位容易过冲。
        frames_file = out / "perception" / "frames" / "frames.json"
        times = ([fr["t"] for fr in
                  json.loads(frames_file.read_text(encoding="utf-8"))["frames"]][:6]
                 if frames_file.is_file() else [])
        for round_index in range(fit_rounds):
            if not times:
                break
            residual = colour_residual(Path(result["game_dir"]), program, evidence,
                                       out / f"colour_fit{round_index}", times)
            if not residual.get("ok"):
                print(f"颜色拟合第 {round_index + 1} 轮没算成（{residual.get('reason')}）",
                      flush=True)
                break
            rows = fit_colours(program, residual, damping=0.7)
            # 反照率撞到 255 之后增益就被截断，残差卡成平台。那时唯一还能动的自由度
            # 是光照强度，它是场景级的，样本量大、噪声小。
            lit = fit_lighting(program, rows,
                               luma_ratio=residual.get("luma_ratio_median"))
            if lit:
                print(f"  反照率余量用尽（{lit['clipped_objects']} 个物体被截断），"
                      f"光照抬 {lit['factor']} 倍", flush=True)
            rebuilt = compile_program(program, out / "game", cfg)
            if not rebuilt["ok"]:
                print("颜色拟合之后校验未过，这一轮放弃", flush=True)
                break
            result = rebuilt
            print(f"颜色拟合第 {round_index + 1} 轮: 调了 {len(rows)} 个物体，"
                  f"调之前每通道 {residual['channel_error_median']}/255、"
                  f"亮度比 {residual['luma_ratio_median']}", flush=True)

    if a.colour_residual:
        # 样式那一半。重渲染相似度几乎看不见颜色，所以这一项必须单独量，
        # 否则「样式恢复得准不准」没有数——#52 要的正是这个。
        frames_file = out / "perception" / "frames" / "frames.json"
        times = []
        if frames_file.is_file():
            times = [fr["t"] for fr in
                     json.loads(frames_file.read_text(encoding="utf-8"))["frames"]][:6]
        if times:
            residual = colour_residual(Path(result["game_dir"]), program, evidence,
                                       out / "colour", times)
            completeness["colour"] = residual
            if residual.get("ok"):
                print(f"颜色残差: 每通道 {residual['channel_error_median']}/255，"
                      f"亮度比 {residual['luma_ratio_median']}"
                      f"（{residual['objects']} 个物体）", flush=True)
            else:
                print(f"颜色残差: 没算成（{residual.get('reason')}）", flush=True)

    print(format_table(completeness), flush=True)
    resolved = sum(1 for r in result["assets"]["requests"] if r["resolved"].startswith("library"))
    summary = {"clip": a.clip, "objects": len(program.get("objects", [])),
               "static": len(program.get("static", [])), "asset_nodes": asset_nodes,
               "assets_resolved": resolved, "image_retrieval": bool(index) if root else False,
               "timings": timings, "completeness": completeness, "stabilised": rows, "decisions": decisions, "metric_scale": scale_report, "resized": resized, "game_dir": result["game_dir"]}
    (out / "rebuild.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "decisions"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
