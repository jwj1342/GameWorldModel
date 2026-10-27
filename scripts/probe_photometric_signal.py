#!/usr/bin/env python
"""把渲染结果和原帧比，这个信号带不带「程序参数该往哪改」的信息。

这是「3DGS 当教师」那条路的第一个 MVP。整条路的前提是：**像素层面的比较能当监督
信号用**。如果把一个物体挪开 10 厘米、渲出来和原帧的相似度没有变化，那整条路从
一开始就是死的，不用去拟合高斯。

`gwm/feedback/likeness.py` 已经实现了「按输入相机轨迹重渲染、和原帧比」，但它
**刻意避开了像素级比较**，注释里的理由是：

> 重建出来的是低多边形的程序化场景，配的是库里的资产，和真实视频在像素上本来就
> 不可能接近；要看的是「结构和布局像不像」。

那个判断可能是对的，也可能挡住了一条路——LEGO-Bench 的外观分恰恰是像素级的
（重渲染后逐像素比，最大通道误差 ≤ 30/255 的占比）。**这个脚本就是去分辨这两者。**

## 做法

拿一份已有的重建，把**一个物体**按已知幅度挪动或缩放，重渲，看两种信号怎么响应：

  像素一致率   LEGO-Bench 那一套：最大通道误差 ≤ 30/255 的像素占比
  DINOv2 余弦  likeness.py 现在用的那一套

判据是**单调性和效应量**：扰动越大、分数越低，而且最小的那一档扰动要能和噪声分开。
两种信号分别判，因为很可能一种带信号另一种不带。

**渲染是确定性的，所以重复渲染的噪声是零。** 真正的噪声地板来自「不同扰动方向
给出的分数差异」，所以这里对同一幅度做多个方向，报方向间的离散度。

用法:
  python -m scripts.probe_photometric_signal --report <产物目录> --work <临时目录>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.config import load_config                    # noqa: E402
from gwm.feedback.likeness import pick_times, render_at  # noqa: E402


def pixel_agreement(render_path: Path, source_path: Path, delta: int = 30) -> tuple[float, float]:
    """返回 (一致像素占比, 平均绝对误差)。

    一致的定义照搬 LEGO-Bench：最大**单通道**误差不超过 delta（0–255 上的 30）。
    渲染图和原帧尺寸可能不同，按原帧尺寸最近邻缩放——不插值，免得把误差抹平。
    """
    from PIL import Image
    r = Image.open(render_path).convert("RGB")
    s = Image.open(source_path).convert("RGB")
    if r.size != s.size:
        r = r.resize(s.size, Image.NEAREST)
    a = np.asarray(r, dtype=np.int16)
    b = np.asarray(s, dtype=np.int16)
    diff = np.abs(a - b)
    return float((diff.max(axis=2) <= delta).mean()), float(diff.mean())


def perturb(program: dict, node_id: str, shift: np.ndarray, scale: float) -> dict:
    """复制一份程序，把指定节点挪动并缩放。只动这一个节点，别的一个字节不改。"""
    out = json.loads(json.dumps(program))
    for node in (out.get("objects") or []) + (out.get("static") or []):
        if node.get("id") != node_id:
            continue
        pos = (node.get("pose") or {}).get("pos")
        if isinstance(pos, list) and len(pos) == 3:
            node["pose"]["pos"] = [float(p) + float(d) for p, d in zip(pos, shift)]
        geom = node.get("geom") or {}
        if isinstance(geom.get("extent"), list):
            geom["extent"] = [float(v) * scale for v in geom["extent"]]
        for key in ("radius", "radius_top", "radius_bottom", "height"):
            if isinstance(geom.get(key), (int, float)):
                geom[key] = float(geom[key]) * scale
    return out


def biggest_object(program: dict, evidence: dict) -> str | None:
    """挑掩码占画面最大的那个物体——它在画面里最显眼，信号最强。
    信号在它身上都看不见的话，在小物体上更不可能看见。"""
    areas = {str(o.get("id")): float(o.get("mask_area_frac") or 0.0)
             for o in (evidence.get("objects") or [])}
    ids = {str(n.get("id")) for n in (program.get("objects") or [])}
    cands = {k: v for k, v in areas.items() if k in ids}
    return max(cands, key=cands.get) if cands else (next(iter(ids), None) if ids else None)


def score_variant(game_src: Path, program: dict, times: list[float], work: Path,
                  tag: str, harness: str, by_time: dict, dino) -> dict | None:
    """把 program 写进 game 目录的副本里渲一遍，返回两种信号的中位数。"""
    stage = work / f"game_{tag}"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(game_src, stage)
    (stage / "program.json").write_text(json.dumps(program), encoding="utf-8")
    out = work / f"render_{tag}"
    res = render_at(stage, times, out, harness)
    if not res["ok"]:
        return None
    px, mae, dn = [], [], []
    for entry in res["index"].get("frames", []):
        t = round(float(entry.get("t", entry.get("requested_t", 0.0))), 3)
        rgb = entry.get("files", {}).get("rgb")
        if not rgb or not by_time:
            continue
        nearest = min(by_time, key=lambda k: abs(k - t))
        if abs(nearest - t) > 0.5:
            continue
        agree, err = pixel_agreement(out / rgb, Path(by_time[nearest]))
        px.append(agree); mae.append(err)
        if dino is not None:
            dn.append(float(dino.embed(out / rgb) @ dino.embed(by_time[nearest])))
    if not px:
        return None
    return {"pixel": float(np.median(px)), "mae": float(np.median(mae)),
            "dino": float(np.median(dn)) if dn else None, "n": len(px)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--clip", default=None, help="只跑一条；不给就按 --clips 跑多条")
    ap.add_argument("--clips", type=int, default=1,
                    help="跑几条片段。要用这个结论去定方向的话，n=1 太小。")
    ap.add_argument("--harness", default="scripts/node_harness.sh")
    ap.add_argument("--times", type=int, default=4)
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args(argv)
    # dinov2 的权重键在 configs/vulcan.yaml 里，default.yaml 没有。不叠上这一份，
    # DINOv2 那一臂会静默缺席——而「像素 vs DINOv2」正是这个实验要比的两件事。
    default_cfgs = [str(REPO / "configs" / "default.yaml")]
    if (REPO / "configs" / "vulcan.yaml").is_file():
        default_cfgs.append(str(REPO / "configs" / "vulcan.yaml"))
    cfg = load_config(a.config or default_cfgs)
    report, work = Path(a.report), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    runs = [d for d in sorted(report.iterdir())
            if (d / "game" / "program.json").is_file() and (d / "program.json").is_file()]
    if a.clip:
        runs = [d for d in runs if d.name == a.clip]
    if not runs:
        print("没有可用的产出目录"); return 1
    runs.sort(key=lambda d: -len(json.loads((d / "program.json").read_text(encoding="utf-8")).get("objects") or []))
    chosen = runs[:max(1, a.clips)]

    dino = None
    try:
        from gwm.feedback.metrics import Dino
        dino = Dino(Path(cfg["paths"]["weights"]) / cfg["weights"]["dinov2"])
    except Exception as e:
        print(f"（DINOv2 加载失败，只报像素信号：{type(e).__name__}: {e}）\n")

    allrows = []
    for run in chosen:
        got = run_one(run, times_n=a.times, work=work, harness=a.harness, dino=dino)
        if got:
            allrows.append((run.name, got))
    if not allrows:
        print("没有一条跑成"); return 1
    summarise(allrows)
    return 0


def run_one(run: Path, times_n: int, work: Path, harness: str, dino):

    import numpy as np
    program = json.loads((run / "program.json").read_text(encoding="utf-8"))
    evidence = json.loads((run / "perception" / "evidence.json").read_text(encoding="utf-8"))
    frames = json.loads((run / "perception" / "frames" / "frames.json").read_text(encoding="utf-8"))["frames"]
    times = pick_times(frames, times_n)
    by_time = {round(float(f["t"]), 3): str(run / "perception" / "frames" / Path(f["file"]).name)
               for f in frames if f.get("file")}
    by_time = {k: v for k, v in by_time.items() if Path(v).is_file()}

    target = biggest_object(program, evidence)
    if target is None or not times or not by_time:
        print(f"片段 {run.name} 缺少可用的物体、时刻或原帧"); return None
    diag = None
    for n in (program.get("objects") or []):
        if n.get("id") == target and isinstance((n.get("geom") or {}).get("extent"), list):
            diag = float(np.linalg.norm(np.asarray(n["geom"]["extent"], float)))
    print(f"片段 {run.name}，扰动对象 {target}（对角线 {diag:.3f} m 如果有），"
          f"{len(times)} 个时刻，原帧 {len(by_time)} 张\n")

    variants = [("base", np.zeros(3), 1.0)]
    for d in (0.05, 0.10, 0.20, 0.40):
        variants.append((f"x+{d}", np.asarray([d, 0, 0]), 1.0))
    # 同一幅度换方向，用来估「方向间离散度」，那就是这个信号的噪声地板
    for axis, name in ((np.asarray([-1.0, 0, 0]), "x-"), (np.asarray([0, 0, 1.0]), "z+"),
                       (np.asarray([0, 0, -1.0]), "z-")):
        variants.append((f"{name}0.20", axis * 0.20, 1.0))
    for s in (0.8, 1.25):
        variants.append((f"scale{s}", np.zeros(3), s))

    rows = []
    for tag, shift, scale in variants:
        prog = program if tag == "base" else perturb(program, target, shift, scale)
        got = score_variant(run / "game", prog, times, work, f"{run.name}_{tag}", harness, by_time, dino)
        if got is None:
            print(f"  {tag:<12} 渲染失败"); continue
        rows.append((tag, got))
        print(f"  {tag:<12} 像素一致率 {got['pixel']:.4f}  平均绝对误差 {got['mae']:6.2f}"
              + (f"  DINOv2 {got['dino']:.4f}" if got["dino"] is not None else ""))

    if not any(t == "base" for t, _ in rows):
        return None
    return rows


def summarise(allrows: list) -> None:
    """把多条片段的结果并排。单条片段的结论不足以定方向，所以这里只看跨片段的一致性。"""
    import numpy as np
    print("\n" + "=" * 72)
    print("—— 判据一：单调性。沿 +x 挪，幅度越大分数该越低（两种信号分别看）——\n")
    print(f"{'片段':<14}{'信号':<10}{'0.05':>9}{'0.10':>9}{'0.20':>9}{'0.40':>9}   单调？")
    verdict = {"pixel": [], "dino": []}
    for name, rows in allrows:
        base = dict(rows)["base"]
        for key, label in (("pixel", "像素"), ("dino", "DINOv2")):
            if base.get(key) is None:
                continue
            seq = []
            for d in (0.05, 0.10, 0.20, 0.40):
                tag = f"x+{d}".rstrip("0").rstrip(".") if False else None
                g = next((g for t, g in rows if t.startswith("x+") and abs(float(t[2:]) - d) < 1e-9), None)
                seq.append(None if g is None or g.get(key) is None else g[key] - base[key])
            ok = all(v is not None for v in seq) and all(seq[i] >= seq[i + 1] for i in range(3))
            verdict[key].append(ok)
            cells = "".join(f"{v:>+9.4f}" if v is not None else f"{'—':>9}" for v in seq)
            print(f"{name:<14}{label:<10}{cells}   {'是' if ok else '**否**'}")
    print()
    for key, label in (("pixel", "像素一致率"), ("dino", "DINOv2")):
        v = verdict[key]
        if v:
            print(f"  {label}：{sum(v)}/{len(v)} 条片段单调")

    print("\n—— 判据二：最小扰动的效应量 对 方向噪声地板 ——\n")
    print(f"{'片段':<14}{'信号':<10}{'0.05m 效应':>12}{'方向极差':>11}{'效应/噪声':>11}")
    ratios = {"pixel": [], "dino": []}
    for name, rows in allrows:
        d = dict(rows)
        base = d["base"]
        for key, label in (("pixel", "像素"), ("dino", "DINOv2")):
            if base.get(key) is None:
                continue
            small = next((g for t, g in rows if t.startswith("x+") and abs(float(t[2:]) - 0.05) < 1e-9), None)
            same = [g[key] for t, g in rows if t.endswith("0.20") or (t.startswith("x+") and abs(float(t[2:]) - 0.20) < 1e-9)]
            if small is None or small.get(key) is None or len(same) < 2:
                continue
            eff = abs(small[key] - base[key])
            floor = float(np.ptp(np.asarray(same)))
            r = eff / floor if floor > 1e-12 else float("inf")
            ratios[key].append(r)
            print(f"{name:<14}{label:<10}{eff:>12.4f}{floor:>11.4f}{r:>11.2f}")
    print()
    for key, label in (("pixel", "像素一致率"), ("dino", "DINOv2")):
        v = ratios[key]
        if v:
            print(f"  {label} 的 效应/噪声 中位 {np.median(v):.2f}"
                  f"（**要远大于 1 才能当监督信号**）")

    print("\n判读：")
    print("  效应/噪声 明显小于 1，说明「把物体挪动最小的那一档」带来的变化淹没在")
    print("  「换个方向挪同样距离」的差异里——这个信号分辨不出参数该往哪改。")
    print("  单调性若不成立，说明它连「错得多」和「错得少」都分不清。")
    print("  **DINOv2 如果呈现为正的差值，那是方向反了**：扰动让它认为更像，")
    print("  那么 likeness.py 现在用它当「像不像」的指标就会奖励把物体挪开。")


if __name__ == "__main__":
    raise SystemExit(main())
