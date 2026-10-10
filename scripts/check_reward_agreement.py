#!/usr/bin/env python3
"""无真值的奖励信号，和有真值的指标，排序一不一致（issue #17）。

为什么要问这个：#17 想用渲染比对的分数当训练信号，因为它在真实视频上也能算，
不需要标注。但 #4 指出那个分数是拿程序和感知结果比的，而感知本身有误差。
于是 #17 的前提就悬着——「等 #4 做完奖励问题就解决了」在逻辑上不成立，
因为真实视频根本没有真值可用。

这个实验给那个矛盾一个可量化的回答。做法是造一批质量已知递减的候选程序，
两套指标各排一次序，看相关系数。

  一致性高 → 感知奖励是真值指标的可用代理，可以在无标注真实视频上训练，
             #17 的选项一（接受噪声奖励）就有了证据
  一致性低 → 这条路走不通，要么只在合成数据上做，要么换信号

论文里这是方法章节对训练信号的验证，支撑第四条贡献。
"""
from __future__ import annotations
import argparse, copy, json, subprocess, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from gwm.config import REPO, load_config, site_name
from gwm.feedback.gt_metrics import evaluate, load_states


def graded_candidates(program: dict) -> list[tuple[str, dict]]:
    """质量已知递减的一串候选。位移越大、改动越多，程序越差。"""
    out = [("原样", copy.deepcopy(program))]
    for shift in (0.15, 0.4, 0.8, 1.6):
        p = copy.deepcopy(program)
        for o in p["objects"]:
            if "pose" in o: o["pose"]["pos"][0] += shift
            for inst in o.get("instances", []): inst["pos"][0] += shift
        out.append((f"整体平移 {shift} 米", p))
    p = copy.deepcopy(program)
    for o in p["objects"]:
        if (o.get("motion") or {}).get("type") not in (None, "static"): o["motion"] = {"type": "static"}
    out.append(("所有运动改成静止", p))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="一次完整运行的目录，要有 perception/ 和 program.json")
    ap.add_argument("--work", required=True)
    ap.add_argument("--harness", default=str(REPO / "scripts" / "node_harness.sh"))
    ap.add_argument("--out", default="docs/results/reward_agreement.json")
    a = ap.parse_args(argv)

    run = Path(a.run); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    cfg = load_config([REPO / "configs/default.yaml", REPO / f"configs/{site_name()}.yaml"])
    gt = json.loads((REPO / "examples/handwritten/program.json").read_text())
    pipeline = json.loads((run / "program.json").read_text())

    # 两族候选都要跑，因为两个指标各自只在其中一族上有定义：
    #   以管线程序为基准 —— 物体 id 和感知证据对得上，所以无真值奖励算得出来；
    #                      但和真值差了七八米，真值指标一个都匹配不上
    #   以真值程序为基准 —— 真值指标算得出来；但 id 和证据对不上，无真值奖励归零
    # 两边都跑一遍，这个「没法放在同一批候选上比」的事实才会显式呈现出来，
    # 而不是看起来像实验失败。根因是感知重建的位移，记在 #7。
    families = [("以管线程序为基准", pipeline), ("以真值程序为基准", gt)]

    from gwm.feedback.metrics import compute_metrics
    from gwm.feedback.render import render
    from gwm.perception.run import load_masks
    ev = json.loads((run / "perception" / "evidence.json").read_text())
    masks = load_masks(run / "perception")
    frames = ev["frames"] if isinstance(ev.get("frames"), list) else ev.get("keyframes", [])
    key_times = [k["t"] for k in ev["keyframes"]]

    def states_of(prog: dict, name: str) -> list[dict]:
        d = work / name; d.mkdir(parents=True, exist_ok=True)
        pj = d / "program.json"; pj.write_text(json.dumps(prog, ensure_ascii=False))
        subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(pj), str(d / "game")],
                       check=True, cwd=REPO, capture_output=True)
        subprocess.run([a.harness, "record.mjs", "--game", str(d / "game"), "--out", str(d / "rec"),
                        "--fps", "10", "--duration", str(gt["meta"]["duration"]), "--width", "320", "--height", "180"],
                       check=True, cwd=REPO, capture_output=True)
        return load_states(d / "rec" / "gt_states.jsonl")

    gt_states = states_of(gt, "gt")
    rows = []
    for fam, base in families:
      print(f"\n  === {fam} ===")
      for i, (label, cand) in enumerate(graded_candidates(base)):
        tag = f"{'p' if fam.startswith('以管线') else 'g'}{i}"
        d = work / f"c{tag}"; d.mkdir(parents=True, exist_ok=True)
        pj = d / "program.json"; pj.write_text(json.dumps(cand, ensure_ascii=False))
        subprocess.run([sys.executable, "-m", "gwm.compiler.compile", str(pj), str(d / "game")],
                       check=True, cwd=REPO, capture_output=True)
        idx = render(d / "game", key_times, d / "render",
                     cfg["feedback"]["render_width"], cfg["feedback"]["render_height"],
                     tuple(cfg["feedback"]["passes"]))
        reward = compute_metrics(cand, idx, d / "render", ev, frames, masks, cfg)["summary"]["score"]
        truth = evaluate(gt, cand, gt_states, states_of(cand, f"s{tag}"))
        traj = truth["full"]["trajectory"]["median_m"]
        rows.append({"family": fam, "label": label, "reward_no_gt": round(reward, 4),
                     "gt_traj_median_m": None if traj is None else round(traj, 4),
                     "gt_motion_acc": truth["full"]["motion_type"]["accuracy"]})
        print(f"    {label:20s} 无真值奖励 {reward:.4f}   真值轨迹误差 {'无法匹配' if traj is None else f'{traj:.4f}'}")

    # 只有两个指标都算得出来、而且奖励不是常数，相关系数才有意义
    ok = [r for r in rows if r["gt_traj_median_m"] is not None
          and r["reward_no_gt"] is not None and r["reward_no_gt"] > 0]
    rho = p = None
    if len(ok) >= 3 and len({r["reward_no_gt"] for r in ok}) > 1:
        rho, p = spearmanr([r["reward_no_gt"] for r in ok], [r["gt_traj_median_m"] for r in ok])
        print(f"\n  斯皮尔曼相关 rho = {rho:.3f}  p = {p:.4f}  （预期为负：奖励高则误差小）")
    else:
        print("\n  算不出相关系数：没有一批候选能同时拿到两个指标。")
        print("  原因是感知重建和真值差了约八米，两边没有共同的物体对应关系（见 #7）。")
        print("  所以 #17 那个「能不能在有噪声的感知奖励上训练」的问题，现在还答不了，卡在感知上。")

    # 无真值奖励单独看：它在自己那一族候选里对质量下降有没有反应
    mono = {}
    for fam, _ in families:
        xs = [r["reward_no_gt"] for r in rows if r["family"] == fam and "平移" in r["label"] or
              (r["family"] == fam and r["label"] == "原样")]
        xs = [r["reward_no_gt"] for r in rows if r["family"] == fam][:5]
        mono[fam] = {"rewards": xs, "单调下降": all(a >= b for a, b in zip(xs, xs[1:])) and len(set(xs)) > 1}
    for fam, m in mono.items():
        print(f"  {fam}：奖励随平移量 {m['rewards']}，单调下降 = {m['单调下降']}")

    report = {"rows": rows, "spearman_rho": None if rho is None else float(rho),
              "p_value": None if p is None else float(p), "n_comparable": len(ok),
              "monotonicity": mono,
              "conclusion": ("两个指标在同一批候选上算不出来，因为感知重建和真值差约八米，"
                             "没有共同的物体对应关系。#17 的问题卡在 #7 上。")
                            if rho is None else "见 spearman_rho"}
    out = REPO / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(f"  明细写到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
