#!/usr/bin/env python
"""样式-行为解耦度：核心主张第四条的两个数（#52）。

主张是「样式与行为分成两半写」，但评测表里一个相关的数都没有——轨迹误差、运动类型
准确率、物体召回，三个全在行为那一半。这个脚本补上。

## 在哪一层测

**在执行结果上测，不在源码文本上。** 在结构化字段上比 `motion` 等于把结论写进了
定义里（#11 那条直接出代码的基线里样式和行为本来就揉在一起，没有「行为那一半」
可供比对）。所以两份程序都经 harness 跑一遍，从逐帧状态里取轨迹来比——这一层对
后端中性，谁解耦得好谁的行为漂移就小，是挣来的不是定义出来的。

## 两个数

对一组 `style_variant`（同一份行为、不同样式的两段视频 V 和 V'），模型各写一份
程序 P 和 P'：

**行为漂移**：P 和 P' 的行为那一半差多少。**真值是 0**——重新上一遍漆不该让模型把
运动写得不一样。这个数越大，说明样式信息泄进了行为通道。

**样式响应**：P 和 P' 的样式那一半差多少。这个数要**大**，否则说明模型根本没在看
样式。两个数一起才有意义：一个把样式字段全写死成默认值的模型，行为漂移是 0，
但样式响应也是 0——它不是解耦，是没在做事。

## 怎么建立对应关系

P 和 P' 是两次独立的重建，物体 id 没有可比性。所以**用真值当锚**：各自和自己那一侧
的真值配对（真值的行为在这组配对里按构造是一样的），然后在「两边都配上了同一个
真值物体」的那些上比。这样不需要 P 和 P' 直接互配，也就避开了一个会引入新误差的
配对问题。

用法:
  python -m scripts.probe_decoupling --report <产物目录> --gt-root <真值根> --work <临时目录>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.feedback.gt_metrics import _rot_y, evaluate, tracks   # noqa: E402
from scripts.eval import states_of, work_dir                   # noqa: E402


def hex_to_rgb(value) -> np.ndarray | None:
    if isinstance(value, dict):
        value = value.get("base_color")
    if not (isinstance(value, str) and value.startswith("#") and len(value) == 7):
        return None
    try:
        return np.asarray([int(value[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0
    except ValueError:
        return None


def gauge_of(metrics: dict):
    g = (metrics.get("full") or {}).get("gauge_alignment") or {}
    if "yaw_deg" not in g:
        return None
    R = _rot_y(float(g["yaw_deg"]))
    t = np.asarray([g["translation_xz_m"][0], 0.0, g["translation_xz_m"][1]], float)
    return R, t, float(g.get("scale") or 1.0)


def sample(track: dict, times: np.ndarray) -> np.ndarray:
    ts, pos = track["t"], track["pos"]
    return np.stack([np.interp(times, ts, pos[:, i]) for i in range(3)], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--harness", default="scripts/node_harness.sh")
    ap.add_argument("--max-dist", type=float, default=0.45)
    a = ap.parse_args(argv)
    report, gtroot, work = Path(a.report), Path(a.gt_root), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    pairs = [r for r in json.loads((gtroot / "pairs.json").read_text(encoding="utf-8"))
             if r.get("kind") == "style_variant"]
    if not pairs:
        print("没有 style_variant 配对"); return 1

    rows = []
    for row in pairs:
        base, var = row["base"], row["variant"]
        got = {}
        for name in (base, var):
            gp, pp = gtroot / name / "program.json", report / name / "program.json"
            if not (gp.is_file() and pp.is_file()):
                break
            gt, pred = (json.loads(p.read_text(encoding="utf-8")) for p in (gp, pp))
            dur = float((gt.get("meta") or {}).get("duration") or 8.0)
            gs = states_of(gp, work_dir(work, f"{name}_gt", name), dur, a.harness)
            ps = states_of(pp, work_dir(work, f"{name}_pred", name), dur, a.harness)
            m = evaluate(gt, pred, gs, ps, max_dist_m=a.max_dist)
            got[name] = {"gt": gt, "pred": pred, "pred_states": ps, "metrics": m,
                         "pairs": ((m.get("matching") or {}).get("pairs") or {})}
        if len(got) != 2:
            continue

        gb, gv = gauge_of(got[base]["metrics"]), gauge_of(got[var]["metrics"])
        if gb is None or gv is None:
            continue
        # 两边都配上了同一个真值物体的，才拿来比
        common = [g for g, q in got[base]["pairs"].items()
                  if q and got[var]["pairs"].get(g)]
        if not common:
            rows.append((base, 0, None, None, None)); continue

        tb = tracks(got[base]["pred_states"])
        tv = tracks(got[var]["pred_states"])
        win = ((got[base]["metrics"].get("matching") or {}).get("window_s") or [0.0, 1.0])
        times = np.linspace(float(win[0]), float(win[1]), 40)

        drifts, colour_d, gt_colour_d = [], [], []
        gt_nodes_b = {n["id"]: n for n in (got[base]["gt"].get("objects") or [])}
        gt_nodes_v = {n["id"]: n for n in (got[var]["gt"].get("objects") or [])}
        pd_b = {n["id"]: n for n in (got[base]["pred"].get("objects") or [])}
        pd_v = {n["id"]: n for n in (got[var]["pred"].get("objects") or [])}
        for g in common:
            qb, qv = got[base]["pairs"][g], got[var]["pairs"][g]
            if qb not in tb or qv not in tv:
                continue
            Rb, tbv, sb = gb
            Rv, tvv, sv = gv
            pb = sb * (sample(tb[qb], times) @ Rb.T) + tbv
            pv = sv * (sample(tv[qv], times) @ Rv.T) + tvv
            drifts.append(float(np.linalg.norm(pb - pv, axis=1).mean()))
            cb, cv = hex_to_rgb((pd_b.get(qb) or {}).get("material")), hex_to_rgb((pd_v.get(qv) or {}).get("material"))
            if cb is not None and cv is not None:
                colour_d.append(float(np.linalg.norm(cb - cv)))
            gb_c, gv_c = hex_to_rgb((gt_nodes_b.get(g) or {}).get("material")), hex_to_rgb((gt_nodes_v.get(g) or {}).get("material"))
            if gb_c is not None and gv_c is not None:
                gt_colour_d.append(float(np.linalg.norm(gb_c - gv_c)))

        rows.append((base, len(drifts),
                     float(np.median(drifts)) if drifts else None,
                     float(np.median(colour_d)) if colour_d else None,
                     float(np.median(gt_colour_d)) if gt_colour_d else None))

    print("样式-行为解耦度（style_variant 配对；真值的行为在组内按构造完全相同）\n")
    print(f"{'配对':<14}{'两边都配上的物体':>18}{'行为漂移 (m)':>14}"
          f"{'样式响应':>10}{'真值样式差':>12}")
    for base, n, drift, cd, gcd in rows:
        f = lambda v: "   —  " if v is None else f"{v:.3f}"
        print(f"{base:<14}{n:>18}{f(drift):>14}{f(cd):>10}{f(gcd):>12}")

    ds = [r[2] for r in rows if r[2] is not None]
    cs = [r[3] for r in rows if r[3] is not None]
    gs = [r[4] for r in rows if r[4] is not None]
    print()
    if ds:
        print(f"行为漂移中位 {np.median(ds):.3f} m（**真值是 0**——重上一遍漆不该改运动）")
    if cs and gs:
        print(f"样式响应中位 {np.median(cs):.3f}，对照真值的样式差 {np.median(gs):.3f}")
        print("  样式响应接近 0 = 模型没在看样式；接近真值样式差 = 跟上了。")
    print("\n两个数要一起看：把样式字段全写死成默认值的模型，行为漂移是 0，")
    print("但样式响应也是 0——那不是解耦，是没在做事。")
    print("\n口径：行为那一半在**执行结果**上比（两份程序都经 harness 跑出逐帧状态），")
    print("不在结构化字段上比，否则等于把结论写进定义。对应关系用真值当锚：各自和")
    print("自己那侧的真值配对，再在「两边都配上同一个真值物体」的那些上比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
