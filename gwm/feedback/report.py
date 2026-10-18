"""把一批运行结果汇总成论文表格要的形状。

现状是测量代码有三份、各算各的、结果散在不同 JSON 里，没法回答
「在 N 条测试样本上方法 A 比方法 B 好多少」——而那是论文表格的全部内容。
这个模块负责最后一步：把逐条运行的结果聚合成可汇报的数。

聚合方式是刻意选的：
  轨迹误差按物体汇总而不是按片段，因为一个片段里物体数量不一样，
    先按片段平均会让物体少的片段权重过高
  运动类型准确率同理
  召回和误检按片段汇总，因为它们本来就是片段级的量
  每一项都报样本量，否则读者没法判断一个差值是不是噪声
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _pool_objects(runs: list[dict], window: str, key: str) -> list[float]:
    """把所有片段的逐物体数汇到一起。"""
    out = []
    for r in runs:
        w = (r.get("metrics") or {}).get(window) or {}
        per = (w.get("trajectory") or {}).get("per_object") or {}
        if key == "trajectory": out.extend(float(v) for v in per.values())
    return out


def aggregate(runs: list[dict], window: str = "full") -> dict:
    """runs 里每一项是 {clip, config, metrics}，metrics 是 gt_metrics.evaluate 的输出。"""
    traj = _pool_objects(runs, window, "trajectory")
    accs, recalls, precs, confusion = [], [], [], Counter()
    for r in runs:
        w = (r.get("metrics") or {}).get(window) or {}
        mt, ob = w.get("motion_type") or {}, w.get("objects") or {}
        # 按**物体**汇总，不是按片段。按片段平均的话，只配上一个物体的片段和配上十个的
        # 片段权重一样，而配上一个的那种片段准确率只能是 0 或 1，中位数会被它们带着走。
        # 实测踩过：二十条片段共 35 个配对物体、判对 5 个（0.143），逐片段中位数却是
        # 0.000，因为多数片段只配上一两个物体且恰好都判错。
        matched = int(mt.get("n") or 0)
        if mt.get("accuracy") is not None and matched:
            accs.extend([1.0] * round(float(mt["accuracy"]) * matched))
            accs.extend([0.0] * (matched - round(float(mt["accuracy"]) * matched)))
        if ob.get("recall") is not None: recalls.append(float(ob["recall"]))
        if ob.get("precision") is not None: precs.append(float(ob["precision"]))
        for truth, preds in (mt.get("confusion") or {}).items():
            for pred, n in preds.items(): confusion[(truth, pred)] += n

    def stat(xs: list[float]) -> dict | None:
        if not xs: return None
        a = np.asarray(xs, float)
        return {"median": float(np.median(a)), "mean": float(a.mean()),
                "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a)}

    return {
        "window": window,
        "n_runs": len(runs),
        "trajectory_m": stat(traj),                 # 按物体汇总
        "motion_type_accuracy": stat(accs),         # 按物体汇总，n 就是配对上的物体数
        "object_recall": stat(recalls),
        "object_precision": stat(precs),
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items(), key=lambda kv: -kv[1])},
    }


def compare(groups: dict[str, list[dict]], window: str = "full") -> dict:
    """几组配置放在一起比，这就是论文主表的形状。"""
    return {name: aggregate(runs, window) for name, runs in groups.items()}


def as_markdown(table: dict) -> str:
    """渲染成 markdown，直接能贴进论文或 status 文档。

    哪一列报中位数、哪一列报均值，不是风格问题，是对不对的问题。
    轨迹误差是连续量，中位数是稳健统计，该报中位数。运动类型准确率汇总到物体一级
    之后是一串 0 和 1，**它的中位数只会取 0 或 1**：判对不到一半就恒等于 0.000，
    超过一半就恒等于 1.000，中间的信息全丢。而 0/1 的均值本来就是准确率的定义。

    实测后果：几何消融的三档真实准确率是 0.182 / 0.158 / 0.200，这张表当时全印成
    0.000，我据此写过「换真值相机、换真值深度都纹丝不动」的结论——结论方向没错，
    但「纹丝不动」的证据其实是量具糊掉了，不是现象。召回和精确率是片段级的连续量，
    仍报中位数。
    """
    rows = ["| 配置 | 片段数 | 轨迹误差中位数 (m) | 运动类型准确率 | 配对物体数 | 物体召回 | 物体精确率 |",
            "|---|---|---|---|---|---|---|"]
    for name, a in table.items():
        f = lambda s, k="median": "—" if not s else (f"{s[k]:.3f}" + (f" ± {s['std']:.3f}" if k == "mean" else ""))
        acc = a["motion_type_accuracy"]
        acc_txt = "—" if not acc else format(acc["mean"], ".3f")
        acc_n = "—" if not acc else acc["n"]
        rows.append(f"| {name} | {a['n_runs']} | {f(a['trajectory_m'])} | "
                    f"{acc_txt} | {acc_n} | "
                    f"{f(a['object_recall'])} | {f(a['object_precision'])} |")
    return "\n".join(rows)


def load_runs(paths: list[str | Path]) -> list[dict]:
    """从一批 eval 产物里读结果。每个文件是一次运行的评测输出。"""
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        out.append({"clip": d.get("clip"), "config": d.get("config"), "metrics": d.get("metrics")})
    return out
