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
        if mt.get("accuracy") is not None: accs.append(float(mt["accuracy"]))
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
        "motion_type_accuracy": stat(accs),         # 按片段汇总
        "object_recall": stat(recalls),
        "object_precision": stat(precs),
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items(), key=lambda kv: -kv[1])},
    }


def compare(groups: dict[str, list[dict]], window: str = "full") -> dict:
    """几组配置放在一起比，这就是论文主表的形状。"""
    return {name: aggregate(runs, window) for name, runs in groups.items()}


def as_markdown(table: dict) -> str:
    """渲染成 markdown，直接能贴进论文或 status 文档。"""
    rows = ["| 配置 | 片段数 | 轨迹误差中位数 (m) | 运动类型准确率 | 物体召回 | 物体精确率 |",
            "|---|---|---|---|---|---|"]
    for name, a in table.items():
        f = lambda s, k="median": "—" if not s else (f"{s[k]:.3f}" + (f" ± {s['std']:.3f}" if k == "mean" else ""))
        rows.append(f"| {name} | {a['n_runs']} | {f(a['trajectory_m'])} | "
                    f"{f(a['motion_type_accuracy'])} | {f(a['object_recall'])} | {f(a['object_precision'])} |")
    return "\n".join(rows)


def load_runs(paths: list[str | Path]) -> list[dict]:
    """从一批 eval 产物里读结果。每个文件是一次运行的评测输出。"""
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        out.append({"clip": d.get("clip"), "config": d.get("config"), "metrics": d.get("metrics")})
    return out
