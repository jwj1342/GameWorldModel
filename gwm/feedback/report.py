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
    accs, recalls, precs, class_hits, confusion = [], [], [], [], Counter()
    scene_recalls, scene_precs = [], []
    shape_hits, scale_hits, scale_ratios = [], [], []
    mean_matched = start_only = 0           # 配对判据的诊断，见 gt_metrics.match_criterion
    for r in runs:
        w = (r.get("metrics") or {}).get(window) or {}
        mt, ob = w.get("motion_type") or {}, w.get("objects") or {}
        # 场景几何那一档：object + static 一起。真值里 static 比 object 还多
        # （整批 69 对 59），只报 objects 等于有一半的几何体没在量。
        sc = w.get("scene") or {}
        if sc.get("recall") is not None: scene_recalls.append(float(sc["recall"]))
        if sc.get("precision") is not None: scene_precs.append(float(sc["precision"]))
        crit = ob.get("match_criterion") or {}
        if isinstance(crit.get("by_mean_distance"), int):
            mean_matched += crit["by_mean_distance"]
            start_only += len(crit.get("only_by_start_position") or [])
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
        # 配对只看位置不看类名，所以召回和精确率量的是**定位**不是**识别**。
        # 类名决定检索哪个资产，资产决定场景长什么样——这一列不报，
        # 「场景复原得像不像」就没在量（见 #56）。按**物体**汇总，和准确率一致。
        # 形状和尺度：按**物体**汇总，和准确率一致。这两列才是「复原得像不像」
        # 该报的数——类名在当前合成数据集上和形状脱钩，恒等于噪声（见 #56）。
        if ob.get("shape_ok_frac") is not None and ob.get("matched"):
            k = round(float(ob["shape_ok_frac"]) * ob["matched"])
            shape_hits.extend([1.0] * k + [0.0] * (ob["matched"] - k))
        if ob.get("scale_ok_frac") is not None and ob.get("matched"):
            k = round(float(ob["scale_ok_frac"]) * ob["matched"])
            scale_hits.extend([1.0] * k + [0.0] * (ob["matched"] - k))
        if ob.get("scale_ratio") is not None:
            scale_ratios.append(float(ob["scale_ratio"]))
        if isinstance(ob.get("class_correct"), int) and ob.get("matched"):
            class_hits.extend([1.0] * ob["class_correct"])
            class_hits.extend([0.0] * (ob["matched"] - ob["class_correct"]))
        for truth, preds in (mt.get("confusion") or {}).items():
            for pred, n in preds.items(): confusion[(truth, pred)] += n

    # 常数基线：「不管看到什么都答 static」能拿多少分，也就是配对物体里真值本来
    # 就是 static 的比例。混淆矩阵的行是真值，按真值一侧求和即可。
    truth_total = sum(confusion.values())
    truth_static = sum(n for (truth, _), n in confusion.items() if truth == "static")
    always_static = round(truth_static / truth_total, 6) if truth_total else None

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
        "scene_recall": stat(scene_recalls),
        "scene_precision": stat(scene_precs),
        "class_accuracy": stat(class_hits),
        "shape_ok": stat(shape_hits),
        "scale_ok": stat(scale_hits),
        "scale_ratio": stat(scale_ratios),
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items(), key=lambda kv: -kv[1])},
        # 按整段平均距离配对，会把运动判错的物体一并算成没检出。这两个数说明那个偏差
        # 有多大：start_position_only 是「位置对上了、但运动没对上所以被判成未检出」的
        # 物体数。它不为零，就说明召回量到的不只是检出，运动类型准确率的分母也有偏。
        "matched_by_mean_distance": mean_matched,
        "matched_by_start_position_only": start_only,
        # 「不管看到什么都答 static」能拿多少分。这一项必须和准确率并排看，否则读不出
        # 区分力：实测当前主线准确率 0.250，而判对的 6 个**全部**是 static->static，
        # 常数基线同样是 6/24 = 0.250——一模一样。那不是「准确率偏低」，是这个指标
        # 此刻对方法没有任何区分力。做成一列是为了下次不用有人想起来才去算。
        "always_static_accuracy": always_static,
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
    rows = ["| 配置 | 片段数 | 轨迹误差中位数 (m) | 运动类型准确率 | 恒答 static | 配对物体数 | 物体召回 | 物体精确率 | 场景召回 | 场景精确率 | 形状对 | 尺度中位 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in table.items():
        f = lambda s, k="median": "—" if not s else (f"{s[k]:.3f}" + (f" ± {s['std']:.3f}" if k == "mean" else ""))
        acc = a["motion_type_accuracy"]
        acc_txt = "—" if not acc else format(acc["mean"], ".3f")
        acc_n = "—" if not acc else acc["n"]
        base = a.get("always_static_accuracy")
        base_txt = "—" if base is None else format(base, ".3f")
        # 类名那一列**不进表**：当前合成数据集上它和形状脱钩，恒等于噪声，
        # 放进主表只会误导。它仍然算在 JSON 里，换数据集时再拿出来。
        shp, scl = a.get("shape_ok"), a.get("scale_ratio")
        shp_txt = "—" if not shp else format(shp["mean"], ".3f")
        scl_txt = "—" if not scl else format(scl["median"], ".2f")
        rows.append(f"| {name} | {a['n_runs']} | {f(a['trajectory_m'])} | "
                    f"{acc_txt} | {base_txt} | {acc_n} | "
                    f"{f(a['object_recall'])} | {f(a['object_precision'])} | "
                    f"{f(a.get('scene_recall'))} | {f(a.get('scene_precision'))} | "
                    f"{shp_txt} | {scl_txt} |")
    return "\n".join(rows)


def load_runs(paths: list[str | Path]) -> list[dict]:
    """从一批 eval 产物里读结果。每个文件是一次运行的评测输出。"""
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        out.append({"clip": d.get("clip"), "config": d.get("config"), "metrics": d.get("metrics")})
    return out
