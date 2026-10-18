"""运动分类到底有没有赢过「恒答 static」，以及压掉转动预测之后还剩多少。

用法:
  python -m scripts.probe_motion_baseline $GWM_SCRATCH/dataset/*_eval.json

实测结论（七份产物、161 个配对物体）：压掉转动预测之后，每一个配置的准确率都
**精确等于**常数基线，差全是 +0.000。只要有任何一次平移类判对了，这个差就会是正的。
所以运动分类一次都没有正确识别出任何一个会动的物体——它拿到的每一分都来自
「把静止的叫静止」。

转动那一路已经查了一整轮，十八条排除，每一条修法都不成立（见 #7）。但位置那一路
**是好的**：轨迹误差中位 0.138 m，而真值里会动的物体整段位移中位 2.549 m，
信噪比 18 倍。

所以换个问法：**如果分类器碰到转动一律说 unknown，只回答平移类的问题，
和「恒答 static」那个常数基线比，差会不会第一次变正？**

这个脚本不改链路、不重跑感知，直接在已有的评测产物上重算：把混淆矩阵里**预测**
为转动类的那些改判成 unknown（也就是不写 motion，等于 static），再算一遍准确率
和基线。
"""
import json
import sys
from pathlib import Path

ROTATION = {"spin", "periodic_rotate", "revolute"}


def recompute(confusion: dict, suppress: set) -> tuple[float, float, int]:
    """返回 (准确率, 恒答 static 的基线, 配对物体数)。

    confusion 的键是 "真值->预测"。把预测落在 suppress 里的改判成 static
    ——链路里 unknown 就是不写 motion 字段，渲出来和 static 没区别。
    """
    total = correct = truth_static = 0
    for key, n in confusion.items():
        truth, pred = key.split("->")
        if pred in suppress:
            pred = "static"
        total += n
        correct += n if truth == pred else 0
        truth_static += n if truth == "static" else 0
    if not total:
        return float("nan"), float("nan"), 0
    return correct / total, truth_static / total, total


def main(paths):
    rows = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        full = json.loads(p.read_text(encoding="utf-8"))["full"]
        for name, agg in full.items():
            conf = agg.get("confusion") or {}
            if not conf:
                continue
            now, base, n = recompute(conf, set())
            supp, base2, _ = recompute(conf, ROTATION)
            rows.append((p.stem, name, n, now, supp, base))
    if not rows:
        print("没有可读的评测产物"); return 1
    print(f"{'产物':<22}{'配对':>5}{'现状':>8}{'压掉转动预测':>13}{'恒答static':>11}"
          f"{'现状-基线':>11}{'压掉后-基线':>12}")
    for stem, _name, n, now, supp, base in rows:
        print(f"{stem:<22}{n:>5}{now:>8.3f}{supp:>13.3f}{base:>11.3f}"
              f"{now - base:>+11.3f}{supp - base:>+12.3f}")
    print()
    print("最后两列是关键：差是正的才说明这一级真的学到了东西。")
    print("压掉转动预测之后差仍然不正，就说明问题不在转动那一路，而在更上游")
    print("——比如动得最多的物体根本没被配上对（见 #7）。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
