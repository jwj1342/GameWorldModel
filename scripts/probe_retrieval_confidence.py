"""检索的置信度能不能用——这条同时服务创新点一和创新点二。

为什么要问。模板路线（拿资产网格当 ICP 模板）在**正确网格**下把静止/转动的
判别比从 2.60 提到 4.31，但在**检索到的网格**下比基线还差。而检索命中率
只有 9.0%（真值分割）。整体开模板 = 9% 的物体受益、91% 受害，净亏——
这就是模板路线挂起的原因。

但「整体开」不是唯一选项。**如果能分辨哪些物体检索对了**，就只在那批上开模板。
不需要检索变好，只需要检索的**置信度可用**。

判据：把每个决策按某个置信信号分桶，看命中率随桶怎么变。
  单调上升、高桶明显高于整体 → 置信度可用，模板路线可以按桶开
  各桶拉平 → 置信度没用，这条也关掉

候选信号：轮廓 IoU、文字分、图像分、图像名次。都已经记在 rebuild.json 里。

纪律：命中率的分母是**该桶里的决策数**，不是全体；每个桶要报样本量，
样本量 < 10 的桶不解读（这条吃过亏）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def gt_refs(program: dict) -> set[str]:
    out = set()
    for n in (program.get("objects") or []) + (program.get("static") or []):
        ref = ((n.get("geom") or {}).get("ref") or "").strip()
        if ref:
            out.add(ref)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--bins", type=int, default=4)
    a = ap.parse_args(argv)

    rows = []
    for d in sorted(Path(a.report).glob("*test_*")):
        rb = d / "rebuild.json"
        if not rb.is_file():
            continue
        clip = "test_" + d.name.split("test_")[-1]
        gp = Path(a.gt_root) / clip / "program.json"
        if not gp.is_file():
            continue
        wanted = gt_refs(json.loads(gp.read_text(encoding="utf-8")))
        if not wanted:
            continue
        try:
            recs = json.loads(rb.read_text(encoding="utf-8")).get("decisions") or []
        except Exception:
            continue
        for r in recs:
            if not r.get("chosen"):
                continue
            rows.append({
                "hit": 1 if str(r["chosen"]) in wanted else 0,
                "轮廓 IoU": r.get("silhouette_iou"),
                "文字分": r.get("score"),
                "图像分": r.get("image_score_best"),
            })
    if not rows:
        print("没有带 chosen 的决策，拒绝出数"); return 1
    base = sum(x["hit"] for x in rows) / len(rows)
    print(f"决策 {len(rows)} 个，整体命中率 {base:.3f}\n")

    for key in ("轮廓 IoU", "文字分", "图像分"):
        vals = [(x[key], x["hit"]) for x in rows if isinstance(x[key], (int, float))]
        if len(vals) < 20:
            print(f"{key}: 样本只有 {len(vals)} 个，不分桶")
            continue
        vals.sort(key=lambda t: t[0])
        n, b = len(vals), a.bins
        print(f"{key}（样本 {n}）")
        for i in range(b):
            lo, hi = i * n // b, (i + 1) * n // b
            chunk = vals[lo:hi]
            if not chunk:
                continue
            rate = sum(h for _v, h in chunk) / len(chunk)
            mark = "  （样本太少，不解读）" if len(chunk) < 10 else ""
            print(f"  第{i+1}桶 [{chunk[0][0]:.3f}, {chunk[-1][0]:.3f}]  "
                  f"命中 {sum(h for _v,h in chunk)}/{len(chunk)} = {rate:.3f}{mark}")
        top = vals[3 * n // 4:]
        if len(top) >= 10:
            tr = sum(h for _v, h in top) / len(top)
            lift = tr / base if base > 0 else float("inf")
            print(f"  最高那 25%：命中 {tr:.3f}，是整体的 {lift:.2f} 倍"
                  + ("  → 这个信号可用" if lift >= 1.5 else "  → 分不开，这个信号没用"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
