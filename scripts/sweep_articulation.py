#!/usr/bin/env python
"""扫关节推断的阈值。

`scripts/probe_motion_signal.py --mode articulation` 会把每个物体的原始判据和真值类型
dump 成 rows.json。这个脚本在那份数据上扫阈值，不用重跑感知。

要回答的问题只有一个：**有没有一组阈值，能让报出来的关节里大多数是真的。** 如果没有，
这个功能就该默认关掉——往程序里塞一个猜的关节，用户拖一下就会发现，比不写更糟。

用法:
  python -m scripts.sweep_articulation --rows $WORK/rows.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def evaluate(rows: list[dict], min_rotation: float, min_consistency: float,
             min_travel: float, min_straightness: float, max_residual: float) -> dict:
    """按一组阈值重新判一遍，算查准查全。

    rows 里记的是**按原阈值**判出来的结果，所以这里只能在原来就触发过的样本上收紧，
    不能放松——放松需要重新跑推断。收紧正是我们要的方向。
    """
    tp = fp = fn = kind_ok = 0
    for row in rows:
        fired = row.get("fired")
        keep = False
        if fired == "hinge":
            keep = (row["total_rotation_deg"] >= min_rotation
                    and (row.get("axis_consistency") or 0) >= min_consistency
                    and row["residual"] <= max_residual)
        elif fired == "slide":
            keep = (row["travel_m"] >= min_travel
                    and (row.get("straightness") or 0) >= min_straightness
                    and row["residual"] <= max_residual)
        if row["is_joint"]:
            if keep:
                tp += 1
                expect = "hinge" if row["gt_kind"] == "revolute" else "slide"
                kind_ok += int(fired == expect)
            else:
                fn += 1
        elif keep:
            fp += 1
    reported = tp + fp
    return {"reported": reported, "true_joints": tp + fn,
            "precision": round(tp / reported, 3) if reported else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "kind_correct": round(kind_ok / tp, 3) if tp else None,
            "useful": round(kind_ok / reported, 3) if reported else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    a = ap.parse_args()
    rows = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    fired = [r for r in rows if r.get("fired")]
    print(f"样本 {len(rows)}，其中真关节 {sum(1 for r in rows if r['is_joint'])}，"
          f"按原阈值触发 {len(fired)}")
    print(f"{'转角':>6}{'轴一致':>8}{'行程':>7}{'直线度':>8}{'残差':>7}"
          f"{'报出':>6}{'查准':>7}{'查全':>7}{'类型对':>8}{'可用率':>8}")
    grid = [(8.0, 0.85, 0.05, 0.90, 9.9),      # 现行阈值
            (15.0, 0.90, 0.10, 0.93, 9.9),
            (25.0, 0.93, 0.15, 0.95, 9.9),
            (25.0, 0.95, 0.15, 0.95, 0.05),
            (40.0, 0.95, 0.25, 0.97, 0.05),
            (60.0, 0.97, 0.35, 0.98, 0.03)]
    for rot, cons, travel, straight, residual in grid:
        m = evaluate(rows, rot, cons, travel, straight, residual)
        print(f"{rot:>6.0f}{cons:>8.2f}{travel:>7.2f}{straight:>8.2f}{residual:>7.2f}"
              f"{m['reported']:>6}{str(m['precision']):>7}{str(m['recall']):>7}"
              f"{str(m['kind_correct']):>8}{str(m['useful']):>8}")
    print("\n可用率 = 报出来的关节里「既是真关节、类型也判对」的比例。"
          "\n这是决定要不要默认开启的那个数：它低于一半，就不该写进程序。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
