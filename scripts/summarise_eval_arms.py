"""把几份 eval_*.json 并排成一张表，方便读 A/B。

字段名以 `gwm/feedback/report.py` 的 `aggregate()` 为准——那里是 `shape_ok` /
`scale_ok`，不是 `shape_ok_frac` / `scale_ok_frac`，也没有 `shape_error` 这个汇总
（逐物体的形状误差没有被汇总成一列）。写错了的话整列会静默变成破折号，所以这里把
**实际存在但没被要求的键**也打印出来，免得下次又猜。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COLS = ["object_recall", "object_precision", "scene_recall", "scene_precision",
        "shape_kind_accuracy", "shape_ok", "scale_ok", "scale_ratio",
        "motion_type_accuracy", "trajectory_m"]


def agg_of(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    full = d.get("full") or {}
    return next(iter(full.values()), None)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="<标签>:<eval json 路径>，可以给多次")
    a = ap.parse_args(argv)

    arms = []
    for spec in a.arm:
        label, _s, path = spec.partition(":")
        if not path:
            ap.error(f"--arm 要写成 <标签>:<路径>，收到 {spec!r}")
        arms.append((label, Path(path)))

    print(f"{'臂':<12s}" + "".join(f"{c:>22s}" for c in COLS))
    seen_keys: set[str] = set()
    n_runs = {}
    for label, path in arms:
        agg = agg_of(path)
        if agg is None:
            print(f"{label:<12s}  （读不到 {path}）"); continue
        seen_keys |= set(agg)
        n_runs[label] = agg.get("n_runs")
        cells = []
        for c in COLS:
            v = agg.get(c)
            cells.append(f"{v['median']:.4f}" if isinstance(v, dict) and v.get("median") is not None else "—")
        print(f"{label:<12s}" + "".join(f"{v:>22s}" for v in cells))
    print("\n片段数：" + "，".join(f"{k} {v}" for k, v in n_runs.items()) +
          "   （不一样就别横着比，少掉的往往是最差的那条）")
    extra = sorted(k for k in seen_keys - set(COLS)
                   if isinstance(k, str) and not k.startswith(("confusion", "window")))
    if extra:
        print(f"\n这些键也在，但没进表：{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
