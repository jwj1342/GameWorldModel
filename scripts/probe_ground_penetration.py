"""我们检出来又因为「陷进地面」扔掉的物体有多少——它们在漏检里被记成了「缺席」。

起因。留出集的 `test_00002` 产出程序编译不过：五个物体全部陷进地面 2.7~3.9 米。
翻它的 `meta` 时发现管线**自己早就记着这件事**：`dropped_sunk` 里逐个列着被丢掉的
节点和它的 `bottom_over_height`（底面低于地面多少个自身高度，负数=陷进去）。

这改变了对漏检的理解。`probe_recall_decomposition.py` 把漏掉的真值物体分成三桶，
其中「缺席」的定义是「那个位置附近没有任何预测节点」，读起来像是检测器根本没看见。
但如果检测器**看见了、算出来却在地下、于是被丢掉**，那它在程序里当然不存在，
照样落进「缺席」——性质却完全不同：前者教师修不了（给不了语义），
后者教师正好能修（几何对了就不会沉到地下）。

所以这里数两件事：
  1. 每条片段 `meta.dropped_sunk` 丢了几个、陷多深；
  2. 活下来的节点里还有多少在地面以下（纯几何量，从 pose 和 extent 算）。

坑记一下：节点分在 `objects` 和 `static` **两个**数组里，地面节点在 `static`。
第一版只找 `objects`，于是「一个地面节点都没找到」，四个配置全空。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def all_nodes(program: dict) -> list[dict]:
    """objects 和 static 合起来才是完整的一份节点。"""
    out = []
    for key in ("objects", "static"):
        value = program.get(key)
        if isinstance(value, list):
            out += [n for n in value if isinstance(n, dict)]
    return out


def is_ground(node: dict) -> bool:
    cls = str(node.get("class", "")).strip().lower()
    return cls == "ground" or str(node.get("id", "")).startswith("ground")


def ground_top(program: dict) -> float | None:
    for node in all_nodes(program):
        if is_ground(node):
            pos = (node.get("pose") or {}).get("pos") or [0, 0, 0]
            ext = ((node.get("geom") or {}).get("extent")) or [0, 0, 0]
            try:
                return float(pos[1]) + float(ext[1]) / 2.0
            except (TypeError, ValueError, IndexError):
                return None
    return None


def penetrations(program: dict) -> list[float]:
    top = ground_top(program)
    if top is None:
        return []
    out = []
    for node in all_nodes(program):
        if is_ground(node):
            continue
        pos = (node.get("pose") or {}).get("pos")
        ext = (node.get("geom") or {}).get("extent")
        if not (isinstance(pos, list) and len(pos) == 3 and isinstance(ext, list) and len(ext) == 3):
            continue
        try:
            out.append(top - (float(pos[1]) - float(ext[1]) / 2.0))     # 正数 = 陷进去
        except (TypeError, ValueError):
            continue
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", action="append", required=True)
    a = ap.parse_args(argv)

    for d in a.report_dir:
        root = Path(d)
        print(f"\n########## {root.name} ##########")
        print(f"  {'片段':14s} {'存活':>4s} {'丢掉(沉底)':>11s} {'丢掉的最深':>11s} {'存活里还陷着':>13s}")
        tot_alive = tot_dropped = tot_still = 0
        drop_depths: list[float] = []
        still_depths: list[float] = []
        no_ground = []
        for clip_dir in sorted(root.glob("test_*")):
            prog_path = clip_dir / "program.json"
            if not prog_path.is_file():
                continue
            program = json.loads(prog_path.read_text(encoding="utf-8"))
            dropped = ((program.get("meta") or {}).get("dropped_sunk")) or []
            depths = [abs(float(x["bottom_over_height"])) for x in dropped
                      if isinstance(x, dict) and isinstance(x.get("bottom_over_height"), (int, float))]
            alive = [n for n in all_nodes(program) if not is_ground(n)]
            pen = penetrations(program)
            if ground_top(program) is None:
                no_ground.append(clip_dir.name)
            still = [p for p in pen if p > 0.01]
            tot_alive += len(alive); tot_dropped += len(dropped); tot_still += len(still)
            drop_depths += depths; still_depths += still
            worst = f"{max(depths):.2f} 倍高" if depths else "—"
            print(f"  {clip_dir.name:14s} {len(alive):4d} {len(dropped):11d} {worst:>11s} {len(still):13d}")
        if no_ground:
            print(f"  （这些片段没有地面节点，存活穿透算不了：{', '.join(no_ground)}）")
        total_seen = tot_alive + tot_dropped
        if not total_seen:
            print("  没有可算的节点"); continue
        print(f"  ---- 合计：检出 {total_seen} 个节点 ----")
        print(f"  因为沉到地下被丢掉：{tot_dropped} / {total_seen} = {tot_dropped / total_seen:.3f}")
        if drop_depths:
            print(f"    丢掉的那些陷多深：中位 {statistics.median(drop_depths):.2f} 倍自身高度，"
                  f"最深 {max(drop_depths):.2f} 倍")
        if still_depths:
            print(f"  存活但仍在地面以下：{tot_still} 个，中位 {statistics.median(still_depths):.2f} m")
    print("""
判读：
  丢掉的比例高、而且陷进去是自身高度的好几倍 → 不是贴合误差，是整体标定/深度错了
  一个量级。这些物体**检出来了**，只是算到了地下、于是被扔掉——在漏检拆解里它们
  被记成「缺席」，但性质上属于教师能修的那一类，不是「检测器没看见」。
  比例低、深度浅 → 这条线索没价值，「缺席」就是真的没检出来。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
