"""我们产出的程序，物体陷进地面多深——这是系统性误差还是个别片段的事。

起因。留出集上 `test_00002` 的产出程序**编译不过**：五个物体全部陷进地面
2.7~3.9 米，被内核的 ground_penetration 校验挡下。eval.py 会把这条记进 failed
并把它排除在汇总之外，于是留出集的表上写着「片段数 11」而不是 12——**被排除的
恰好是最差的那条**。

但一条片段说明不了什么。真正要问的是：陷地面是普遍的系统性偏移，还是个别事故？
如果是前者，那么「漏检」里的大头就不是随机噪声，而是整体标定/深度错了一个量级，
那正是稠密三维重建能修的东西；如果是后者，这条线索没价值。

做法很直白：对每个产出程序，拿地面节点的上表面当基准，算每个非地面节点的
**最低点**低于它多少。不调用内核，只读程序里的 pose 和 extent——这是个纯几何量，
不需要跑仿真。

口径说明：旋转没有展开算（只按轴对齐的半高估最低点），所以这个数是**下界**，
真实穿透只会更深不会更浅。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def ground_top(program: dict) -> float | None:
    """地面上表面的 y。没有地面节点就返回 None。"""
    for node in program.get("objects", []):
        if str(node.get("class", "")).strip().lower() == "ground" or node.get("id") == "ground":
            pos = (node.get("pose") or {}).get("pos") or [0, 0, 0]
            ext = ((node.get("geom") or {}).get("extent")) or [0, 0, 0]
            return float(pos[1]) + float(ext[1]) / 2.0
    return None


def penetrations(program: dict) -> list[tuple[str, float]]:
    top = ground_top(program)
    if top is None:
        return []
    out = []
    for node in program.get("objects", []):
        cls = str(node.get("class", "")).strip().lower()
        if cls == "ground" or node.get("id") == "ground" or str(node.get("id", "")).startswith("ground"):
            continue
        pos = (node.get("pose") or {}).get("pos")
        ext = (node.get("geom") or {}).get("extent")
        if not (isinstance(pos, list) and len(pos) == 3 and isinstance(ext, list) and len(ext) == 3):
            continue
        try:
            low = float(pos[1]) - float(ext[1]) / 2.0
        except (TypeError, ValueError):
            continue
        out.append((str(node.get("id", "?")), top - low))     # 正数 = 陷进去
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", action="append", required=True,
                    help="一个产出目录（里面是 test_*/program.json），可以给多次")
    a = ap.parse_args(argv)

    for d in a.report_dir:
        root = Path(d)
        print(f"\n########## {root.name} ##########")
        all_pen: list[float] = []
        n_clip = n_node = n_sunk = 0
        no_ground = []
        for clip_dir in sorted(root.glob("test_*")):
            prog_path = clip_dir / "program.json"
            if not prog_path.is_file():
                continue
            program = json.loads(prog_path.read_text(encoding="utf-8"))
            if ground_top(program) is None:
                no_ground.append(clip_dir.name); continue
            pen = penetrations(program)
            if not pen:
                continue
            n_clip += 1
            depths = [p for _i, p in pen]
            sunk = [p for p in depths if p > 0.01]
            n_node += len(depths); n_sunk += len(sunk)
            all_pen += depths
            worst = max(depths)
            print(f"  {clip_dir.name:14s} 节点 {len(depths):2d}  陷进去 {len(sunk):2d}  "
                  f"最深 {worst:6.2f} m  中位 {statistics.median(depths):6.2f} m")
        if no_ground:
            print(f"  （没有地面节点、跳过：{', '.join(no_ground)}）")
        if not all_pen:
            print("  没有可算的节点"); continue
        sunk_all = [p for p in all_pen if p > 0.01]
        print(f"  ---- {n_clip} 条片段，{n_node} 个节点 ----")
        print(f"  陷进地面的节点：{n_sunk} / {n_node} = {n_sunk / n_node:.3f}")
        print(f"  所有节点的穿透深度：中位 {statistics.median(all_pen):+.2f} m"
              f"（正数=陷进去）")
        if sunk_all:
            print(f"  只看陷进去的那些：中位 {statistics.median(sunk_all):.2f} m，"
                  f"最深 {max(sunk_all):.2f} m")
    print("""
判读：
  陷进去的比例高、而且深度和物体尺寸同量级 → 是整体标定/深度错了，不是随机噪声，
  这一类正是稠密三维重建（教师）能修的。
  零散几个、深度很浅 → 就是贴合误差，说明不了什么。
提醒：旋转没展开算，所以这是**下界**，真实穿透只会更深。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
