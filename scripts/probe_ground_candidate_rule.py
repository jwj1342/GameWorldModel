"""换一条地平面选择规则，在**已经记下来的候选**上模拟，不重跑感知。

背景。`align_world` 把候选平面筛成 `ok` 之后，按 `inliers * (0.5 + 0.5*frac_obj_above)`
挑最大的那个。对相机高度唯一的约束是 `cam_height > 0`——0.0768 也算通过。
实测三条片段因此选错：

  ab015fix/test_00000  选中候选0 相机高 0.0768，而候选1/2 给出 1.4668/1.4672
                       真值相机高 1.1085 → 候选1 缩放后距真值 8%，候选0 缩放 15.6 倍
                       结果整条片段的物体被推到地下，21 个节点被 dropped_sunk 丢光

好消息是**正确的候选就在列表里**，只是没被选中。所以这是选择规则的问题，
不是候选生成的问题。

这个脚本在 `evidence.json` 已经记下的 `plane_candidates` 上模拟新规则，
报哪些片段会改、改成什么。不重跑感知，所以验证很便宜；真要改代码之前先看这个。

新规则（**无量纲**，这点很重要——重建是相对尺度，绝对米制阈值在这里没有意义）：
  在原来的 ok 集合里，先算各候选 cam_height 的中位数 m，
  把 cam_height < ratio * m 的候选判为「平面压在相机上」丢掉，
  再在剩下的里按原来的打分挑。丢完了就退回原来的集合（不改行为）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def score(c: dict) -> float:
    return c["inliers"] * (0.5 + 0.5 * c.get("frac_obj_above", 1.0))


def old_pick(cands: list[dict]) -> dict | None:
    ok = [c for c in cands
          if c["angle_deg"] < 50 and c["cam_height"] > 0 and c.get("frac_obj_above", 1.0) >= 0.5]
    if not ok:
        ok = [c for c in cands
              if c.get("angle_to_view_deg", 999) < 35 and c.get("frac_obj_above", 1.0) >= 0.3]
    return max(ok, key=score) if ok else None


def new_pick(cands: list[dict], ratio: float) -> tuple[dict | None, bool]:
    """返回 (选中的候选, 是否因为新规则丢掉过候选)。"""
    ok = [c for c in cands
          if c["angle_deg"] < 50 and c["cam_height"] > 0 and c.get("frac_obj_above", 1.0) >= 0.5]
    if not ok:
        ok = [c for c in cands
              if c.get("angle_to_view_deg", 999) < 35 and c.get("frac_obj_above", 1.0) >= 0.3]
    if not ok:
        return None, False
    med = statistics.median([c["cam_height"] for c in ok])
    kept = [c for c in ok if c["cam_height"] >= ratio * med]
    fired = len(kept) < len(ok)
    if not kept:                      # 全被丢完就退回去，宁可不改也不要改瞎
        kept, fired = ok, False
    return max(kept, key=score), fired


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True,
                    help="产出目录和它对应的真值目录，用冒号分开：<report>:<gt_root>。"
                         "必须配对给——两个数据集的片段同名（都是 test_000NN），"
                         "按片段名索引真值会让后一个数据集覆盖前一个。")
    ap.add_argument("--ratio", type=float, default=0.25,
                    help="低于候选 cam_height 中位数这个比例的，判为平面压在相机上")
    ap.add_argument("--target", type=float, default=1.2, help="assumed_camera_height_m")
    a = ap.parse_args(argv)

    def gt_heights(root: str) -> dict[str, float]:
        """这个真值目录里每条片段的相机高度中位数。

        **按 (真值目录, 片段名) 索引，不能只按片段名。** 两个数据集的片段都叫
        test_000NN，只按名字索引的话后一个会把前一个覆盖掉，打印出来的真值就是
        另一个数据集的——第一版就是这么错的。
        """
        out: dict[str, float] = {}
        for prog in sorted(Path(root).glob("test_*/program.json")):
            try:
                program = json.loads(prog.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            keys = (program.get("camera") or {}).get("keyframes") or []
            hs = [float(k["pos"][1]) for k in keys
                  if isinstance(k.get("pos"), list) and len(k["pos"]) == 3]
            if hs:
                out[prog.parent.name] = statistics.median(hs)
        return out

    pairs = []
    for spec in a.pair:
        report, _sep, gt_root = spec.partition(":")
        if not gt_root:
            ap.error(f"--pair 要写成 <report>:<gt_root>，收到的是 {spec!r}")
        pairs.append((report, gt_root))

    print(f"规则：cam_height < {a.ratio} × 候选中位数 的判为「平面压在相机上」，丢掉再挑\n")
    changed = total = 0
    units_old: list[float] = []
    units_new: list[float] = []
    for d, gt_root in pairs:
        root = Path(d)
        gt_by_clip = gt_heights(gt_root)
        print(f"########## {root.name}   真值 {Path(gt_root).name} ##########")
        for clip_dir in sorted(root.glob("test_*")):
            ev = clip_dir / "perception" / "evidence.json"
            if not ev.is_file():
                continue
            try:
                info = ((json.loads(ev.read_text(encoding="utf-8")).get("meta") or {})
                        .get("alignment") or {})
            except (OSError, ValueError):
                continue
            cands = info.get("plane_candidates") or []
            if not cands:
                continue
            total += 1
            o, (n, fired) = old_pick(cands), new_pick(cands, a.ratio)
            if o is None or n is None:
                continue
            gt = gt_by_clip.get(clip_dir.name)
            mark = "改了" if n["idx"] != o["idx"] else ("—" if not fired else "规则触发但没改")
            # 平面选对了的话，「相对单位高度 ÷ 真值高度」就是这段重建的任意单位，
            # 各片段之间该是一致的。选错的会是离群点。这比「改了几条」更说明问题。
            def unit(c):
                return c["cam_height"] / gt if gt and gt > 1e-6 else None
            uo, un = unit(o), unit(n)
            if uo is not None:
                units_old.append(uo)
            if un is not None:
                units_new.append(un)
            gt_s = (f"真值 {gt:.3f} m  单位比 旧 {uo:.2f} → 新 {un:.2f}"
                    if gt is not None and uo is not None and un is not None else "无真值")
            print(f"  {clip_dir.name:14s} 旧选 idx{o['idx']} h={o['cam_height']:.4f} "
                  f"(×{a.target / max(o['cam_height'], 1e-9):.2f})   "
                  f"新选 idx{n['idx']} h={n['cam_height']:.4f} "
                  f"(×{a.target / max(n['cam_height'], 1e-9):.2f})   {mark:10s} {gt_s}")
            if n["idx"] != o["idx"]:
                changed += 1
    print(f"\n合计 {total} 条片段，**{changed} 条选择发生变化**")
    for label, units in (("旧规则", units_old), ("新规则", units_new)):
        if len(units) >= 3:
            lo, hi = min(units), max(units)
            print(f"  {label} 的「相对单位 ÷ 真值高度」：中位 {statistics.median(units):.2f}，"
                  f"范围 {lo:.2f}~{hi:.2f}，极差倍数 **{hi / lo:.1f}×**")
    print("  平面选对了的话这个比值该集中（它就是重建的任意单位）；极差越小越好。")
    print("""
怎么读：
  只有本来就选错的那几条该变，其余一条都不该变——否则这条规则在动好的片段。
  变了的那几条，新选中的 cam_height 应该落在一个像样的范围（缩放倍数接近 1），
  而不是从一个极端跳到另一个极端。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
