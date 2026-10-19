#!/usr/bin/env python
"""第三档（关节交互）真产出来的那些关节，对不对。

背景（#63）。可交互性四档里第三档的运行时早就做完了（`Manipulator.constrain` 支持
slide/hinge 带限位，`harness/interaction_probe.mjs` 会往限位外拖去探它），但链路
**从来没产出过一个关节**——`infer_articulation` 默认关着，理由是扫阈值时「报出来的
里既是真关节、类型也判对」的比例过不了一半。

这个脚本回答的是下一个问题：**把它打开之后产出的那些关节，落在哪、对不对。**
扫阈值那个脚本是在逐物体的原始判据上算的，不经过合成；这里看的是**真的写进
program.json 的 `interaction.grab`**，并用评测的场景配对去对真值。

两个要点，都是踩过才知道的：

  `interaction.grab` 是个**字符串**（"hinge" / "slide"），不是字典。第一版写成
  `grab.get("type")`，于是一个都没数到，还报了个干干净净的 0。

  关节会落在 **static** 节点上，不只 objects。只遍历 objects 会漏掉大半。

用法:
  python -m scripts.probe_articulation_output --report <产物目录> \
      --eval <eval.json> --gt-root <真值根>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# hinge <-> 绕轴转，slide <-> 沿直线平移。往复与否不影响这个对应
# （和 scripts/sweep_articulation.py 的 AGREE 保持一致）。
AGREE = {"hinge": {"revolute", "periodic_rotate", "spin"},
         "slide": {"prismatic", "periodic_translate"}}
JOINT_KINDS = {"revolute", "prismatic"}


def grab_kind(node: dict) -> str | None:
    grab = (node.get("interaction") or {}).get("grab")
    if isinstance(grab, str):
        return grab or None
    return (grab or {}).get("type") if isinstance(grab, dict) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gt-root", required=True)
    a = ap.parse_args(argv)

    rows = json.loads(Path(a.eval).read_text(encoding="utf-8")).get("rows", [])
    emitted = matched = right = gt_joints = 0
    detail: list[tuple] = []
    for r in rows:
        scene = ((r.get("metrics") or {}).get("full") or {}).get("scene") or {}
        pairs = {q: g for g, q in (scene.get("pairs") or {}).items() if q}
        pdp = Path(a.report) / r["clip"] / "program.json"
        gtp = Path(a.gt_root) / r["clip"] / "program.json"
        if not (pdp.is_file() and gtp.is_file()):
            continue
        pd = json.loads(pdp.read_text(encoding="utf-8"))
        gt = json.loads(gtp.read_text(encoding="utf-8"))
        kind_of = {n["id"]: (n.get("motion") or {}).get("type", "static")
                   for n in (gt.get("objects") or [])}
        kind_of.update({f"static::{n['id']}": (n.get("motion") or {}).get("type", "static")
                        for n in (gt.get("static") or [])})
        gt_joints += sum(1 for k in kind_of.values() if k in JOINT_KINDS)
        for node in ((pd.get("objects") or []) + (pd.get("static") or [])):
            kind = grab_kind(node)
            if not kind:
                continue
            emitted += 1
            g = pairs.get(node["id"]) or pairs.get(f"static::{node['id']}")
            if g is None:
                detail.append((r["clip"], node["id"], kind, "没配上任何真值节点"))
                continue
            matched += 1
            truth = kind_of.get(g, "?")
            ok = truth in AGREE.get(kind, set())
            right += ok
            detail.append((r["clip"], node["id"], kind,
                           f"真值 {truth} {'对' if ok else '错'}"))

    print(f"写出关节 {emitted} 个；配上真值节点的 {matched} 个；类型判对 {right} 个")
    print(f"真值里 revolute + prismatic 合计 {gt_joints} 个")
    if emitted:
        print(f"  可用率（配上且类型对 / 写出）= {right / emitted:.3f}")
    if matched:
        print(f"  查准（类型对 / 配上）        = {right / matched:.3f}")
    print(f"  查全（类型对 / 真值关节）      = {right / max(gt_joints, 1):.3f}")
    if emitted and matched < emitted:
        print(f"\n  **{emitted - matched} / {emitted} 个关节落在配不上任何真值节点的检出上。**")
        print("  「配不上」是什么意思要看这一轮用的是哪种感知，两种情况别混：")
        print("    神经分割 —— 多半是幻觉检出。它们的点云本来就不稳，配准会读出很大的")
        print("                表观转动，于是被判成 hinge。等于给不存在的东西装铰链。")
        print("    真值分割 —— 不存在幻觉，所以「配不上」只意味着**位置误差超过了容差**。")
        print("                这时候这一行说明的是摆位不准，不是检测在瞎报。")
        print("  判断关节推断本身行不行，要看「类型判对 / 真值关节」那个查全——它不受")
        print("  配不上的影响。")
    print("\n逐条：")
    for d in detail:
        print("   ", *d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
