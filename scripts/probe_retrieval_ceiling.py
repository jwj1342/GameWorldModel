"""检索的上限：**拿真值类名当查询词**，还挑不挑得到那个资产。

为什么要分这一刀。真值用资产库搭之后第一次量到「检索挑对了吗」——**71 次 0 命中**。
但管线里的查询词是**检测器说的类名**，不是真值类名：真值里那个物体叫 `bedbunk`，
检测器大概率说「床」或别的词。**查询词本身就不对，再好的排序也挑不到。**

所以 0/71 可能主要是**命名**的锅，不是排序的锅，而这两件事要做的改动完全不同：

  命名的锅 → 要改检测/标注那一侧，或者让真值用检测器说得出的词；
  排序的锅 → 要改检索本身（而形状重排正是在改这个）。

这个脚本把检测这一环整个拿掉：直接拿**真值节点的类名**去查，报命中率。
跳上去 → 排序是好的，瓶颈在命名；还是零 → 检索本身就定位不到，形状重排再怎么
调也没用。

口径：报 top-1 和 top-5 两档。top-5 还是零的话，问题不在「排序把它压下去了」，
而在「它根本没进候选」——那是文字打分的事，不是重排的事。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gwm.assets.library import library_root, load_catalog      # noqa: E402
from gwm.assets.retrieve import retrieve                        # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args(argv)

    root = library_root({})
    if root is None:
        print("没有资产库"); return 1
    catalog = load_catalog(root)
    print(f"资产库 {len(catalog)} 个")

    n = top1 = topk = no_candidate = 0
    misses: list[tuple[str, str, list[str]]] = []
    for prog_path in sorted(Path(a.gt_root).glob("test_*/program.json")):
        program = json.loads(prog_path.read_text(encoding="utf-8"))
        for node in (program.get("objects") or []) + (program.get("static") or []):
            geom = node.get("geom") or {}
            ref = geom.get("ref")
            if geom.get("kind") not in ("asset", "generated") or not ref:
                continue
            n += 1
            query = str(node.get("class") or geom.get("query") or "")
            hits = retrieve(query, catalog, k=a.top_k)
            ids = [asset.id for asset, _s in hits]
            if not ids:
                no_candidate += 1
            if ids and ids[0] == ref:
                top1 += 1
            if ref in ids:
                topk += 1
            elif len(misses) < 5:
                misses.append((query, str(ref), ids[:3]))
    print("=" * 66)
    if not n:
        print("真值里没有资产节点"); return 1
    print(f"真值资产节点 {n} 个（查询词用**真值类名**，检测这一环整个拿掉）")
    print(f"  top-1 命中  {top1}/{n} = {top1 / n:.3f}")
    print(f"  top-{a.top_k} 命中  {topk}/{n} = {topk / n:.3f}")
    print(f"  一个候选都没有（文字分全零）：{no_candidate}/{n} = {no_candidate / n:.3f}")
    if misses:
        print("\n  没命中的例子（查询词 → 真值资产 ／ 实际挑出来的前三）：")
        for q, ref, got in misses:
            print(f"    {q!r} → {ref}   挑到：{got}")
    print("""
怎么读：
  top-1 跳上去 → 排序是好的，管线里那 0/71 的瓶颈在**命名**（检测器说的词和资产名
      对不上），该改的是检测/标注那一侧，不是检索。
  top-5 也接近零 → 问题不在「排序把它压下去了」而在「它根本没进候选」，
      那是文字打分的事，形状重排再怎么调也没用。
  「一个候选都没有」占比高 → 资产名和我们的词表根本不在一个语言里。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
