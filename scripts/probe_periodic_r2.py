"""周期拟合的 r² 按真值运动类别分开看——转角修好了，这个分布有没有正过来。

为什么量它。转动闸把静止物体的假转动压下去了（低于铰链门槛 8 度的比例 0.235 →
0.588，而真转动一点没被压，0.200 不变），但端到端只把运动类型从 0.1923 挪到
0.2308——26 个配对物体里**多对了一个**，噪声量级。所以改善卡在转角的下游。

下游的链条是：转角序列 → `_sinusoid` 的 r² → 过 `periodic_min_r2`（0.70）才生成
periodic 候选 → 否则只剩 spin 和 static 两个靠得太近的 → unknown → 不写 motion
字段 = static。

已知的毛病是这个分布**是反的**：真 periodic 的 r² 中位 0.124，真静止 0.310，
而且**两边都远低于 0.70**，所以候选对谁都不生成。往下调门槛会先放进静止的那些。

所以要回答两个问题，而且顺序不能颠倒：
  一、闸有没有让分布**正过来**（真 periodic 的 r² 高于真静止的）。这是先决条件。
  二、正过来之后离 0.70 还差多少。

跑在**真值掩码**上，所以物体 id 直接就是真值节点 id，不需要配对——形状那次的教训。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROTATING = ("spin", "revolute", "rotate")


def gt_motion(program: dict) -> dict[str, str]:
    out = {o["id"]: (o.get("motion") or {}).get("type", "static")
           for o in program.get("objects") or []}
    for o in program.get("static") or []:          # 结构件也是静止的，别漏掉
        out.setdefault(o["id"], "static")
    return out


def bucket(kind: str) -> str:
    """**按真值运动类型精确分桶，不要把 spin 和 periodic_rotate 并在一起。**

    第一版并了，于是「真在转」的 r² 中位只有 0.13~0.20，看起来像「周期拟合对真在转
    的物体也不灵」。但 `spin` 是**持续转动**，角度序列是一条斜坡；拿正弦去拟合斜坡，
    r² 本来就该低。这个检验只对 `periodic_rotate` 有意义，并进 spin 是混口径——
    会把一个可能正常的子类拖进一个假的否定结论里。
    """
    k = str(kind or "static")
    if k in ("static", "", "none"):
        return "真静止"
    if "periodic_rotate" in k:
        return "真·周期转动"          # 只有这一类该拟合出高 r²
    if "spin" in k:
        return "真·持续自转"          # 斜坡，正弦拟合本来就低，不该算失败
    if "revolute" in k:
        return "真·铰链"
    if "periodic" in k:
        return "真·周期平移"
    return "真平移"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True,
                    help="<臂标签>:<产出根>:<真值根>，可以给多次")
    a = ap.parse_args(argv)

    for spec in a.pair:
        label, report, gt_root = spec.split(":", 2)
        by_bucket: dict[str, list[float]] = defaultdict(list)
        gate_value = None
        n_obj = n_with_r2 = 0
        for clip_dir in sorted(Path(report).glob("*test_*")):
            ev_path = clip_dir / "perception" / "evidence.json"
            clip = clip_dir.name.split("test_")[-1]
            gt_path = Path(gt_root) / f"test_{clip}" / "program.json"
            if not (ev_path.is_file() and gt_path.is_file()):
                continue
            kinds = gt_motion(json.loads(gt_path.read_text(encoding="utf-8")))
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            for obj in ev.get("objects") or []:
                oid = str(obj.get("id"))
                node = oid if oid in kinds else oid.rsplit("_", 1)[0]
                if node not in kinds:
                    continue
                n_obj += 1
                guess = obj.get("motion_guess") or {}
                best = None
                for entry in guess.get("rejected_periodic") or []:
                    if entry.get("type") == "periodic_rotate" and entry.get("r2") is not None:
                        best = float(entry["r2"])
                        gate_value = entry.get("gate", gate_value)
                if best is None:
                    continue
                n_with_r2 += 1
                by_bucket[bucket(kinds[node])].append(best)
        print(f"\n########## {label} ##########")
        print(f"  物体 {n_obj} 个，其中有 periodic_rotate 的 r² 记录的 {n_with_r2} 个"
              f"（门槛 {gate_value}）")
        order = ["真·周期转动", "真·持续自转", "真·铰链", "真·周期平移", "真平移", "真静止"]
        for name in order:
            vs = by_bucket.get(name) or []
            if not vs:
                print(f"    {name:6s} 没有样本"); continue
            over = sum(1 for v in vs if gate_value and v >= float(gate_value)) / len(vs)
            print(f"    {name:6s} 中位 {statistics.median(vs):.4f}  均值 {statistics.mean(vs):.4f}"
                  f"  最大 {max(vs):.4f}  过门槛比例 {over:.3f}  样本 {len(vs)}")
        # 判据只在 periodic_rotate 上成立——它是唯一「该拟合出高 r²」的类。
        rot = by_bucket.get("真·周期转动") or []
        sta = by_bucket.get("真静止") or []
        if rot and sta:
            d = statistics.median(rot) - statistics.median(sta)
            print(f"    → **真·周期转动 减 真静止** 的中位差 {d:+.4f}"
                  f"（{chr(0x65b9)+chr(0x5411)+chr(0x5bf9)+chr(0x4e86) if d > 0 else chr(0x8fd8)+chr(0x662f)+chr(0x53cd)+chr(0x7684)}）")
        else:
            print("    → 没有 periodic_rotate 的样本，这个判据算不了")
    print("""
怎么读（顺序不能颠倒）：
  一、先看「真在转 减 真静止」是不是正的。负的话分布是反的，往下调门槛只会先放进
      静止的那些，调阈值没有意义。
  二、正了之后再看离门槛还差多少、过门槛比例有没有起来。
  一没成立之前，二的任何数字都不用看。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
