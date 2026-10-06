#!/usr/bin/env python3
"""检查弹性碰撞链场景的物理行为是否说得通。

这是 DSL 的 dynamic 运动类型的功能性验收：光能编译、能渲染不够，
得确认 Rapier 真的在驱动这些球，而且动量确实从一头传到了另一头。
"""
from __future__ import annotations
import json, sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def track(states: list[dict], oid: str) -> list[tuple[float, float]]:
    """返回 (时间, x 坐标) 序列。"""
    out = []
    for s in states:
        obj = next((o for o in s.get("objects", []) if o["id"] == oid), None)
        if obj: out.append((s["t"], obj["pos"][0]))
    return out


def main(argv: list[str]) -> int:
    states = load(Path(argv[1]))
    if not states:
        print("  状态文件是空的"); return 1

    first, last = track(states, "ball_1"), track(states, "ball_5")
    if not first or not last:
        print("  找不到 ball_1 或 ball_5，物体 id 对不上"); return 1

    moved_first = first[-1][1] - first[0][1]
    moved_last = last[-1][1] - last[0][1]
    print(f"  ball_1 的 x 位移 = {moved_first:+.3f} m")
    print(f"  ball_5 的 x 位移 = {moved_last:+.3f} m")

    checks = []
    # 1) 物理真的在动。脚本运动类型下这些球会原地不动，所以这一条能区分物理有没有接上。
    any_motion = any(abs(x - t[0][1]) > 1e-3 for t in (first, last) for _, x in t)
    checks.append(("物理在驱动物体（有位移）", any_motion))

    # 2) 左边那个球是被撞停或撞回来的，不该一路穿过去。
    checks.append(("ball_1 没有穿过整列球", moved_first < 1.6))

    # 3) 动量传到了最右边：它必须朝正 x 方向被推走。
    checks.append(("动量传到了 ball_5（它向 +x 移动）", moved_last > 0.05))

    # 4) 没有爆掉。穿透或求解发散会让坐标飞出场景。
    sane = all(abs(x) < 20 for t in (first, last) for _, x in t)
    checks.append(("坐标没有发散", sane))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '失败'}] {name}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
