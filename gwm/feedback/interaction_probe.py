"""把可交互程度从「声明的」变成「量出来的」。

completeness.interactivity 数的是程序**写了**什么：`interaction.grab` 不是 false 就
算一个能拖的。声明和能用是两回事——物体可能和桌面穿插着生成、碰撞体退化、或者被别的
东西整个挡在后面，这些情况下声明值照样是 1.0。

这里调 harness/interaction_probe.mjs，在无头物理里真的抓一遍、拖一遍、松手一遍，把
结果并回完整度报告。探针不开浏览器、不加载 GLB，所以每条片段跑完顺手量一次不贵。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def measure(game_dir: Path, out_dir: Path, harness: str,
            style_swap: bool = False, timeout: int = 900,
            behaviour_swap: bool = False) -> dict:
    """跑一遍探针。跑不起来就返回 {"ok": False}，不让它挡住整条链路。"""
    target = Path(out_dir) / "interaction_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [harness, "interaction_probe.mjs", "--game", str(game_dir), "--out", str(target)]
    if style_swap:
        cmd.append("--style-swap")
    if behaviour_swap:
        cmd.append("--behaviour-swap")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        report = json.loads(target.read_text(encoding="utf-8"))
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return {"ok": False, "reason": type(error).__name__}
    report["ok"] = True
    return report


def merge(interactivity: dict, probe: dict) -> dict:
    """把量出来的几个数并进可交互程度那一轴，键名都带 measured_ 前缀。

    声明值保留不动。两个都在，差距本身才是有信息的那一项：声明 1.0、实测 0.44，
    说的是「程序以为十六个物体都能拖，实际只有七个拖得动」。
    """
    out = dict(interactivity)
    if not probe.get("ok"):
        out["measured"] = probe.get("reason", "没跑成")
        return out
    s = probe.get("summary") or {}
    out["measured_declared_grabbable"] = s.get("declared_grabbable")
    out["measured_works"] = s.get("works")
    # 声明能抓的物体里，真的抓得动的比例。这是「可交互程度」该报的那个数。
    out["measured_success_fraction"] = s.get("manipulation_success_fraction")
    # 整个场景里有多少物体是真能动的
    out["measured_interactive_fraction"] = s.get("interactive_fraction")
    # 从开场那台相机看过去，平均有多少面积点得到。点不到就拖不到
    out["measured_clickable_fraction"] = s.get("median_clickable_fraction")
    # 拖一个物体把旁边的东西带出去多远。生成时互相穿插的话这个数会很大
    out["measured_disturbance"] = s.get("median_disturbance")
    out["measured_fell_through"] = s.get("fell_through")
    swap = probe.get("style_swap")
    if swap:
        # 论文第四条主张的那个数：换一套完全不同的外形网格，同一串操作的轨迹差多少。
        # 碰撞体从基本体那一套算（kernel/scene.js 的 buildNode），所以应该是 0。
        out["style_swap_max_deviation"] = swap.get("max_position_deviation")
        out["style_swap_steps"] = swap.get("compared_steps")
    back = probe.get("behaviour_swap")
    if back:
        # 反过来那一半：改运动、改质量、改能不能抓，样式应该一个字段都不变。
        # 两个方向都成立，「分开写的两半」才不是一句口号。
        out["behaviour_swap_changed_nodes"] = back.get("differing_nodes")
        out["behaviour_swap_nodes"] = back.get("compared_nodes")
    return out
