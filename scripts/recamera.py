"""给已有的真值程序换一套相机，场景内容一个字不动。

为什么要这个。现在整条链路默认「一台手持相机、站着拍、把整个关卡装进画面」，
而这个取景是写死在生成器里的（`sample_camera`：半径 = 场地跨度 × 0.62 + 2）。
实测的后果：在 VGGT 的 518 px 输入里，物体掩码中位只有 **116 个像素、
包围盒短边 9 px**。再被 `unproject(stride=4)` 除以 16，逐帧中位只剩 7.2 个点，
于是 30 点的闸把 79.5% 的关节物体挡在管线外——「关节信号太弱」这个结论，
是在 9 像素宽的物体上得出来的。

光提高渲染分辨率没用：VGGT 输入封顶 518 px（`gt_backend.py:28`），现在已经顶到了。
唯一能让物体多占像素的办法是改取景，也就是把相机挪近、或者多放几台。

这个脚本只重写 `program.json` 的 `camera` 字段。场景、物体、运动、真值全部不变，
所以「换相机」是个干净的单变量对照：同一批场景、同一套答案，只有观测方式不同。

三种取景，都标了用到哪些真值信息：
  near_orbit     把现有轨迹整体朝注视点压近。不读场景内容，零泄漏。
  walkthrough    低空穿行，路点取自物体 t=0 的位置。只用首帧位置。
  object_centric 盯住某个物体 t=0 的位置转一圈，距离按目标像素占比反算。
                 只用首帧位置——**不跟踪运动**，免得相机轨迹把「谁在动」泄给模型。

多视角是免费的：录制器的 `seek(t)` 是确定性的，同一个程序用不同相机渲两遍，
两段视频逐帧时间戳对齐，天然就是同步多目。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

FLOOR_M = 0.25          # 相机不许低于这个高度，否则会钻到地板下面
MIN_DIST_M = 0.35       # 离注视点太近会糊成一片，也容易穿进物体里


def _sub(a: list, b: list) -> list:
    return [a[i] - b[i] for i in range(3)]


def _add(a: list, b: list) -> list:
    return [a[i] + b[i] for i in range(3)]


def _mul(a: list, k: float) -> list:
    return [v * k for v in a]


def _norm(a: list) -> float:
    return math.sqrt(sum(v * v for v in a))


def _round3(a: list) -> list:
    return [round(float(v), 3) for v in a]


def object_anchors(program: dict) -> list[dict]:
    """每个可动物体在 t=0 的位置和尺寸。只读首帧，不碰 motion 的后续关键帧。"""
    out = []
    for node in (program.get("objects") or []):
        pose = node.get("pose") or {}
        pos = pose.get("pos")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        geom = node.get("geom") or {}
        ext = geom.get("extent") or geom.get("size")
        if isinstance(ext, list) and len(ext) == 3:
            size = max(float(v) for v in ext)
        else:
            size = float(geom.get("radius") or geom.get("height") or 0.3) * 2
        out.append({"id": node.get("id"), "pos": [float(v) for v in pos], "size": max(size, 0.05)})
    return out


def distance_for_fraction(size_m: float, fov_deg: float, frac: float) -> float:
    """要让 size_m 的物体占满画面高度的 frac，相机该站多远。

    针孔模型：画面在距离 d 处的高度是 2*d*tan(vfov/2)，物体占 frac 就是
    size = frac * 2*d*tan(vfov/2)。反解 d。
    """
    half = math.tan(math.radians(fov_deg) / 2.0)
    return max(MIN_DIST_M, size_m / (2.0 * half * max(frac, 1e-3)))


def _lift(pos: list) -> list:
    return [pos[0], max(pos[1], FLOOR_M), pos[2]]


def near_orbit(camera: dict, k: float) -> dict:
    """沿每个关键帧的视线把相机朝注视点拉近到 k 倍。轨迹形状不变。"""
    out = json.loads(json.dumps(camera))
    for kf in out.get("keyframes") or []:
        look = kf.get("look_at") or [0.0, 0.0, 0.0]
        pos = kf["pos"]
        v = _sub(pos, look)
        d = _norm(v)
        if d < 1e-6:
            continue
        nd = max(MIN_DIST_M, d * k)
        kf["pos"] = _round3(_lift(_add(look, _mul(v, nd / d))))
    return out


def walkthrough(camera: dict, anchors: list[dict], rng: random.Random,
                duration: float, count: int, height: float, standoff: float) -> dict:
    """低空穿行：依次路过各个物体，看向下一个路点。只用 t=0 的位置。"""
    out = json.loads(json.dumps(camera))
    if not anchors:
        return out
    order = list(range(len(anchors)))
    rng.shuffle(order)
    stops = [anchors[i]["pos"] for i in order]
    # 路点不落在物体正中心，退开 standoff，免得相机穿模
    way = []
    for i, p in enumerate(stops):
        ang = rng.uniform(0, 2 * math.pi)
        way.append([p[0] + standoff * math.cos(ang), height, p[2] + standoff * math.sin(ang)])
    if len(way) == 1:
        ang = rng.uniform(0, 2 * math.pi)
        way.append([way[0][0] + standoff * 2 * math.cos(ang), height,
                    way[0][2] + standoff * 2 * math.sin(ang)])
    kfs = []
    for i in range(count):
        u = i / (count - 1) if count > 1 else 0.0
        s = u * (len(way) - 1)
        j = min(int(s), len(way) - 2)
        f = s - j
        pos = [way[j][m] + (way[j + 1][m] - way[j][m]) * f for m in range(3)]
        look_i = min(j + 1, len(stops) - 1)
        kfs.append({"t": round(duration * u, 3), "pos": _round3(_lift(pos)),
                    "look_at": _round3(stops[look_i])})
    out["keyframes"] = kfs
    return out


def object_centric(camera: dict, anchor: dict, rng: random.Random, duration: float,
                   count: int, frac: float) -> dict:
    """围着某个物体 t=0 的位置转一小段，距离按目标像素占比反算。不跟踪运动。"""
    out = json.loads(json.dumps(camera))
    fov = float((out.get("intrinsics") or {}).get("fov_deg") or 60.0)
    d = distance_for_fraction(anchor["size"], fov, frac)
    centre = anchor["pos"]
    start = rng.uniform(0, 2 * math.pi)
    sweep = rng.choice([-1, 1]) * rng.uniform(0.5, 1.1) * math.pi
    rise = rng.uniform(0.25, 0.75) * d
    kfs = []
    for i in range(count):
        u = i / (count - 1) if count > 1 else 0.0
        a = start + sweep * u
        pos = [centre[0] + d * math.cos(a), centre[1] + rise, centre[2] + d * math.sin(a)]
        kfs.append({"t": round(duration * u, 3), "pos": _round3(_lift(pos)),
                    "look_at": _round3(centre)})
    out["keyframes"] = kfs
    return out


def build_views(program: dict, mode: str, views: int, seed: int, *,
                closeness: float, frac: float, count: int,
                height: float, standoff: float) -> list[dict]:
    camera = program.get("camera") or {}
    duration = float((program.get("meta") or {}).get("duration") or 8.0)
    anchors = object_anchors(program)
    rng = random.Random(seed)
    out = []
    for v in range(views):
        r = random.Random(rng.randrange(1 << 30))
        if mode == "near_orbit":
            # 多视角时每个视角再转一个起始角，否则 K 个视角完全重合
            cam = near_orbit(camera, closeness)
            if v and cam.get("keyframes"):
                phi = 2 * math.pi * v / views
                for kf in cam["keyframes"]:
                    look = kf.get("look_at") or [0.0, 0.0, 0.0]
                    dx, dz = kf["pos"][0] - look[0], kf["pos"][2] - look[2]
                    kf["pos"] = _round3(_lift([
                        look[0] + dx * math.cos(phi) - dz * math.sin(phi),
                        kf["pos"][1],
                        look[2] + dx * math.sin(phi) + dz * math.cos(phi)]))
            out.append(cam)
        elif mode == "walkthrough":
            out.append(walkthrough(camera, anchors, r, duration, count, height, standoff))
        elif mode == "object_centric":
            if not anchors:
                out.append(near_orbit(camera, closeness)); continue
            out.append(object_centric(camera, anchors[v % len(anchors)], r, duration, count, frac))
        else:
            raise SystemExit(f"不认识的取景模式：{mode}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="源数据集根目录，下面是一堆 <clip>/program.json")
    ap.add_argument("--out", required=True, help="输出根目录，每个视角一个 <clip>_v<k> 目录")
    ap.add_argument("--mode", default="near_orbit",
                    choices=["near_orbit", "walkthrough", "object_centric"])
    ap.add_argument("--views", type=int, default=1, help="每条片段出几个视角")
    ap.add_argument("--closeness", type=float, default=0.35,
                    help="near_orbit：距离压到原来的几倍")
    ap.add_argument("--frac", type=float, default=0.3,
                    help="object_centric：物体占画面高度的比例")
    ap.add_argument("--keyframes", type=int, default=9,
                    help="新轨迹的关键帧数。相机走得近，轨迹要更密才不会在插值里穿墙")
    ap.add_argument("--height", type=float, default=0.9, help="walkthrough 的行进高度")
    ap.add_argument("--standoff", type=float, default=0.8, help="walkthrough 路点离物体多远")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261025)
    a = ap.parse_args(argv)

    src, out = Path(a.src), Path(a.out)
    clips = sorted(p for p in src.iterdir() if (p / "program.json").is_file())
    if a.limit:
        clips = clips[:a.limit]
    if not clips:
        print(f"{src} 下面没有带 program.json 的片段"); return 1
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for c in clips:
        program = json.loads((c / "program.json").read_text(encoding="utf-8"))
        # 不能用内置 hash：字符串哈希逐进程随机，同一条命令跑两遍结果不一样
        digest = hashlib.sha256(c.name.encode()).digest()
        seed = a.seed ^ int.from_bytes(digest[:4], "big")
        cams = build_views(program, a.mode, a.views, seed, closeness=a.closeness,
                           frac=a.frac, count=a.keyframes, height=a.height,
                           standoff=a.standoff)
        for v, cam in enumerate(cams):
            d = out / (f"{c.name}_v{v}" if a.views > 1 else c.name)
            d.mkdir(parents=True, exist_ok=True)
            p = json.loads(json.dumps(program))
            p["camera"] = cam
            # 片段名要跟着变，否则评测按 clip 名找真值时 K 个视角会互相覆盖
            p.setdefault("meta", {})["clip"] = d.name
            p["meta"]["recamera"] = {"mode": a.mode, "view": v, "source_clip": c.name}
            (d / "program.json").write_text(json.dumps(p, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
            written += 1
    # 顺手写一份 manifest，render_dataset.py 要靠它列片段。少了它这个目录渲不了，
    # 而「渲不了」会等到提交作业之后才暴露。
    # duration 和 fps 必须来自源数据集。悄悄退回默认值 8.0/30 的话，
    # 一旦源头不是这两个数，渲出来的片段长度就和真值对不上，而且不会报错。
    src_manifest, found = None, None
    for cand in (src / "manifest.json",
                 src.parent / src.name.replace("_rendered", "") / "manifest.json"):
        if cand.is_file():
            src_manifest, found = json.loads(cand.read_text(encoding="utf-8")), cand
            break
    if src_manifest is None or "duration_s" not in src_manifest or "fps" not in src_manifest:
        print(f"找不到源数据集的 manifest（试过 {src}/manifest.json 和 "
              f"{src.parent / src.name.replace('_rendered', '')}/manifest.json），"
              f"duration 和 fps 没法确定，不猜。", file=sys.stderr)
        return 1
    names = sorted(d.name for d in out.iterdir() if (d / "program.json").is_file())
    (out / "manifest.json").write_text(json.dumps({
        "split": "recamera", "seed": a.seed,
        "duration_s": src_manifest["duration_s"],
        "fps": src_manifest["fps"],
        "source_manifest": str(found),
        "recamera": {"mode": a.mode, "views": a.views, "closeness": a.closeness,
                     "frac": a.frac, "keyframes": a.keyframes, "source": str(src)},
        "clips": [{"clip": n} for n in names],
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"写了 {written} 份程序，来自 {len(clips)} 条片段，模式 {a.mode}，每条 {a.views} 个视角")
    print(f"manifest：{len(names)} 条，duration {src_manifest['duration_s']}s fps {src_manifest['fps']}（读自 {found}）")
    print(f"输出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
