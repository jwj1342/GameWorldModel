"""把三路采样装配成一份真值程序。

三路各有自己的随机源：**行为**（场景结构、几何、运动）、**样式**（背景、光照、材质）、
**观测**（相机轨迹）。三个种子独立给，所以可以固定两路只变一路——
「同行为不同样式」这组配对就是核心主张第四条在数据这一侧的证据，
也是 #13 可编辑性实验的标准答案：一次只改样式，应该且只应该改到 style 子树和 material。

相机四档里没有「固定俯视」。RP 5.8 原本列了这一档，但机位不动就没有基线，
而合成数据必须跑一遍真实感知管线，照抽不误会产出一批建立在没有基线位姿上的训练样本。
这里把它换成 `overhead_dolly`：仍然是俯视，但机位沿直线平移，基线有了，俯视这个因子保住了。
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

from .behavior import (BINDING_CLASSES, TEST_FAMILIES, TRAIN_FAMILIES,
                       sample_layout, sample_objects)
from .style import sample_material, sample_style

CAMERA_TRACKS = ("handheld", "follow", "overhead_dolly", "orbit")


@dataclass(frozen=True)
class SceneSpec:
    clip: str
    family: str
    behavior_seed: int
    style_seed: int
    camera_seed: int
    profile: str = "test"                 # test = 八类等权；train = 按常见度配
    duration: float = 8.0
    fps: int = 30
    objects: int = 0                      # 0 = 按家族随机抽 1..8
    camera_track: str = ""                # 空 = 按 camera_seed 抽
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def with_style(self, style_seed: int, clip: str) -> "SceneSpec":
        """同行为、同观测，只换样式。"""
        return SceneSpec(clip=clip, family=self.family, behavior_seed=self.behavior_seed,
                         style_seed=style_seed, camera_seed=self.camera_seed, profile=self.profile,
                         duration=self.duration, fps=self.fps, objects=self.objects,
                         camera_track=self.camera_track, notes=self.notes,
                         tags=tuple(self.tags) + ("style_variant",))

    def with_behavior(self, behavior_seed: int, clip: str) -> "SceneSpec":
        """同样式、同观测，只换行为。"""
        return SceneSpec(clip=clip, family=self.family, behavior_seed=behavior_seed,
                         style_seed=self.style_seed, camera_seed=self.camera_seed, profile=self.profile,
                         duration=self.duration, fps=self.fps, objects=self.objects,
                         camera_track=self.camera_track, notes=self.notes,
                         tags=tuple(self.tags) + ("behavior_variant",))


def sample_camera(rng: random.Random, layout: dict, duration: float, track: str) -> dict:
    span = max(layout["width"], layout["depth"])
    radius = span * 0.62 + 2.0
    height = round(span * 0.28 + 1.8, 2)
    count = 5
    keyframes = []
    if track == "orbit":
        start = rng.uniform(0, 2 * math.pi)
        sweep = rng.choice([-1, 1]) * rng.uniform(0.6, 1.6) * math.pi
        for i in range(count):
            u = i / (count - 1)
            angle = start + sweep * u
            keyframes.append({"t": round(duration * u, 3),
                              "pos": [round(radius * math.cos(angle), 2), height, round(radius * math.sin(angle), 2)],
                              "look_at": [0.0, 1.0, 0.0]})
    elif track == "overhead_dolly":
        high = round(span * 0.85 + 4.0, 2)
        axis = rng.choice(["x", "z"])
        travel = span * 0.45
        for i in range(count):
            u = i / (count - 1)
            offset = round(-travel / 2 + travel * u, 2)
            pos = [offset, high, round(rng.uniform(-0.6, 0.6), 2)] if axis == "x" else [round(rng.uniform(-0.6, 0.6), 2), high, offset]
            keyframes.append({"t": round(duration * u, 3), "pos": pos,
                              "look_at": [pos[0] * 0.25, 0.0, pos[2] * 0.25]})
    elif track == "follow":
        start_angle = rng.uniform(0, 2 * math.pi)
        for i in range(count):
            u = i / (count - 1)
            angle = start_angle + 0.5 * math.pi * u
            r = radius * (0.75 + 0.25 * u)
            keyframes.append({"t": round(duration * u, 3),
                              "pos": [round(r * math.cos(angle), 2), round(height * 0.7, 2), round(r * math.sin(angle), 2)],
                              "look_at": [round(rng.uniform(-1.5, 1.5), 2), 1.0, round(rng.uniform(-1.5, 1.5), 2)]})
    else:                                   # handheld：缓慢平移叠上小幅抖动
        start_angle = rng.uniform(0, 2 * math.pi)
        base = [radius * math.cos(start_angle), height * 0.8, radius * math.sin(start_angle)]
        # 平移量按「方向 + 模长」抽，不按三个分量各自抽：后者会抽出模长接近零的那一组，
        # 也就是一台几乎不动的手持相机，那又回到了没有基线的老问题上。
        drift_angle, drift_len = rng.uniform(0, 2 * math.pi), rng.uniform(2.5, 5.0)
        drift = [drift_len * math.cos(drift_angle), rng.uniform(-0.6, 0.9), drift_len * math.sin(drift_angle)]
        for i in range(count):
            u = i / (count - 1)
            pos = [round(base[k] + drift[k] * u + rng.uniform(-0.22, 0.22), 2) for k in range(3)]
            keyframes.append({"t": round(duration * u, 3), "pos": pos,
                              "look_at": [round(rng.uniform(-0.8, 0.8), 2), 1.0, round(rng.uniform(-0.8, 0.8), 2)]})
    return {"intrinsics": {"fov_deg": round(rng.uniform(42, 72), 1), "aspect": 1.7778,
                           "far": round(span * 4 + 40, 1)},
            "keyframes": keyframes, "interp": "catmull_rom"}


def generate_scene(spec: SceneSpec) -> tuple[dict, dict]:
    """返回 (真值程序, 生成事实)。

    生成事实**不进程序**。程序是模型要预测的答案，把家族名、相机档、profile 写进
    `meta.notes` 等于在答案里留小抄——模型会学着把它背出来，而真实视频上没有这一行。
    这些事实只进 manifest。
    """
    behavior_rng = random.Random(spec.behavior_seed)
    camera_rng = random.Random(spec.camera_seed)
    # 样式这一路要拆成两个独立的流：材质的抽取次数等于节点数，而节点数是行为那边定的。
    # 共用一个流的话，只换行为种子会把光照也带着变，两半就不再分得开了。
    style_rng = random.Random(spec.style_seed)
    material_rng = random.Random(spec.style_seed ^ 0x9E3779B9)

    statics, layout = sample_layout(behavior_rng, spec.family)
    count = spec.objects or behavior_rng.randint(1, 8)
    objects = sample_objects(behavior_rng, layout, spec.profile, spec.duration, count)

    track = spec.camera_track or camera_rng.choice(CAMERA_TRACKS)
    camera = sample_camera(camera_rng, layout, spec.duration, track)

    # 样式最后刷上去：行为那一半已经定死，这里只写 material 和 style 子树。
    # 结构件压住饱和度也不给自发光：会发光的地板和洋红的墙只会白白拉大合成到真实的差距。
    for node in statics:
        node["material"] = sample_material(material_rng, muted=True, emissive_allowed=False)
    for node in objects:
        node["material"] = sample_material(material_rng)
    meta = {"clip": spec.clip, "fps": spec.fps, "duration": spec.duration, "units": "m", "up": "y"}
    if spec.notes:
        meta["notes"] = spec.notes
    program = {
        "meta": meta,
        "style": sample_style(style_rng),
        "camera": camera,
        "static": statics,
        "objects": objects,
    }
    return program, {"camera_track": track, "family": spec.family, "profile": spec.profile}


def generate(spec: SceneSpec) -> dict:
    """只要程序。同一个 spec 永远产出同一份程序，三路随机源互不影响。"""
    return generate_scene(spec)[0]


def describe(program: dict, spec: SceneSpec, facts: dict) -> dict:
    """写进 manifest 的那一行。全是能从程序本身算出来的事实，不含任何估计量。"""
    motions = [(o.get("motion") or {}).get("type", "static") for o in program["objects"]]
    classes = [o.get("class", "") for o in program["objects"]]
    materials = [n.get("material") for n in program["static"] + program["objects"]]
    forms = {"palette": 0, "hex": 0, "surface": 0}
    for m in materials:
        forms["surface" if isinstance(m, dict) else ("hex" if isinstance(m, str) and m.startswith("#") else "palette")] += 1
    return {
        "clip": spec.clip, "family": facts["family"], "profile": facts["profile"],
        "camera_track": facts["camera_track"],
        "behavior_seed": spec.behavior_seed, "style_seed": spec.style_seed, "camera_seed": spec.camera_seed,
        "duration_s": spec.duration, "fps": spec.fps,
        "object_count": len(program["objects"]), "static_count": len(program["static"]),
        "motion_types": motions,
        "material_forms": forms,
        # #5 点名的「自己跟自己对答案」：类名落在绑定词表里的比例，报出来而不是埋着
        "binding_vocab_hits": sum(1 for c in classes if c in BINDING_CLASSES),
        "tags": list(spec.tags),
    }


def style_half(program: dict) -> dict:
    """样式那一半：`style` 子树加每个节点的 `material`。"""
    return {"style": program.get("style"),
            "materials": {node["id"]: node.get("material")
                          for group in ("static", "objects") for node in program.get(group, [])}}


def behavior_half(program: dict) -> dict:
    """行为那一半：结构、几何、运动、相机——也就是除样式之外的全部。

    两半的定义只写在这里一处。配对测试要用它，#52 的解耦度指标也要用它；
    各写一份的话，以后加一个样式字段，另一边会悄悄把它算进行为里。
    """
    return {
        "static": [{k: v for k, v in node.items() if k != "material"} for node in program.get("static", [])],
        "objects": [{k: v for k, v in node.items() if k != "material"} for node in program.get("objects", [])],
        "camera": program.get("camera"),
    }


def family_group(family: str) -> str:
    if family in TRAIN_FAMILIES:
        return "train"
    if family in TEST_FAMILIES:
        return "test"
    raise ValueError(f"unknown scene family: {family}")
