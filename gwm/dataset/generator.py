"""把三路采样装配成一份真值程序。

三路各有自己的随机源：**行为**（场地上摆什么、怎么动）、**样式**（背景、光照、材质）、
**观测**（场地尺度与相机轨迹）。三个种子独立给，所以可以固定两路只变一路——
「同行为不同样式」这组配对就是核心主张第四条在数据这一侧的证据，
也是 #13 可编辑性实验的标准答案：一次只改样式，应该且只应该改到 style 子树和 material。

相机四档里没有「固定俯视」。RP 5.8 原本列了这一档，但机位不动就没有基线，
而合成数据必须跑一遍真实感知管线，照抽不误会产出一批建立在没有基线位姿上的训练样本。
这里把它换成 `overhead_dolly`：仍然是俯视，但机位沿直线平移，基线有了，俯视这个因子保住了。
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field, replace

from .framing import scale_program
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
    stage_scale: float = 1.0              # 整体相似缩放。1.0 = 关卡尺度（原样）；
                                          # 0.15 左右把场景缩到房间/桌面尺度，相机高度
                                          # 随之落进手持的范围，和链路的标定假设对上。
                                          # 见 gwm/dataset/framing.py 和 #68
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    # 真值场景用资产库搭（见 behavior._asset_geom）。给配置字典，里面要有
    # assets.library 和 assets.shape_cache；留空或库不可用就退回基本体，行为不变。
    #
    # 为什么需要：现在的真值是「随机名字 + 独立挑的基本体」，于是「该挑哪个资产」
    # 没有标准答案（检索改进测不出来——实测换掉 64% 的资产，所有指标逐字节不动），
    # 而且类名和形状脱钩导致 class_accuracy 恒等于噪声。真值改从同一个库里挑，
    # 两件事一起修好。
    asset_cfg: dict | None = None

    # 两个派生方法都用 `dataclasses.replace`，**不要逐字段重建**。
    #
    # 原来是逐字段抄的，于是 `stage_scale` 被漏掉、退回默认的 1.0——基准片段缩到了
    # 0.15，派生出来的配对没缩，两边差 6.667 倍。后果是 `style_variant` 这组
    # 「同行为、只换样式」的不变量**根本不成立**，而 #52 的整个指标就建立在这个
    # 不变量上。实测六组 style_variant 行为全不同，一组都用不了。
    #
    # 逐字段重建是这类 bug 的温床：以后往 SceneSpec 加字段，一样会被静默漏掉，
    # 而且不报错。replace 只改要改的，其余自动带上。
    def with_style(self, style_seed: int, clip: str) -> "SceneSpec":
        """同行为、同观测，只换样式。"""
        return replace(self, clip=clip, style_seed=style_seed,
                       tags=tuple(self.tags) + ("style_variant",))

    def with_behavior(self, behavior_seed: int, clip: str) -> "SceneSpec":
        """同样式、同观测，只换行为。"""
        return replace(self, clip=clip, behavior_seed=behavior_seed,
                       tags=tuple(self.tags) + ("behavior_variant",))


def _material_rng(style_seed: int, node_id: str) -> random.Random:
    """某个节点的材质流。字符串种子走的是 sha512，跨平台跨进程都给同一串数。"""
    return random.Random(f"{style_seed}:{node_id}")


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
    # 场地尺度归在观测这一路：相机的取景距离是它的函数，同源才能做到只换行为时机位不动。
    stage_rng = random.Random(spec.camera_seed ^ 0x51ED270B)
    style_rng = random.Random(spec.style_seed)

    statics, layout = sample_layout(behavior_rng, spec.family, stage_rng)
    count = spec.objects or behavior_rng.randint(1, 8)
    objects = sample_objects(behavior_rng, layout, spec.profile, spec.duration, count,
                             cfg=spec.asset_cfg)

    track = spec.camera_track or camera_rng.choice(CAMERA_TRACKS)
    camera = sample_camera(camera_rng, layout, spec.duration, track)

    # 样式最后刷上去：行为那一半已经定死，这里只写 material 和 style 子树。
    # 每个节点的材质各起一个由「样式种子 + 节点 id」决定的流，而不是顺着一条流往下抽。
    # 顺着抽的话抽取次序取决于节点数，而节点数是行为那边定的——只换行为种子，同一个
    # obj_3 会拿到另一种材质，样式那一半就跟着行为动了。
    for node in statics:
        node["material"] = sample_material(_material_rng(spec.style_seed, node["id"]),
                                           muted=True, emissive_allowed=False)
    for node in objects:
        node["material"] = sample_material(_material_rng(spec.style_seed, node["id"]))
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
    # 整体相似缩放放在最后一步：生成逻辑一点不用改，缩放只动带长度量纲的字段。
    # 场景内部的相对结构不变，变的只是它和「一米二高的手持相机」之间的比例。
    if spec.stage_scale != 1.0:
        program = scale_program(program, spec.stage_scale)
    return program, {"camera_track": track, "family": spec.family, "profile": spec.profile,
                     "stage_scale": spec.stage_scale}


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
