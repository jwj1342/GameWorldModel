"""Evidence -> program without any VLM (baseline, ablation, and fallback when generation fails)."""
from __future__ import annotations
import math
from ..perception.ground_proximity import is_geometric_proximity
import numpy as np
from ..perception.contract import format_evidence_errors
from ..perception.quality import assess_evidence_quality
from ..perception.color import measured_color

from ..taxonomy import is_ground_class, is_rigid_structure, is_static_class, slug

def guess_material(cls: str) -> str:
    c = cls.lower()
    for key, mat in [("coin", "gold"), ("gold", "gold"), ("box", "cardboard"), ("carton", "cardboard"), ("train", "red"), ("car", "red"), ("door", "wood"), ("table", "wood"),
                     ("track", "plastic"), ("rail", "plastic"), ("belt", "rubber"), ("conveyor", "metal"), ("ball", "blue"), ("wall", "stone"), ("floor", "concrete")]:
        if key in c: return mat
    return "default"

def material_from_evidence(obj: dict):
    color = measured_color(obj.get("appearance"))
    palette = guess_material(obj["class_guess"])
    if not color:
        return palette
    # Preserve the existing class prior, not a measured reflectance estimate.
    # Parity with kernel/scene.js is checked by the material regression test.
    return {**PALETTE_SURFACES[palette], "base_color": color}


PALETTE_SURFACES = {
    "default": {"roughness": 0.8}, "grass": {"roughness": 0.95},
    "stone": {"roughness": 0.9}, "wood": {"roughness": 0.8},
    "metal": {"roughness": 0.4, "metalness": 0.6},
    "concrete": {"roughness": 0.95}, "plastic": {"roughness": 0.5},
    "rubber": {"roughness": 1.0}, "gold": {"roughness": 0.3, "metalness": 0.8},
    "red": {"roughness": 0.6}, "blue": {"roughness": 0.6},
    "green": {"roughness": 0.6}, "yellow": {"roughness": 0.6},
    "white": {"roughness": 0.7}, "black": {"roughness": 0.7},
    "lava": {"emissive": "#ff2200", "emissive_intensity": 0.8, "roughness": 1.0},
    "water": {"roughness": 0.2, "metalness": 0.1, "opacity": 0.8},
    "glass": {"roughness": 0.1, "opacity": 0.4},
    "cardboard": {"roughness": 0.9},
}

def _confidence(value):
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else None

def motion_from_guess(mg: dict, obbs: list[dict]) -> dict | None:
    t = mg.get("type", "unknown")
    if t in ("static", "unknown") or len(obbs) < 2: return None
    ts = [o["t"] for o in obbs]; cs = np.asarray([o["center"] for o in obbs])
    if t == "periodic_translate":
        # kernel: c(t) = base + axis*amp*sin(2pi t/T + phase); base must be the mean centre; pick the phase branch matching the initial velocity
        ax = np.asarray(mg["axis"]); mean = cs.mean(0); proj = (cs - mean) @ ax; amp = max(mg.get("amp") or 0.1, 1e-3); T = float(mg["period"])
        phi = math.asin(max(-1, min(1, proj[0] / amp)))
        v0 = float(proj[min(2, len(proj) - 1)] - proj[0])
        if math.cos(phi) * v0 < 0: phi = math.pi - phi
        phase = phi - 2 * math.pi * ts[0] / T
        return {"type": "periodic_translate", "axis": [float(v) for v in ax], "amp": float(amp), "period": T, "phase": float(phase), "_base": [float(v) for v in mean]}
    if t == "periodic_rotate":
        return {"type": "periodic_rotate", "axis": [0, 1, 0], "amp_deg": float(mg.get("amp_deg") or 30), "period": float(mg.get("period") or max(ts[-1] - ts[0], 1.0) * 2), "phase": float(mg.get("phase") or 0.0)}
    if t == "spin":
        return {"type": "spin", "axis": [0, 1, 0], "rate_dps": float(mg.get("rate") or 90)}
    if t == "revolute" and mg.get("pivot"):
        rng = mg.get("range") or [0, 90]
        return {"type": "revolute", "axis": [0, 1, 0], "pivot": [float(v) for v in mg["pivot"]], "range_deg": [0.0, float(max(rng[1], 5))],
                "schedule": [{"t": float(ts[0]), "to_deg": float(max(rng[1], 5)), "duration": float(max(ts[-1] - ts[0], 0.5))}]}
    # prismatic / trajectory: keyframes (subsample to <= 12)
    idx = np.unique(np.round(np.linspace(0, len(obbs) - 1, min(12, len(obbs)))).astype(int))
    return {"type": "trajectory", "interp": "linear", "keyframes": [{"t": float(ts[i]), "pos": [float(v) for v in cs[i]]} for i in idx]}

def observed_trajectory(obbs: list[dict], min_span_frac: float) -> dict | None:
    """判不出运动模型，但看得出它动了——那就把**观测到的轨迹**直接写进去。

    这就是可交互性第一档「观测运动」：不需要知道它是 spin 还是 periodic_rotate，
    只需要让它在场景里按视频里走过的路再走一遍。

    判据是**位移相对物体自身对角线**，不是绝对米数。实测（十条真值感知片段、
    moves=True 的 57 个物体）：

      真会动的   整段最大位移中位 0.402 m，合自身对角线的 **1.71 倍**
      误报的     0.055 m，合 **0.23 倍**

    四分位几乎不重叠。按对角线比例扫阈值：

      0.00 倍  留下 57  命中率 0.649  查全 0.902
      0.40 倍  留下 38  命中率 0.868  查全 0.805   <- 默认
      1.00 倍  留下 21  命中率 0.952  查全 0.488

    用相对量不用绝对米数，是因为绝对阈值不是尺度不变的——这个坑这个项目踩过两次
    （`max_dist_m`、`static_pos_thresh_m`），不想再踩第三次。
    """
    if len(obbs) < 3:
        return None
    centres = np.asarray([o["center"] for o in obbs], float)
    span = float(np.linalg.norm(centres - centres[0], axis=1).max())
    diag = float(np.linalg.norm(np.median([o["size"] for o in obbs], axis=0)))
    if diag <= 1e-6 or span < min_span_frac * diag:
        return None
    ts = [o["t"] for o in obbs]
    idx = np.unique(np.round(np.linspace(0, len(obbs) - 1, min(12, len(obbs)))).astype(int))
    return {"type": "trajectory", "interp": "linear",
            "keyframes": [{"t": float(ts[i]), "pos": [float(v) for v in centres[i]]} for i in idx]}


def recentre_periodic(node: dict, obb: list[dict]) -> None:
    """periodic_* 的 pose.pos 必须是振荡中心（见 prompts/common_dsl.md）。模型常给成首帧位置，
    那样整条轨迹会偏掉最多一个振幅，这里按观测中心改回来。"""
    m = node.get("motion") or {}
    if m.get("type") not in ("periodic_translate", "periodic_rotate") or not obb: return
    node.setdefault("pose", {})["pos"] = [float(v) for v in np.asarray([b["center"] for b in obb]).mean(0)]

def evidence_to_program(ev: dict, clip: str | None = None, cfg: dict | None = None) -> dict:
    quality_report = assess_evidence_quality(ev, cfg)
    if quality_report["decision"] == "block":
        raise ValueError("invalid evidence:\n" + format_evidence_errors(quality_report["validation"]))
    ev = quality_report["evidence"]
    meta = ev["meta"]; cam = ev["camera"]; g = ev["static"]["planes"][0] if ev["static"]["planes"] else None
    program = {"meta": {"clip": clip or meta["clip"], "fps": 30, "duration": float(max(meta["duration"], 1.0)), "units": "m", "up": "y", "notes": "direct translation of evidence (no VLM)"},
               "style": {"background": "#8fb3d9"},
               "camera": {"intrinsics": {"fov_deg": float(cam["intrinsics"].get("fov_deg", 60)), "aspect": cam["intrinsics"]["width"] / cam["intrinsics"]["height"], "far": 100},
                          "keyframes": [{"t": p["t"], "pos": p["pos"], "quat": p["quat"]} for p in cam["poses"][::max(1, len(cam["poses"]) // 24)]], "interp": "catmull_rom"},
               "static": [], "objects": [], "binding": {"template": "platformer_3p", "slots": {}}, "residual": None}
    program["meta"]["evidence_quality"] = {"decision": quality_report["decision"],
                                             "diagnostic_codes": sorted({item["code"] for item in quality_report["diagnostics"]})}
    if g:
        ext = g.get("extent_hint") or [10, 0.2, 10]; c = g.get("center_hint") or [0, 0, 0]
        ground_node = {"id": "ground", "class": "ground", "geom": {"kind": "primitive", "shape": "box", "extent": [float(max(ext[0], 2)) * 1.4, 0.3, float(max(ext[2], 2)) * 1.4]}, "pose": {"pos": [float(c[0]), -0.15, float(c[2])]}, "material": "grass"}
        if _confidence(g.get("conf")) is not None: ground_node["confidence"] = _confidence(g["conf"])
        program["static"].append(ground_node)
    for o in ev["objects"]:
        if not o["obb"]: continue
        cls = o["class_guess"].lower()
        # 占画面太大的检测几乎一定是地面或墙，不是可以拿起来的东西。实测一条合成片段上
        # 检出一个占画面 67% 的「slab」和一个占 61% 的「ground」，而真值里最大的物体
        # 只占 2.1%。这种东西进了 objects 列表，既压低物体精确率，又会变成一块能用鼠标
        # 拖动的、和整个场景一样大的板子。按结构件处理，不是丢掉——大而平的东西通常
        # 确实是结构。
        area_cap = float((cfg or {}).get("synthesis", {}).get("max_object_area_frac", 0.35))
        too_large = float(o.get("mask_area_frac") or 0.0) >= area_cap
        is_ground = is_ground_class(cls)
        # 类名像结构件（墙、地板、轨道、传送带）的，只要**没有证据表明它会动**，就按
        # 结构件处理。
        #
        # 原来这里要求的是运动分类**肯定地说**它静止（`type == "static"`），那是个
        # 坏掉的默认：运动分类 85% 的时候返回 `unknown`，而这一轮量清楚了它的贡献
        # 精确为零——七个配置的准确率等于「恒答 static」常数基线，161 个配对物体里
        # 一次都没判对过任何一个会动的（#7）。实测后果：类名含 rail/block 的 61 个
        # 检测里只有 4 个满足这个与条件，其余全默认进了 objects，在评测里变成假阳性
        # ——哪怕检测本身完全正确、位置也摆对了。真值里 rail 有 45 个，全在 static。
        #
        # 换成 `moves`：它绕开七分类的两道闸单独算「会不会动」，真值感知上查全 0.925，
        # 是我们唯一验证过有信息的运动信号。语义也更对——「没有信息」不该被当成
        # 「它会动」，对一个名字叫 rail/wall/floor 的东西尤其不该。
        guess = o["motion_guess"]
        # Legacy Evidence has no `moves` field. A known motion type is positive
        # evidence too; even a conflicting binary flag must not erase it.
        # **但 `moves` 不能用在这里。** 当初把与条件换成 `moves` 的理由是它
        # 「真值感知上查全 0.925」——**只量了查全，没量精确率**，而这里需要的
        # 恰恰是精确率（「有没有**证据表明**它会动」）。实测 25 条片段的 evidence：
        #
        #     rail_*   moves=True  60/68 = **88.2%**   真值里全是 static
        #     block_*  moves=True  19/26 = 73%         真值里全是 static
        #
        # 多半是 ICP 在相机运动下无中生有解出转动。后果是结构件规则被这条假阳
        # 整个关掉：rail 进 objects、标成 kind=asset、送进资产检索——而真值里
        # 结构件是 primitive，**根本没有「该挑哪个」这个答案**。
        #
        # 所以对**名字就像结构件**的东西，只有**明确的运动类型**才算证据，
        # 二元标志不算。名字不像结构件的走原来那条，`moves` 仍然生效。
        _typed_motion = guess.get("type") in {
            "prismatic", "trajectory", "revolute", "spin", "periodic_translate", "periodic_rotate"}
        # 刚性结构件（墙/地板/轨道/方块）忽略二元的 moves，只认明确的运动类型；
        # 传送带、桌子这种「通常不动但可以动」的仍然按原来那条走。
        _moving = _typed_motion if is_rigid_structure(cls) else (
            guess.get("moves") is True or _typed_motion)
        looks_structural = is_static_class(cls, cfg) and not _moving
        if too_large or is_ground or looks_structural:
            # fixed structure: median OBB as a static box
            #
            # **位置取首帧，不取逐帧中位。** 逐帧包围盒中心不是在真值附近抖，是在
            # 整段视频里**单调漂移**——拿真值结构件（位置恒定，靶子不动）量，逐帧
            # 误差中位是 0.366 / 0.419 / 0.511 / 0.578 米，一路走高。取中位等于把
            # 估计往漂移里拖。四种估计量在真值结构件上的对照（15 个）：
            #
            #   首帧         离真值中位 0.277 m，容差 0.45 内 0.867
            #   逐帧中位     离真值中位 0.458 m，容差 0.45 内 0.467
            #   best_frame   离真值中位 0.427 m，容差 0.45 内 0.667
            #   最高 conf 帧 离真值中位 0.401 m，容差 0.45 内 0.600
            #
            # 尺寸仍然取逐帧中位：那是另一个量，而且之前量过中位更好（#64）。
            centers = np.asarray([b["center"] for b in o["obb"]], float)
            cs = (centers[0].copy() if (cfg or {}).get("synthesis", {}).get("structure_pos_first_frame", True)
                  else np.median(centers, 0))
            sz = np.median(np.asarray([b["size"] for b in o["obb"]]), 0)
            sz = [float(max(v, 0.05)) for v in sz]
            if any(k in cls for k in ("floor", "ground", "lawn", "grass", "carpet", "mat", "road")): sz[1] = min(sz[1], 0.3); cs[1] = -sz[1] / 2 + 0.0  # thin slab flush with the ground plane
            static_node = {"id": slug(o["id"]), "class": cls[:40], "geom": {"kind": "primitive", "shape": "box", "extent": sz}, "pose": {"pos": [float(v) for v in cs], "quat": [float(v) for v in o["obb"][0]["quat"]]},
                           "material": material_from_evidence(o),
                           "notes": ("structure by area: mask covers "
                                     f"{float(o.get('mask_area_frac') or 0):.0%} of the frame"
                                     if too_large else
                                     "ground-class node is always fixed structure: a floor that "
                                     "appears to move is camera motion attributed to the floor"
                                     if is_ground else "static structure from evidence")}
            if _confidence(o.get("confidence")) is not None: static_node["confidence"] = _confidence(o["confidence"])
            program["static"].append(static_node)
            continue
        # 位置和朝向取第一帧——`motion` 描述的轨迹就是从这一帧的位姿出发的，换一帧
        # 会和运动对不上。但**尺寸**不随时间变，没有理由只信一帧：实测 300 个物体里
        # 51 个（17%）的首帧最长边和逐帧中位差了四分之一以上，最狠的 test_00000
        # 的 spool_6 首帧量出 4.34 m、中位只有 0.93 m，差 4.68 倍。那一个物体会被
        # 按 4.34 m 拉伸，挡住半个场景还拖不动（#64）。
        # 同一个文件上面十几行，结构件走的就是逐帧中位，这里只是把两边拉齐。
        first = o["obb"][0]
        size = [float(max(s, 0.05)) for s in np.median(np.asarray([b["size"] for b in o["obb"]], float), 0)]
        node = {"id": slug(o["id"]), "class": o["class_guess"][:40],
                "geom": {"kind": "asset", "query": o["class_guess"][:80], "extent": size}, "pose": {"pos": [float(v) for v in first["center"]], "quat": [float(v) for v in first["quat"]]},
                "material": material_from_evidence(o), "notes": o["motion_guess"].get("notes", "")[:200]}
        if _confidence(o.get("confidence")) is not None: node["confidence"] = _confidence(o["confidence"])
        # 感知看出来的关节写进交互那一半。它和 motion 并列但不是一回事：
        # motion 说的是「视频里它自己怎么动」，interaction 说的是「用户能怎么动它」。
        art = o.get("articulation")
        if isinstance(art, dict) and art.get("grab") in ("hinge", "slide"):
            node["interaction"] = {k: v for k, v in art.items() if k != "evidence"}
        m = motion_from_guess(o["motion_guess"], o["obb"])
        if m:
            base = m.pop("_base", None)
            if base: node["pose"]["pos"] = base
            node["motion"] = m
        elif o["motion_guess"].get("moves") is True:
            # 运动模型判不出来，但「它在动」这件事判得出来，而且判得挺准（查全 0.925）。
            # 原来这两件事绑在一起——七分类过不了闸，连「它会动」也一起丢掉，节点默默
            # 变成一个静止的盒子。
            #
            # 不写一个**猜的**参数化模型（那会穿帮，也正是那两道闸存在的理由），
            # 而是把**观测到的轨迹**原样写进去：第一档要的就是「在场景里把视频里走过
            # 的路再走一遍」，它不需要知道这是 spin 还是 periodic_rotate。
            syn = (cfg or {}).get("synthesis", {})
            # 默认关着：在真值感知上这是明确的改进（会动查全 0.136 -> 0.750，
            # 物体召回和精确率也跟着涨），但在**神经感知**上是净亏——查全同样涨到
            # 0.500，可召回 0.500 -> 0.429、精确率 0.218 -> 0.207、实测可交互性
            # 0.911 -> 0.821 全都降了。差别在轨迹质量：神经轨迹带噪声，写进去等于
            # 给物体加抖动，于是配不上；而且被轨迹动画驱动的物体不跟鼠标走，
            # 第一档和第二档在这里打架。详见 configs/default.yaml 的注释和 #7。
            observed = (observed_trajectory(
                o["obb"], float(syn.get("observed_motion_min_span_frac", 0.4)))
                if syn.get("write_observed_motion", False) else None)
            if observed:
                node["motion"] = observed
                node["notes"] = "运动模型判不出来，按观测到的轨迹回放"
            else:
                node["notes"] = (f"观测到它在动，但位移太小、模型也判不出来"
                                 f"（{o['motion_guess'].get('notes', '') or '候选分不开'}）")[:200]
        if any(c["with_id"] == "ground" and not is_geometric_proximity(c)
               for c in o.get("contacts", [])) and g: node["support"] = "ground"
        program["objects"].append(node)
    return program
