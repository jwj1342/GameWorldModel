"""生成器的不变量。重点是两条：产出必须过校验，样式与行为必须真的分得开。"""
import json
import math

import pytest

from gwm.compiler.validate import validate_candidate
from gwm.dataset.behavior import MOTION_TYPES, TEST_FAMILIES, TRAIN_FAMILIES
from gwm.dataset.generator import (CAMERA_TRACKS, SceneSpec, describe, family_group,
                                   generate, generate_scene, style_half, behavior_half)


def spec(i=0, family="room", profile="test", **kw):
    return SceneSpec(clip=f"c{i}", family=family, behavior_seed=100 + i, style_seed=900 + i,
                     camera_seed=7000 + i, profile=profile, **kw)


def test_same_spec_gives_byte_identical_program():
    assert generate(spec(3)) == generate(spec(3))


@pytest.mark.parametrize("family", TRAIN_FAMILIES + TEST_FAMILIES)
@pytest.mark.parametrize("profile", ["train", "test"])
def test_generated_programs_pass_schema_and_semantic_validation(family, profile):
    for i in range(12):
        program = generate(spec(i, family=family, profile=profile))
        report = validate_candidate(program)
        assert report["ok"], (family, profile, i, report["errors"][:3])


def test_no_drive_and_ignored_params_warnings_never_appear():
    """「写了运动类型但它不会动」和「dynamic 写了脚本参数」都是哑样本，一条都不能有。"""
    seen = set()
    for i in range(60):
        report = validate_candidate(generate(spec(i, family=TEST_FAMILIES[i % 2])))
        seen.update(w["code"] for w in report["warnings"])
    assert "no_drive" not in seen and "ignored_params" not in seen, seen


def test_style_variant_changes_style_only():
    base = spec(11)
    variant = base.with_style(style_seed=424242, clip="c11_style")
    a, b = generate(base), generate(variant)
    assert behavior_half(a) == behavior_half(b)          # 行为一模一样
    assert style_half(a) != style_half(b)                # 样式确实变了


def test_behavior_variant_changes_behavior_only():
    base = spec(12)
    variant = base.with_behavior(behavior_seed=515151, clip="c12_behavior")
    a, b = generate(base), generate(variant)
    assert a["style"] == b["style"]                           # style 子树逐字节相同
    assert a["camera"] == b["camera"]                         # 机位一帧不动
    shared = set(style_half(a)["materials"]) & set(style_half(b)["materials"])
    assert len(shared) >= 2
    for node_id in shared:                                    # 同名节点拿同一种材质
        assert style_half(a)["materials"][node_id] == style_half(b)["materials"][node_id]
    assert behavior_half(a) != behavior_half(b)


@pytest.mark.parametrize("behavior_seed", [112, 515151, 7, 90210, 33331])
def test_behavior_seed_never_leaks_into_style_or_observation(behavior_seed):
    """只换行为种子：样式和观测必须逐字节不动。

    这两条是核心主张第四条在数据这一侧的全部依据。一旦行为种子能顺带改掉光照或机位，
    「只改了行为」这组配对就不再是对照，拿它算出来的解耦度是假的。
    """
    base = spec(14)
    a = generate(base)
    b = generate(base.with_behavior(behavior_seed, f"c14_b{behavior_seed}"))
    assert a["style"] == b["style"]
    assert a["camera"] == b["camera"]
    assert [n["id"] for n in a["static"]] == [n["id"] for n in b["static"]]
    for x, y in zip(a["static"], b["static"]):                # 场地本身也归观测那一路
        assert x["material"] == y["material"]
        if x["class"] == "ground":
            assert x["geom"] == y["geom"] and x["pose"] == y["pose"]


def test_material_count_follows_node_count_so_style_swap_stays_aligned():
    """换样式不能顺带改节点数，否则配对就不是同一个场景了。"""
    base = spec(13)
    a, b = generate(base), generate(base.with_style(77, "c13_style"))
    assert [n["id"] for n in a["objects"]] == [n["id"] for n in b["objects"]]
    assert [n["id"] for n in a["static"]] == [n["id"] for n in b["static"]]


def test_test_profile_covers_every_motion_type_with_room_to_spare():
    """表 1 的混淆结构要求八类都有样本，而且不能有哪一类只零星出现几个。

    RP 5.8 要的是「每类在测试集里至少有数十个实例」。八类等权的话每类期望占 12.5%，
    这里卡 6%，既拦得住「某一类几乎抽不到」的分布 bug，又不会被正常涨落误伤。
    """
    counts = {k: 0 for k in MOTION_TYPES}
    for i in range(200):
        for o in generate(spec(i, family=TEST_FAMILIES[i % 2], profile="test"))["objects"]:
            counts[(o.get("motion") or {}).get("type")] += 1
    total = sum(counts.values())
    assert total > 400, total
    thin = {k: v for k, v in counts.items() if v / total < 0.06}
    assert not thin, (thin, total)


def test_train_profile_is_skewed_but_still_covers_every_type():
    """训练集按常见度配，所以允许偏，但不能把某一类配成零——那样模型永远学不到它。"""
    counts = {k: 0 for k in MOTION_TYPES}
    for i in range(200):
        for o in generate(spec(i, family=TRAIN_FAMILIES[i % 3], profile="train"))["objects"]:
            counts[(o.get("motion") or {}).get("type")] += 1
    assert all(v > 0 for v in counts.values()), counts
    assert counts["static"] > counts["dynamic"]          # 偏的方向要和声明的一致


def test_train_and_test_families_are_disjoint():
    assert not set(TRAIN_FAMILIES) & set(TEST_FAMILIES)
    assert all(family_group(f) == "train" for f in TRAIN_FAMILIES)
    assert all(family_group(f) == "test" for f in TEST_FAMILIES)


@pytest.mark.parametrize("track", CAMERA_TRACKS)
def test_every_camera_track_has_a_real_baseline(track):
    """没有一档是机位不动的。固定俯视被换成了平移俯视，就是为了这条。

    合成数据要跑一遍真实感知，而没有基线的位姿估不出来；留着那一档等于
    往训练集里塞一批建立在估不出的位姿上的样本。
    """
    for i in range(10):
        program = generate(spec(i, family="yard", camera_track=track))
        kf = program["camera"]["keyframes"]
        first, last = kf[0]["pos"], kf[-1]["pos"]
        assert math.dist(first, last) > 1.0, (track, i, first, last)
        assert [k["t"] for k in kf] == sorted(k["t"] for k in kf)


def test_scripted_motion_stays_over_the_ground():
    """脚本运动不走物理，写出界的关键帧没人拦——物体会飘出地面或穿墙。

    这种画面物理上不成立，而我们恰恰主张产物带着真实的可运动性和物理特性，
    训练数据里不能有反例。
    """
    families = TRAIN_FAMILIES + TEST_FAMILIES
    checked = 0
    for i in range(120):
        program = generate(spec(i, family=families[i % len(families)]))
        ground = next(n for n in program["static"] if n["class"] == "ground")
        half_w, half_d = ground["geom"]["extent"][0] / 2, ground["geom"]["extent"][2] / 2
        for obj in program["objects"]:
            motion = obj.get("motion") or {}
            for kf in motion.get("keyframes") or []:
                checked += 1
                assert abs(kf["pos"][0]) <= half_w + 1e-6, (obj["id"], kf["pos"], half_w)
                assert abs(kf["pos"][2]) <= half_d + 1e-6, (obj["id"], kf["pos"], half_d)
    assert checked > 50, checked          # 真的检到了轨迹，而不是一条都没抽到


def test_program_carries_no_generator_provenance():
    """真值程序是模型要预测的答案，里面不能有家族名、相机档、profile 这些小抄。

    留着的话模型会学着把它背出来，而真实视频上根本没有这一行，到时候就是训练推理不一致。
    """
    allowed = {"clip", "fps", "duration", "units", "up", "notes"}
    for i in range(20):
        families = TRAIN_FAMILIES + TEST_FAMILIES
        program, facts = generate_scene(spec(i, family=families[i % len(families)]))
        assert set(program["meta"]) <= allowed, set(program["meta"]) - allowed
        # 相机档是纯粹的生成器概念，场景里没有任何东西该叫这个名字
        assert facts["camera_track"] not in json.dumps(program, ensure_ascii=False)


def test_manifest_row_reports_the_self_answering_rate():
    rows = []
    for i in range(40):
        s = spec(i, family="yard")
        program, facts = generate_scene(s)
        rows.append(describe(program, s, facts))
    objects = sum(r["object_count"] for r in rows)
    hits = sum(r["binding_vocab_hits"] for r in rows)
    assert objects > 0 and 0 < hits < objects          # 既不是零，也不是全部——这个比例要能报出来
    assert sum(sum(r["material_forms"].values()) for r in rows) > 0


def test_structural_materials_stay_muted_and_never_glow():
    """地面和墙不该是洋红色，也不该自己发光。合成到真实那道坎本来就够宽了。"""
    for i in range(40):
        program = generate(spec(i, family=(TRAIN_FAMILIES + TEST_FAMILIES)[i % 5]))
        for node in program["static"]:
            material = node["material"]
            if isinstance(material, dict):
                assert "emissive" not in material, (node["id"], material)
                channels = [int(material["base_color"][k:k + 2], 16) for k in (1, 3, 5)]
            elif material.startswith("#"):
                channels = [int(material[k:k + 2], 16) for k in (1, 3, 5)]
            else:
                continue                      # 调色板词，由 STRUCTURAL_PALETTE 保证
            assert max(channels) - min(channels) <= 60, (node["id"], material)


def test_all_three_material_forms_appear():
    forms = {"palette": 0, "hex": 0, "surface": 0}
    for i in range(40):
        program, facts = generate_scene(spec(i, family="room"))
        row = describe(program, spec(i, family="room"), facts)
        for key, value in row["material_forms"].items():
            forms[key] += value
    assert all(count > 0 for count in forms.values()), forms


def _diag(node):
    geom = node["geom"]
    extent = geom.get("extent")
    if not extent:
        r = geom.get("radius", 0.5)
        extent = [2 * r, geom.get("height", 2 * r), 2 * r]
    return math.sqrt(sum(v * v for v in extent))


def test_in_place_rotation_only_lands_on_shapes_where_it_is_visible():
    """球、圆柱、圆锥绕竖轴自转，画面一帧都不变——这种样本任何方法都只能瞎猜。

    RP 5.8 要求测试集按可测性配。观测不到的组合留在里面，只会把表 1 的
    运动类型准确率整列变成噪声。
    """
    families = TRAIN_FAMILIES + TEST_FAMILIES
    checked = 0
    for i in range(150):
        for node in generate(spec(i, family=families[i % len(families)]))["objects"]:
            if (node.get("motion") or {}).get("type") not in ("spin", "periodic_rotate"):
                continue
            checked += 1
            extent = node["geom"].get("extent")
            assert node["geom"]["shape"] == "box" and extent, (node["id"], node["geom"])
            assert max(extent[0], extent[2]) / min(extent[0], extent[2]) >= 1.6, (node["id"], extent)
    assert checked > 30, checked


def test_translation_moves_further_than_the_object_is_big():
    """位移比物体自己还小的时候，质心那点变化和静止没区别。

    用物体自身尺寸作尺度是刻意的：不去对着评测器的阈值调，那是拿数据迁就指标。
    """
    families = TRAIN_FAMILIES + TEST_FAMILIES
    checked = 0
    for i in range(150):
        program = generate(spec(i, family=families[i % len(families)]))
        duration = program["meta"]["duration"]
        for node in program["objects"]:
            motion = node.get("motion") or {}
            kind, bar = motion.get("type"), 0.5 * _diag(node)
            if kind == "trajectory":
                pts = [k["pos"] for k in motion["keyframes"]]
                span = max(math.dist(a, b) for a in pts for b in pts)
            elif kind == "prismatic":
                span = (motion["schedule"][0]["to"] if motion.get("schedule")
                        else motion.get("rate", 0) * duration)
            elif kind == "periodic_translate":
                span = 2 * motion["amp"]                     # 峰到峰
            else:
                continue
            checked += 1
            assert abs(span) >= bar, (node["id"], kind, span, bar)
    assert checked > 60, checked


def test_ring_track_never_stacks_two_rails_on_one_spot():
    """轨道条数是抽的，角度必须按抽到的条数均分。

    之前角度写死除以 8，抽到 9 条时最后一条整个压在第一条上。重复位姿是坏几何，
    真值里出现两个完全重合的物体，评测那边的配对也会跟着选错。
    """
    seen = set()
    for i in range(80):
        program = generate(spec(600 + i, family="track"))
        rails = [n for n in program["static"] if n["class"] == "rail"]
        seen.add(len(rails))
        spots = {tuple(n["pose"]["pos"]) for n in rails}
        assert len(spots) == len(rails), f"{len(rails)} 条轨道只占了 {len(spots)} 个位置"
    assert {9, 10} <= seen, f"没抽到 9 或 10 条的样本，这条测试没覆盖到边界：{sorted(seen)}"
