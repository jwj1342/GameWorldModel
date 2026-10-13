"""生成器的不变量。重点是两条：产出必须过校验，样式与行为必须真的分得开。"""
import json
import math

import pytest

from gwm.compiler.validate import validate_candidate
from gwm.dataset.behavior import MOTION_TYPES, TEST_FAMILIES, TRAIN_FAMILIES
from gwm.dataset.generator import (CAMERA_TRACKS, SceneSpec, describe, family_group,
                                   generate, generate_scene)
from gwm.dataset.style import style_fields


def spec(i=0, family="room", profile="test", **kw):
    return SceneSpec(clip=f"c{i}", family=family, behavior_seed=100 + i, style_seed=900 + i,
                     camera_seed=7000 + i, profile=profile, **kw)


def behavior_fields(program):
    """样式之外的全部：结构、几何、运动、相机。配对测试用它比对。"""
    return {
        "static": [{k: v for k, v in n.items() if k != "material"} for n in program["static"]],
        "objects": [{k: v for k, v in n.items() if k != "material"} for n in program["objects"]],
        "camera": program["camera"],
    }


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
    assert behavior_fields(a) == behavior_fields(b)          # 行为一模一样
    assert style_fields(a) != style_fields(b)                # 样式确实变了


def test_behavior_variant_changes_behavior_only():
    base = spec(12)
    variant = base.with_behavior(behavior_seed=515151, clip="c12_behavior")
    a, b = generate(base), generate(variant)
    assert a["style"] == b["style"]                           # style 子树逐字节相同
    assert behavior_fields(a) != behavior_fields(b)


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


def test_all_three_material_forms_appear():
    forms = {"palette": 0, "hex": 0, "surface": 0}
    for i in range(40):
        program, facts = generate_scene(spec(i, family="room"))
        row = describe(program, spec(i, family="room"), facts)
        for key, value in row["material_forms"].items():
            forms[key] += value
    assert all(count > 0 for count in forms.values()), forms
