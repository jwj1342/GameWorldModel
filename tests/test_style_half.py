"""样式那一半的字段：结构化材质与光照。

会上把「样式与行为分成两半写」定成第四条核心主张，而样式这一半在 DSL 里只有一个
平色枚举，光照干脆写死在内核里。写死的后果不是难看——是合成数据生成器随机化了
光照之后，真值程序里没有地方记它，于是每条样本的样式那一半按构造就是有损的。

这里钉三件事：三种材质写法都认、编出来的词一律拒绝、光照进了程序而且缺省不变。
"""
import json
import re

import pytest

from gwm.compiler.validate import validate
from gwm.config import REPO


def scene(**over):
    p = {"meta": {"clip": "x", "duration": 4, "fps": 30, "units": "m", "up": "y"},
         "camera": {"keyframes": [{"t": 0, "pos": [8, 6, 8], "look_at": [0, 0, 0]}]},
         "static": [{"id": "ground", "class": "floor",
                     "geom": {"kind": "primitive", "shape": "box", "extent": [24, 0.5, 16]},
                     "pose": {"pos": [0, -0.25, 0]}}],
         "objects": [{"id": "crate", "class": "box",
                      "geom": {"kind": "primitive", "shape": "box", "extent": [1, 1, 1]},
                      "pose": {"pos": [0, 0.5, 0]}}],
         "binding": {"template": "platformer_3p", "slots": {"walkable": "auto"}}}
    p.update(over)
    return p


def with_material(m):
    p = scene()
    p["objects"][0]["material"] = m
    return p


@pytest.mark.parametrize("material", [
    "wood",                                                        # 调色板
    "#c04a3b",                                                     # 十六进制
    {"base_color": "#c04a3b"},                                     # 结构化，只给底色
    {"base_color": "#c04a3b", "roughness": 0.2, "metalness": 0.8},
    {"base_color": "#ff5a1f", "emissive": "#ff2200", "emissive_intensity": 0.8},
    {"base_color": "#bfe4ea", "opacity": 0.4},
])
def test_the_three_ways_of_writing_a_material_are_all_accepted(material):
    assert validate(with_material(material))["ok"], material


@pytest.mark.parametrize("material", [
    "brushed_aluminium",          # 编出来的词：内核会悄悄渲染成灰色，所以不能放过
    "Wood",                       # 大小写也不行，内核查表是精确匹配
    "#c04a3",                     # 少一位
    "rgb(192,74,59)",
    {"roughness": 0.5},           # 没有底色
    {"base_color": "#c04a3b", "shininess": 3},   # 编出来的字段
    {"base_color": "#c04a3b", "metalness": 1.5},
])
def test_invented_material_words_and_fields_are_rejected(material):
    """这条是防「悄悄变灰」：内核 MATERIALS 查不到就落回 default，
    渲染出来是一块灰，没有任何报错，而样式那一半的分数就这么丢了。"""
    assert not validate(with_material(material))["ok"], material


def test_lighting_lives_in_the_program():
    p = scene(style={"background": "#8fb3d9",
                     "lighting": {"ambient": {"sky_color": "#ffeedd", "ground_color": "#223344", "intensity": 0.6},
                                  "key": {"color": "#fff2cc", "intensity": 2.4,
                                          "direction": [-4, 9, 3], "shadow": False}}})
    assert validate(p)["ok"]


@pytest.mark.parametrize("style", [
    {"background": "skyblue"},                                  # 颜色名不收，内核那边才认
    {"background": "#8fb3d9", "fog": "#cccccc"},                # 编出来的顶层样式字段
    {"lighting": {"key": {"intensity": -1}}},
    {"lighting": {"sun": {"intensity": 1}}},                    # 编出来的灯名
])
def test_invented_style_fields_are_rejected(style):
    assert not validate(scene(style=style))["ok"], style


def test_a_zero_key_light_direction_is_an_error():
    """方向是零向量的时候 three.js 把灯摆在原点，整个场景只剩环境光，
    而这在 schema 里是合法的 vec3，只能语义层拦。"""
    r = validate(scene(style={"lighting": {"key": {"direction": [0, 0, 0]}}}))
    assert not r["ok"]
    assert any(e["code"] == "zero_direction" for e in r["errors"])


def test_emissive_intensity_without_emissive_is_only_a_warning():
    """它不会让程序跑不起来，只是那个数没人读，所以是提示不是错误。"""
    r = validate(with_material({"base_color": "#c04a3b", "emissive_intensity": 2}))
    assert r["ok"]
    assert any(w["code"] == "emissive_intensity_without_emissive" for w in r["warnings"])


def test_the_palette_in_the_schema_and_in_the_kernel_do_not_drift():
    """调色板有两份：schema 里的闭集和 kernel/scene.js 里的查表。
    两边漂开的话，schema 放行的词内核渲染成灰色，正是这个 PR 要堵的那个洞。"""
    schema = json.loads((REPO / "gwm/compiler/schema/program.schema.json").read_text())
    declared = set(schema["$defs"]["materialName"]["enum"])
    source = (REPO / "kernel/scene.js").read_text()
    start = source.index("export const MATERIALS = {")
    body = source[start:source.index("\n};", start)]
    implemented = set(re.findall(r"^\s{2}(\w+):\s*\{", body, flags=re.M))
    assert declared == implemented, f"schema 多了 {declared - implemented}，内核多了 {implemented - declared}"
