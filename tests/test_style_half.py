"""样式那一半的字段：结构化材质与光照。

会上把「样式与行为分成两半写」定成第四条核心主张，而样式这一半在 DSL 里只有一个
平色枚举，光照干脆写死在内核里。写死的后果不是难看——是合成数据生成器随机化了
光照之后，真值程序里没有地方记它，于是每条样本的样式那一半按构造就是有损的。

这里钉三件事：三种材质写法都认、编出来的词一律拒绝、光照进了程序而且缺省不变。
"""
import json
import re
import copy

import pytest

from gwm.compiler.validate import validate
from gwm.compiler.compile import compile_program
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


def assert_nonfinite_path(program, path):
    report = validate(program)
    assert not report["ok"]
    assert any(e["code"] == "non_finite" and e["path"] == path
               and e["stage"] == "semantic" and e["severity"] == "error"
               for e in report["errors"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("light", ["ambient", "key"])
def test_nonfinite_light_intensity_has_exact_path(value, light):
    p = scene(style={"lighting": {light: {"intensity": value}}})
    assert_nonfinite_path(p, f"/style/lighting/{light}/intensity")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_nonfinite_light_direction_has_component_path(value, index):
    direction = [1, 2, 3]
    direction[index] = value
    assert_nonfinite_path(scene(style={"lighting": {"key": {"direction": direction}}}),
                          f"/style/lighting/key/direction/{index}")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["roughness", "metalness", "emissive_intensity", "opacity"])
@pytest.mark.parametrize("collection", ["objects", "static"])
@pytest.mark.parametrize("location", ["material", "geom/material"])
def test_nonfinite_material_at_all_supported_locations(value, field, collection, location):
    p = scene()
    node = p[collection][0]
    target = node["geom"] if location.startswith("geom/") else node
    target["material"] = {"base_color": "#aabbcc", "emissive": "#ffffff", field: value}
    assert_nonfinite_path(p, f"/{collection}/0/{location}/{field}")


def test_json_numeric_overflow_is_rejected_before_any_compile_work(tmp_path, monkeypatch):
    import importlib
    compiler = importlib.import_module("gwm.compiler.compile")
    p = scene(style=json.loads('{"lighting":{"key":{"intensity":1e400}}}'))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid style must not reach asset resolution or bundling")

    monkeypatch.setattr(compiler, "resolve_assets", forbidden)
    monkeypatch.setattr(compiler, "bundle", forbidden)
    out = tmp_path / "game"
    result = compile_program(p, out)
    assert not result["ok"] and not out.exists()
    assert_nonfinite_path(p, "/style/lighting/key/intensity")


def test_valid_style_validation_does_not_mutate_program():
    p = scene(style={"lighting": {"ambient": {"intensity": 0},
                                 "key": {"intensity": 2, "direction": [1, 2, 3]}}})
    p["objects"][0]["geom"]["material"] = {
        "base_color": "#aabbcc", "roughness": 0, "metalness": 1,
        "opacity": 1, "emissive_intensity": 0}
    before = copy.deepcopy(p)
    assert validate(p)["ok"]
    assert p == before


@pytest.mark.parametrize("program", [None, [], {"style": []},
    {"style": {"lighting": {"key": []}}}, {"objects": [None]},
    {"objects": [{"material": {"roughness": float("inf")}}]}])
def test_style_finite_check_is_safe_on_schema_invalid_inputs(program):
    assert not validate(program)["ok"]
