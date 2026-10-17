"""资产这一层的不变量。重点两条：挑错了要退回基本体，以及挑什么都不该动到碰撞体。

这里不碰真实素材库（它在 /project 上，几个 GB），所有用例都在临时目录里造一个小库。
"""
import json

import pytest

from gwm.assets.library import Asset, build_catalog, load_catalog, write_catalog
from gwm.assets.retrieve import retrieve, score_asset, tokenize
from gwm.compiler.assets import resolve_assets
from gwm.compiler.bundle import copy_assets


@pytest.fixture
def library(tmp_path):
    """一个够用的小库：train kit 里既有机车也有连接件，furniture kit 里有桌子。"""
    root = tmp_path / "lib"
    (root / "3d").mkdir(parents=True)
    (root / "3d" / "kits.tsv").write_text(
        "slug\ttitle\tmodels\ntrain\tTrain Kit\t2\nfurniture\tFurniture Kit\t1\n", encoding="utf-8")
    for kit, names in {"train": ["train-locomotive-a", "train-connector"], "furniture": ["table"]}.items():
        (root / "3d" / kit).mkdir()
        for name in names:
            (root / "3d" / kit / f"{name}.glb").write_bytes(b"glTF\x02\x00\x00\x00fake")
    return root


def program_with(query, **geom_extra):
    return {"meta": {"clip": "c"}, "static": [],
            "objects": [{"id": "obj_1", "class": "toy train",
                         "geom": {"kind": "asset", "query": query, "extent": [2, 1, 1], **geom_extra},
                         "pose": {"pos": [0, 0.5, 0]}}]}


def test_tokenize_splits_the_three_naming_styles():
    # 结尾那个 a 是同一个模型的变体编号（还有 -b、-c），不带查询信息，当停用词切掉；
    # 变体之间选哪个交给图像重排，文字这边不该因为字母不同就分高下
    assert tokenize("train-locomotive-a") == ["train", "locomotive"]
    assert tokenize("bathroomCabinet") == ["bathroom", "cabinet"]
    assert tokenize("Train Kit") == ["train"]            # kit 是通用词，切掉
    assert tokenize("tile_0007") == ["tile", "0007"]


def test_name_match_beats_kit_match():
    in_name = Asset(id="train/table", path="p", kit="train", kit_title="Train Kit", name="table")
    in_kit = Asset(id="furniture/stool", path="p", kit="furniture", kit_title="Furniture Kit", name="stool")
    assert score_asset(tokenize("table"), in_name) > score_asset(tokenize("furniture"), in_kit)


def test_kit_affinity_pulls_the_right_kit_up(library):
    """查 "toy train" 应该落到 train 这个 kit，而不是别的 kit 里名字带 train 的零件。"""
    catalog = build_catalog(library) + [
        Asset(id="coaster/coaster-train", path="x", kit="coaster", kit_title="Coaster Kit", name="coaster-train")]
    top = retrieve("toy train", catalog, k=1)[0][0]
    assert top.kit == "train", f"挑到了 {top.id}"


def test_image_scores_reorder_but_do_not_change_the_candidate_set(library):
    """只看文字时连接件和机车分不出高下；图像分要能把机车顶上来，但不能引入新候选。"""
    catalog = build_catalog(library)
    text_only = retrieve("toy train", catalog, k=2)
    reranked = retrieve("toy train", catalog, k=2,
                        image_scores={"train/train-locomotive-a": 0.95, "train/train-connector": 0.10},
                        image_weight=0.6)
    assert reranked[0][0].id == "train/train-locomotive-a"
    assert {a.id for a, _ in text_only} == {a.id for a, _ in reranked}, "重排不该改候选集合"


def test_image_scores_never_change_the_accept_decision(library):
    """返回的分数始终是文字分。图像分再低也不该把一个文字上明确命中的资产挡在阈值外——
    DINOv2 比的是照片和低多边形渲染，分数天然偏低，掺进去会把正确结果一起砍掉。"""
    catalog = build_catalog(library)
    plain = dict((a.id, s) for a, s in retrieve("table", catalog, k=5))
    with_images = dict((a.id, s) for a, s in retrieve(
        "table", catalog, k=5, image_scores={"furniture/table": 0.02}, image_weight=0.9))
    assert plain["furniture/table"] == with_images["furniture/table"]


def test_image_score_cannot_pull_in_an_asset_the_text_never_matched(library):
    """图像分再高，没进文字候选的资产也不该被选中——否则「书架」会匹配到「门洞」。"""
    catalog = build_catalog(library)
    hits = retrieve("table", catalog, k=5, image_scores={"train/train-connector": 1.0}, image_weight=0.9)
    assert "train/train-connector" not in {a.id for a, _ in hits}


def test_retrieval_is_deterministic(library):
    catalog = build_catalog(library)
    assert [a.id for a, _ in retrieve("table", catalog, k=3)] == [a.id for a, _ in retrieve("table", catalog, k=3)]


def test_catalog_round_trips(library):
    built = build_catalog(library)
    write_catalog(library, built)
    assert [a.id for a in load_catalog(library)] == [a.id for a in built]


def test_ref_in_the_program_wins_over_text_search(library):
    cfg = {"assets": {"library": str(library), "min_score": 0.45}}
    manifest = resolve_assets(program_with("toy train", ref="train/train-connector"), cfg)
    request = manifest["requests"][0]
    assert request["resolved"] == "library_ref" and request["asset_id"] == "train/train-connector"


def test_unknown_ref_falls_back_to_search_and_says_so(library):
    cfg = {"assets": {"library": str(library), "min_score": 0.45}}
    request = resolve_assets(program_with("toy train", ref="train/does-not-exist"), cfg)["requests"][0]
    assert request["ref_miss"] == "train/does-not-exist"
    assert request["resolved"] in ("library", "primitive_fallback")


def test_low_score_keeps_the_primitive(library):
    """宁可放个盒子，也不要塞一个不相干的网格——否则「像不像」这个指标没法解释。"""
    cfg = {"assets": {"library": str(library), "min_score": 0.99}}
    assert resolve_assets(program_with("toy train"), cfg)["requests"][0]["resolved"] == "primitive_fallback"


def test_no_library_means_everything_is_a_primitive():
    manifest = resolve_assets(program_with("toy train"), {"assets": {"enabled": False}})
    assert manifest["strategy"] == ["primitive"]
    assert manifest["requests"][0]["resolved"] == "primitive_fallback"


def test_bundling_copies_the_glb_and_records_the_licence(library, tmp_path):
    cfg = {"assets": {"library": str(library), "min_score": 0.3}}
    manifest = resolve_assets(program_with("toy train"), cfg)
    out = tmp_path / "game"; out.mkdir()
    mapping = copy_assets(out, manifest)
    entry = mapping["objects"]["obj_1"]
    assert (out / entry["file"]).is_file() and entry["license"] == "CC0"


def test_missing_source_degrades_instead_of_raising(library, tmp_path):
    cfg = {"assets": {"library": str(library), "min_score": 0.3}}
    manifest = resolve_assets(program_with("toy train"), cfg)
    manifest["requests"][0]["source"] = str(tmp_path / "gone.glb")
    mapping = copy_assets(tmp_path / "game", manifest)
    assert mapping["missing_sources"] == ["obj_1"]
    assert manifest["requests"][0]["resolved"] == "primitive_fallback"


def test_asset_choice_never_touches_the_behaviour_half(library):
    """换资产只能改 geom.ref。运动、位姿、事件、碰撞相关的字段一个都不能动——
    碰撞体在内核里是从 extent 算的，所以 extent 也必须原样保留。"""
    cfg = {"assets": {"library": str(library), "min_score": 0.3}}
    program = program_with("toy train")
    program["objects"][0]["motion"] = {"type": "trajectory", "points": [[0, 0.5, 0], [2, 0.5, 0]], "duration": 4}
    before = json.loads(json.dumps(program))
    resolve_assets(program, cfg)
    assert program == before, "resolve_assets 不该就地改程序"


def test_crop_replaces_the_background_so_only_the_object_is_compared():
    """抠图必须把背景换掉：拿一整块带房间背景的裁剪去比资产缩略图，比的是背景。"""
    import numpy as np
    from gwm.assets.annotate import THUMB_BG, crop_object
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    frame[15:25, 15:25] = (200, 10, 10)
    mask = np.zeros((40, 40), dtype=bool); mask[15:25, 15:25] = True
    # 留一点边距，裁剪里才会同时出现物体和它周围，好检查周围确实被换掉了
    crop = crop_object(frame, mask, [15, 15, 25, 25], pad=0.3)
    assert crop is not None
    colours = {tuple(c) for c in crop.reshape(-1, 3)}
    assert (200, 10, 10) in colours, "物体本身应该保留原色"
    assert (10, 20, 30) not in colours, "原背景色不该留在裁剪里"
    assert THUMB_BG in colours, "掩码外应该换成缩略图的底色"


def test_empty_mask_gives_no_crop():
    import numpy as np
    from gwm.assets.annotate import crop_object
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert crop_object(frame, np.zeros((20, 20), dtype=bool), [0, 0, 20, 20]) is None


def test_structured_bbox_is_understood():
    """证据里的框是带单位的结构体，不是裸四元组；读错会让整条图像检索静默失效。"""
    from gwm.assets.annotate import bbox_of, best_observation
    structured = {"frame_index": 0, "bbox": {"format": "xyxy", "values": [1, 2, 11, 22],
                                             "space": "pixel", "image_size": [518, 294]}}
    values, size = bbox_of(structured)
    assert values == [1.0, 2.0, 11.0, 22.0] and size == [518, 294]
    assert bbox_of({"bbox": [1, 2, 3, 4]}) == ([1.0, 2.0, 3.0, 4.0], None)
    assert bbox_of({"bbox": {"format": "cxcywh", "values": [1, 2, 3, 4]}}) is None
    small = {"frame_index": 1, "bbox": {"format": "xyxy", "values": [0, 0, 2, 2], "image_size": [518, 294]}}
    assert best_observation({"observations": [small, structured]}) is structured


@pytest.mark.parametrize("ref", [None, "train/train-locomotive-a"])
def test_bundling_copies_both_search_hits_and_explicit_refs(library, tmp_path, ref):
    """走 ref 的那条也必须被拷进 game 目录。只认文字检索那条的话，看过画面挑出来的
    那一批会整批被跳过，而清单上仍然显示解析成功——画面还是盒子，日志却说一切正常。"""
    cfg = {"assets": {"library": str(library), "min_score": 0.3}}
    program = program_with("toy train", **({"ref": ref} if ref else {}))
    manifest = resolve_assets(program, cfg)
    out = tmp_path / ("game-" + (ref or "search").replace("/", "-")); out.mkdir()
    mapping = copy_assets(out, manifest)
    assert "obj_1" in mapping["objects"], f"resolved={manifest['requests'][0]['resolved']} 没被拷贝"
    assert (out / mapping["objects"]["obj_1"]["file"]).is_file()


@pytest.mark.parametrize("query,expect", [
    ("book", "furniture/books"),          # 库里只有复数形式
    ("books", "furniture/books"),
    ("bookshelf", "furniture/bookcase"),  # 库里叫 bookcase
    ("couch", "furniture/sofa"),
])
def test_单复数与同义词能对上库里的实际命名(query, expect):
    """资产命中率的缺口大半不是算法问题，是词对不上：库里有 books.glb 没有 book.glb，
    有 bookcaseOpen 没有 bookshelf。"""
    from gwm.assets.library import Asset
    from gwm.assets.retrieve import retrieve
    catalog = [Asset(id="furniture/books", path="p", kit="furniture", kit_title="Furniture Kit", name="books"),
               Asset(id="furniture/bookcase", path="p", kit="furniture", kit_title="Furniture Kit", name="bookcaseOpen"),
               Asset(id="furniture/sofa", path="p", kit="furniture", kit_title="Furniture Kit", name="sofa"),
               Asset(id="train/track", path="p", kit="train", kit_title="Train Kit", name="track")]
    hits = retrieve(query, catalog, k=1)
    assert hits and hits[0][0].id == expect, f"{query} -> {[a.id for a, _ in hits]}"


def test_字面对上的排在同义词前面():
    from gwm.assets.library import Asset
    from gwm.assets.retrieve import retrieve
    catalog = [Asset(id="a/cup", path="p", kit="a", kit_title="A", name="cup"),
               Asset(id="a/mug", path="p", kit="a", kit_title="A", name="mug")]
    assert retrieve("cup", catalog, k=2)[0][0].id == "a/cup"
