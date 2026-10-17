"""按文本给资产打分排序。

查询是视频里那个物体的类名（"toy train"、"cardboard box"）。库里每个资产能拿来比的
只有文件名和 kit 标题，所以这里做的是词级匹配，不是语义检索。够不够用要看命中率，
命中分低于阈值时调用方应该退回基本体，而不是硬塞一个不相干的网格进去。

图像检索（用视频里该物体的最佳裁剪帧比资产缩略图）是下一步，接口留在 retrieve() 的
image_scores 参数上：调用方算好每个资产的图像相似度传进来，这里只负责融合与排序。
"""
from __future__ import annotations

import re
from .library import Asset

# kit 标题里的通用词，带不来任何区分度
STOP = frozenset({"kit", "pack", "assets", "3d", "the", "a", "of", "and"})
# 先按大小写和数字边界插空格，再按非字母数字切。顺序不能反：先转小写或先按
# [^a-z0-9] 切的话，大写字母本身会被当成分隔符吃掉，bathroomCabinet 会切成
# bathroom + abinet，furniture 这一套 camelCase 命名的资产就全查不准。
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_DIGIT = re.compile(r"(?<=[A-Za-z])(?=[0-9])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """把 "train-carriage-box"、"bathroomCabinet"、"tile_0007" 都切成词。"""
    spaced = _DIGIT.sub(" ", _CAMEL.sub(" ", text or ""))
    return [p.lower() for p in _SPLIT.split(spaced) if p and p.lower() not in STOP]


# 同义词。只收那些在库里真的各自存在、而查询时又常用另一个说法的词对。
# 库里有 bookcaseOpen 没有 bookshelf，有 books 没有 book——这两类是资产命中率
# 缺口里最主要的来源，不是检索算法的问题，是词对不上。
SYNONYMS: dict[str, tuple[str, ...]] = {
    "bookshelf": ("bookcase",), "bookcase": ("bookshelf",),
    "sofa": ("couch",), "couch": ("sofa",),
    "rug": ("carpet",), "carpet": ("rug",),
    "cupboard": ("cabinet",), "cabinet": ("cupboard",),
    "mug": ("cup",), "cup": ("mug",),
    "tv": ("television", "monitor"), "television": ("tv", "monitor"),
    "bin": ("trash",), "trash": ("bin",),
    "plant": ("planter", "pot"), "planter": ("plant",),
    "block": ("cube", "brick"), "brick": ("block",), "cube": ("block",),
    "couchtable": ("coffee",),
}
VARIANT_WEIGHT = 0.85      # 同义词和单复数算命中，但比原词低一点，优先挑字面对上的


def _variants(token: str) -> set[str]:
    """一个查询词的所有可接受写法：本身、单复数、同义词。

    单复数要双向处理：查 "book" 要能命中 `books.glb`，查 "books" 也要能命中
    名字写单数的资产。
    """
    out = {token, token + "s", token + "es"}
    if len(token) > 3 and token.endswith("es"):
        out.add(token[:-2])
    if len(token) > 2 and token.endswith("s"):
        out.add(token[:-1])
    out.update(SYNONYMS.get(token, ()))
    return out


def score_asset(query_tokens: list[str], asset: Asset) -> float:
    """[0, 1]。名字里命中算满分，kit 标题里命中算四折；名字越长扣得越多。

    扣长度是因为查 "train" 时 `train.glb` 比 `train-carriage-container-blue.glb` 更可能
    是想要的那个——后者是一节特定的集装箱车厢，前者是这个 kit 对「火车」的代表作。
    """
    if not query_tokens:
        return 0.0
    name_tokens = tokenize(asset.name)
    kit_tokens = tokenize(asset.kit_title) + tokenize(asset.kit)
    name_set, kit_set = set(name_tokens), set(kit_tokens)
    hit, in_name = 0.0, 0
    for token in query_tokens:
        variants = _variants(token)
        if token in name_set:
            hit += 1.0; in_name += 1
        elif variants & name_set:
            hit += VARIANT_WEIGHT; in_name += 1
        elif token in kit_set:
            hit += 0.4
        elif variants & kit_set:
            hit += 0.4 * VARIANT_WEIGHT
    base = hit / len(query_tokens)
    if base <= 0:
        return 0.0
    extra = max(0, len(name_set) - in_name)
    score = base / (1.0 + 0.12 * extra)
    # 查询词正好是某个 kit 的 slug 时给这个 kit 加权：查 "toy train" 应该落到 train 这个
    # kit 的机车上，而不是过山车 kit 里那节叫 coaster-train 的车厢。
    if any(t == asset.kit for t in query_tokens):
        score *= 1.25
    return round(min(score, 1.0), 4)


def retrieve(query: str, catalog: list[Asset], k: int = 5,
             image_scores: dict[str, float] | None = None,
             image_weight: float = 0.5, shortlist: int = 24) -> list[tuple[Asset, float]]:
    """返回前 k 个 (资产, **文字分**)，按融合分排序。

    两个分数各管一件事，别混：

    * **文字分决定收不收。** 调用方拿它和阈值比，低了就退回基本体。
    * **图像分决定收哪个。** 它只在文字选出来的候选里重排。

    不让图像分参与阈值判断是因为量纲不同：DINOv2 比的是真实照片和低多边形渲染，
    余弦相似度天然就低，把它掺进总分再去和为文字分调好的阈值比，会把本来正确的
    结果一起砍掉。重排也顺带挡住了一类错误——图像分再高，没进文字候选的资产也
    不会被选中，所以不会出现「书架」匹配到「门洞」这种语义上不沾边的结果。
    """
    tokens = tokenize(query)
    ranked = [(asset, score) for asset, score in
              ((a, score_asset(tokens, a)) for a in catalog) if score > 0]
    ranked.sort(key=lambda pair: (-pair[1], pair[0].id))
    if image_scores:
        head, tail = ranked[:shortlist], ranked[shortlist:]
        head.sort(key=lambda pair: (-((1 - image_weight) * pair[1]
                                      + image_weight * image_scores.get(pair[0].id, 0.0)),
                                    pair[0].id))
        ranked = head + tail
    return ranked[:k]
