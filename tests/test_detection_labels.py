"""检测标签收敛。Grounding DINO 的提示词是把所有短语拼成一串的，一个框落在多个短语的
词上时返回的是拼接，那种类名在资产库里查不到任何东西，写进程序也没法读。"""
import pytest

from gwm.perception.grounded_sam2_backend import snap_label

PHRASES = ["toy train", "wooden block", "cardboard box", "storage box", "table", "cup"]


@pytest.mark.parametrize("label,expect", [
    ("cardboard box", "cardboard box"),                                   # 本来就干净
    ("wooden block cardboard box storage box", "cardboard box"),          # 取最长的那个
    ("toy train wooden block cardboard box", "cardboard box"),
    ("Cardboard Box", "cardboard box"),                                   # 大小写无关
    ("  table  ", "table"),
])
def test_拼接标签收敛到一个真实短语(label, expect):
    assert snap_label(label, PHRASES) == expect


def test_对不上的标签原样返回而不是瞎猜():
    assert snap_label("fire hydrant", PHRASES) == "fire hydrant"


def test_空标签不炸():
    assert snap_label("", PHRASES) == ""
    assert snap_label("table", []) == "table"


# ---- 词片走样：检测器返回的是 token 拼起来的串，不是我们问的那句话 ----

def test_子词标记的尾巴能认回原短语():
    # "spool" 被切成 "sp" + "##ool"，框只落在后半片上
    assert snap_label("##ool", ["spool", "crate", "barrel"]) == "spool"


def test_下划线被拆开也能认回来():
    # 下划线当成独立 token，还原时两边各加一个空格
    assert snap_label("cone _ marker", ["cone_marker", "drum"]) == "cone_marker"
    # 后面还粘了别的词片的情况
    assert snap_label("cone _ markertem", ["cone_marker", "drum"]) == "cone_marker"


def test_片段能对上好几个短语的时候宁可不认():
    # "box" 同时是两个短语的一段，认哪个都是猜
    raw = "box"
    assert snap_label(raw, ["storage box", "tool box"]) == raw


def test_太短的片段不认():
    raw = "ol"
    assert snap_label(raw, ["spool"]) == raw


def test_整串拼接仍然取最长的那个():
    # 原来就有的行为不能被新规则改掉
    assert snap_label("wooden block cardboard box storage box drawer",
                      ["block", "storage box", "drawer"]) == "storage box"
