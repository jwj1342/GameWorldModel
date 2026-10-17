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
