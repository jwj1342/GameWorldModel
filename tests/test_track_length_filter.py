"""只露几帧的轨迹直接丢掉——这一轮唯一真正推动了精细度的改动。

扫了 0.0 / 0.10 / 0.20 / 0.30 四档，十二条片段。把各片段的计数加起来算（不是片段级
中位）：

    物体精确率  0.214 -> 0.247 -> 0.300    四档单调
    场景精确率  0.285 -> 0.307 -> 0.389    四档单调
    物体召回    0.407 -> 0.373 -> 0.356
    预测物体数  112 -> 89 -> 70
    可交互性    0.929 -> 0.921 -> 0.943    0.30 是四档里最高的

逐片段配对（同一条片段两臂比，消掉片段难度的差异）：场景精确率 8 赢 / 2 输 / 2 平，
物体召回 0 赢 / 3 输 / 9 平——召回只在三条片段上掉。

代价是实在的：召回 0.407 -> 0.356。用户定的优先级是「先把物体精确率拉起来」。
"""
import pytest

from gwm.config import load_config
from pathlib import Path


def cfg():
    return load_config([str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")])


def test_默认开着且是零点三():
    assert cfg()["perception"]["min_track_frame_frac"] == pytest.approx(0.30)


def test_注释里留下了召回的代价():
    """这是个有代价的取舍，不是白捡的。配置注释必须把掉了多少召回写出来，
    否则以后有人看到精确率涨了就以为是纯赚。"""
    text = (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text(encoding="utf-8")
    # 取设定行**之前**那一段注释。第一版用 index(...) - 2000 去切，第一次出现的
    # "min_track_frame_frac" 正是在注释块里面，减 2000 直接切到了文件开头之外，
    # block 成了空串——断言于是在一个空字符串上失败，报的错和真实原因对不上。
    end = text.index("min_track_frame_frac: 0.30")
    block = text[max(0, end - 2000):end]
    assert "召回" in block and "0.356" in block, "要写明召回掉到多少"
    assert "更大的值没扫过" in block, "要写明 0.30 是扫过的最好一档，不是最优解"
