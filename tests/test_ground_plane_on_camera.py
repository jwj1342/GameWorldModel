"""压在相机上的地平面候选要被丢掉，否则整条片段产出零个物体。

`align_world` 原来对相机高度只有 `cam_height > 0` 一条约束。实测 ab015fix/test_00000
上，候选 0 的 cam_height 只有 0.0768（相对单位），而选择打分只看内点数，它有 11350 个
内点，正确的候选 1 只有 5296——于是必然选错。

后果不是尺度估偏：地面错了 1.39 个单位，再乘上为了把相机凑到 1.2 米解出的 15.6 倍，
物体落到地下二十多米，被下陷闸丢光，**整条片段产出零个物体**。评测里它表现为
「召回 0」，和「检测什么都没找到」长得一样（#84）。

下面的候选数值全部是从 `evidence.json` 里实测抄来的，不是编的。
"""
import pytest

from gwm.perception.evidence import reject_planes_on_camera

RATIO = 0.25


def cand(idx, inliers, cam_height, frac_obj_above=1.0):
    return {"idx": idx, "inliers": inliers, "cam_height": cam_height,
            "frac_obj_above": frac_obj_above}


def pick(ok):
    """复刻 align_world 的选择打分。"""
    return max(ok, key=lambda c: c["inliers"] * (0.5 + 0.5 * c["frac_obj_above"]))


# （片段, 候选, 原来会选中的 idx, 修好之后该选中的 idx）
实测的三条退化片段 = [
    ("ab015fix/test_00000",
     [cand(0, 11350, 0.0768), cand(1, 5296, 1.4668), cand(2, 3243, 1.4672), cand(3, 2695, 3.4626)],
     0, 1),
    ("hold015fix/test_00009",
     [cand(0, 18273, 0.0350), cand(1, 4207, 0.7478), cand(2, 2882, 0.7500)],
     0, 1),
]


@pytest.mark.parametrize("clip, cands, 旧的, 新的", 实测的三条退化片段)
def test_贴着相机的候选被丢掉之后选中正确的那个(clip, cands, 旧的, 新的):
    assert pick(cands)["idx"] == 旧的, "先确认不修的话确实会选错，否则这个用例没在测东西"
    kept, rejected = reject_planes_on_camera(cands, RATIO)
    assert 旧的 in rejected, f"{clip}：贴着相机那个候选该被丢掉"
    assert pick(kept)["idx"] == 新的, f"{clip}：丢掉之后该选中 idx{新的}"


def test_正常片段一个候选都不丢():
    """高度都在同一量级时，规则不该动任何东西。"""
    cands = [cand(0, 9000, 0.9932), cand(1, 4000, 1.4), cand(2, 2000, 0.8)]
    kept, rejected = reject_planes_on_camera(cands, RATIO)
    assert rejected == []
    assert kept == cands
    assert pick(kept)["idx"] == 0


def test_全被丢完就原样返回():
    """宁可不改，也不要改瞎——留不下候选时必须退回原集合。"""
    cands = [cand(0, 100, 1e-9), cand(1, 50, 1e-9)]
    kept, rejected = reject_planes_on_camera(cands, RATIO)
    assert rejected == []
    assert kept == cands


def test_空集合不炸():
    assert reject_planes_on_camera([], RATIO) == ([], [])


def test_阈值是无量纲的():
    """整体乘一个常数，结论必须不变——重建是相对尺度，绝对米制阈值在这里没意义。"""
    base = [cand(0, 11350, 0.0768), cand(1, 5296, 1.4668), cand(2, 2695, 3.4626)]
    for k in (0.001, 1.0, 1000.0):
        scaled = [cand(c["idx"], c["inliers"], c["cam_height"] * k) for c in base]
        kept, rejected = reject_planes_on_camera(scaled, RATIO)
        assert rejected == [0], f"乘 {k} 之后结论变了，说明判据带了量纲"
        assert pick(kept)["idx"] == 1
