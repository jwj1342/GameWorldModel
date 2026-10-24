"""世界对齐退化要出声，不要一声不响地产出一个空场景。

实测（ab015 的 test_00000）：`align_world` 选中的「地面」让相机离地只有 0.0768
（相对单位），标定倍数被推到 **15.62**。物体全落到地面以下，下陷闸丢掉 18 个里的
15 个，这条片段最终产出 **0 个物体**。

丢掉是对的——地面都估错了，位置不可能对，产出十五个摆错的物体比不产出更糟。
**问题是整个过程没有任何地方说这条片段的对齐退化了。** 在评测里它表现为「召回 0」，
和「检测什么都没找到」长得一模一样，而两者该做的事完全不同：前者要去修对齐，
后者要去修检测。

这条诊断不改任何行为，只是让这个失败可见。
"""
from gwm.perception.quality import assess_evidence_quality


def base_evidence(scale_factor=None, cam_h=None):
    meta = {"clip": "c", "duration": 2.0, "scale": "relative"}
    if scale_factor is not None:
        meta["alignment"] = {"scale_factor": scale_factor,
                             "camera0_height_before_scale": cam_h}
    return {
        "meta": meta,
        "camera": {"intrinsics": {"fov_deg": 60, "width": 640, "height": 360},
                   "poses": [{"t": float(t), "pos": [0, 1.2, 5], "quat": [0, 0, 0, 1]}
                             for t in (0.0, 1.0, 2.0)]},
        "static": {"planes": [{"kind": "ground", "conf": 0.8, "center_hint": [0, 0, 0],
                               "extent_hint": [10, 0.2, 10]}]},
        "objects": [], "frames": [], "keyframes": [],
    }


def codes(report):
    return {d["code"] for d in report["diagnostics"]}


def test_标定倍数爆掉要报出来():
    r = assess_evidence_quality(base_evidence(scale_factor=15.62, cam_h=0.0768), {})
    assert "world_alignment_degenerate" in codes(r)
    msg = next(d for d in r["diagnostics"] if d["code"] == "world_alignment_degenerate")
    assert "15.62" in msg["message"], "要把实际倍数写进去，方便直接判断有多离谱"


def test_正常的标定倍数不报():
    for factor, h in ((1.2005, 0.9995), (4.3812, 0.2739), (1.0, 1.2)):
        r = assess_evidence_quality(base_evidence(scale_factor=factor, cam_h=h), {})
        assert "world_alignment_degenerate" not in codes(r), f"{factor} 不该报"


def test_没有对齐记录时不报():
    """老产物没有这个字段，不能因此就说它退化。"""
    r = assess_evidence_quality(base_evidence(), {})
    assert "world_alignment_degenerate" not in codes(r)


def test_这条诊断不改变任何行为():
    """它只是让失败可见。退化的片段仍然走完整条链路，decision 不因此变成 block。"""
    r = assess_evidence_quality(base_evidence(scale_factor=15.62, cam_h=0.0768), {})
    assert r["decision"] != "block"
