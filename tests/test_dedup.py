"""同一个物体被检出两遍的，合掉一个。

#56 里量过：没配上真值的预测物体里，38% 离最近的真阳性不到一个物体尺度，其中
一部分连类别都相同——那是重复检出。剩下 62% 离得更远，是无中生有，这一步碰不到。
去重的天花板因此是算得出来的：精确率 0.173 -> 0.239。

判据用的是和刻画它时同一个统计量：中心距离除以两者对角线的平均值。阈值必须保守，
因为堆叠的箱子、紧挨的栏杆本来就该是两个物体。
"""
from gwm.synthesis.dedup import merge_duplicates


def node(oid, cls, pos, extent=(0.3, 0.3, 0.3)):
    return {"id": oid, "class": cls,
            "geom": {"kind": "primitive", "shape": "box", "extent": list(extent)},
            "pose": {"pos": list(pos)}}


def test_同类且几乎重合的合掉一个():
    program = {"objects": [node("crate_1", "crate", (0.0, 0.15, 0.0)),
                           node("crate_2", "crate", (0.05, 0.15, 0.02))]}
    merged = merge_duplicates(program, 0.5)
    assert len(merged) == 1
    assert len(program["objects"]) == 1
    assert merged[0]["dropped"] == "crate_2" or merged[0]["dropped"] == "crate_1"


def test_类别不同的挨着也不合():
    # 杯子放在托盘上：两个真的物体，不是重复
    program = {"objects": [node("tray_1", "tray", (0.0, 0.1, 0.0)),
                           node("cup_1", "cup", (0.02, 0.15, 0.01))]}
    assert merge_duplicates(program, 0.5) == []
    assert len(program["objects"]) == 2


def test_同类但隔开的不合():
    # 一个物体尺度之外。堆叠和并排的箱子都该留着
    program = {"objects": [node("crate_1", "crate", (0.0, 0.15, 0.0)),
                           node("crate_2", "crate", (0.6, 0.15, 0.0))]}
    assert merge_duplicates(program, 0.5) == []
    assert len(program["objects"]) == 2


def test_留下证据更多的那个():
    program = {"objects": [node("weak", "crate", (0.0, 0.15, 0.0)),
                           node("strong", "crate", (0.03, 0.15, 0.0))]}
    evidence = {"objects": [{"id": "weak", "obb": [{}] * 3, "confidence": 0.9},
                            {"id": "strong", "obb": [{}] * 30, "confidence": 0.4}]}
    merged = merge_duplicates(program, 0.5, evidence=evidence)
    assert len(merged) == 1
    assert merged[0]["kept"] == "strong", "被看到的帧数优先于置信度"
    assert [n["id"] for n in program["objects"]] == ["strong"]


def test_阈值为零就是关掉():
    program = {"objects": [node("a", "crate", (0.0, 0.15, 0.0)),
                           node("b", "crate", (0.01, 0.15, 0.0))]}
    assert merge_duplicates(program, 0.0) == []
    assert len(program["objects"]) == 2


def test_记录里有合并的依据():
    program = {"objects": [node("a", "crate", (0.0, 0.15, 0.0)),
                           node("b", "crate", (0.05, 0.15, 0.0))]}
    merged = merge_duplicates(program, 0.5)
    assert merged[0]["class"] == "crate"
    assert 0.0 < merged[0]["relative_distance"] <= 0.5, "相对距离要记下来，便于事后核阈值"


def test_去重的缺省阈值进了配置():
    """扫参在十一条片段上做过：0 / 0.3 / 0.5 / 0.7 的 F1 是 0.257 / 0.265 / 0.284 / 0.293，
    而召回 0.500 / 0.481 / 0.481 / 0.462。取 0.5 是因为再往上召回掉得明显。"""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    assert cfg["synthesis"]["dedup_relative_distance"] == 0.5
