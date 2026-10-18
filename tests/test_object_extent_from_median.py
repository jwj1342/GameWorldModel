"""动态物体的尺寸取逐帧中位，不取第一帧。

物体的尺寸不随时间变，而逐帧 OBB 是从带噪声的点云量出来的。原来 `direct.py`
给动态物体取 `obb[0]["size"]`——只信第一帧，而同一个文件里结构件走的是逐帧中位。

实测 300 个物体里 51 个（17%）的首帧最长边和逐帧中位差了四分之一以上。最狠的
是 test_00000 的 spool_6：首帧量出 4.34 m，逐帧中位只有 0.93 m。按首帧生成的话，
那一个 spool 会被拉成 4.34 米挡住半个场景，而且大到拖不动（#64）。

位置和朝向仍然取第一帧——`motion` 描述的轨迹是从那一帧的位姿出发的，换一帧
会和运动对不上。这条测试同时盯着这一点，防止「都改成中位」的过度修正。
"""
from gwm.synthesis.direct import evidence_to_program


def evidence_with_sizes(sizes, centers=None):
    """一个物体，逐帧给定尺寸；中心默认不动。"""
    n = len(sizes)
    centers = centers or [[0.0, 0.5, 0.0]] * n
    obb = [{"center": list(c), "quat": [0, 0, 0, 1], "size": list(s), "conf": 0.9, "t": float(i)}
           for i, (s, c) in enumerate(zip(sizes, centers))]
    return {"meta": {"clip": "c", "duration": float(n - 1), "scale": "relative"},
            "camera": {"intrinsics": {"fov_deg": 60, "width": 640, "height": 360},
                       "poses": [{"t": float(i), "pos": [0, 1.2, 5], "quat": [0, 0, 0, 1]}
                                 for i in range(n)]},
            "static": {"planes": [{"kind": "ground", "conf": 0.8, "center_hint": [0, 0, 0],
                                   "extent_hint": [10, 0.2, 10]}]},
            "objects": [{"id": "spool_6", "class_guess": "spool", "confidence": 0.9,
                         "is_dynamic": True, "obb": obb,
                         "motion_guess": {"type": "trajectory", "conf": 0.9,
                                          "axis": [1, 0, 0], "speed": 0.5},
                         "contacts": [], "mask_area_frac": 0.03}],
            "keyframes": []}


def only_object(program):
    nodes = [n for n in program["objects"] if n["id"] == "spool_6"]
    assert nodes, f"物体没进程序：{[n['id'] for n in program['objects']]}"
    return nodes[0]


def test_首帧偏大不会把物体撑成巨物():
    # 第一帧量糊了（4.34 m），后面十帧稳定在 0.93 m 上下
    sizes = [[4.34, 4.34, 4.34]] + [[0.93, 0.90, 0.95]] * 10
    node = only_object(evidence_to_program(evidence_with_sizes(sizes), "c", {}))
    longest = max(node["geom"]["extent"])
    assert longest < 1.1, f"尺寸应该落在中位 0.93 附近，实际 {longest}"


def test_首帧偏小不会把物体压成一点():
    sizes = [[0.04, 0.04, 0.04]] + [[0.80, 0.80, 0.80]] * 10
    node = only_object(evidence_to_program(evidence_with_sizes(sizes), "c", {}))
    assert min(node["geom"]["extent"]) > 0.5


def test_位置仍然取第一帧而不是中位():
    """尺寸换成中位，位置不能跟着换——motion 的轨迹是从首帧位姿起算的。"""
    sizes = [[1.0, 1.0, 1.0]] * 5
    centers = [[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [2.0, 0.5, 0.0],
               [3.0, 0.5, 0.0], [4.0, 0.5, 0.0]]
    program = evidence_to_program(evidence_with_sizes(sizes, centers), "c", {})
    node = only_object(program)
    # 运动可能把 pose.pos 换成轨迹基点，所以这里只断言它不是中位（2.0）
    assert abs(node["pose"]["pos"][0] - 2.0) > 1e-6, "位置被误改成了逐帧中位"
