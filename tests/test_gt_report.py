

def test_运动类型准确率按物体汇总而不是按片段():
    """一个片段只配上一个物体时，它的准确率只能是 0 或 1。按片段取中位数的话，
    这种片段会把整列带偏——实测二十条片段共 35 个配对物体、判对 5 个（0.143），
    逐片段中位数却是 0.000。文档里本来就写了要按物体汇总，实现没跟上。"""
    from gwm.feedback.report import aggregate
    runs = [
        {"clip": "a", "metrics": {"full": {"motion_type": {"accuracy": 0.0, "n": 1, "confusion": {}}}}},
        {"clip": "b", "metrics": {"full": {"motion_type": {"accuracy": 0.0, "n": 1, "confusion": {}}}}},
        {"clip": "c", "metrics": {"full": {"motion_type": {"accuracy": 0.0, "n": 1, "confusion": {}}}}},
        # 这一条配上了十个物体，判对八个
        {"clip": "d", "metrics": {"full": {"motion_type": {"accuracy": 0.8, "n": 10, "confusion": {}}}}},
    ]
    report = aggregate(runs)
    acc = report["motion_type_accuracy"]
    assert acc["n"] == 13, "样本量该是配对上的物体数，不是片段数"
    assert abs(acc["mean"] - 8 / 13) < 1e-6, f"加权之后该是 8/13，实际 {acc['mean']}"


def test_没有配对的片段不参与汇总():
    from gwm.feedback.report import aggregate
    runs = [{"clip": "a", "metrics": {"full": {"motion_type": {"accuracy": None, "n": 0}}}},
            {"clip": "b", "metrics": {"full": {"motion_type": {"accuracy": 1.0, "n": 2}}}}]
    acc = aggregate(runs)["motion_type_accuracy"]
    assert acc["n"] == 2 and acc["mean"] == 1.0
