"""试玩判定要能说出自己是怎么来的。

玩法绑定正常走 gwm/binding/，槽位写进程序的 binding.slots。内核模板里还有一条兜底
（inferSlots），在槽位缺失时自己猜：把最扁的那块静态几何当地面，出生点放一个角、
目标放对角。走这条路的试玩几乎必然「一出生就通关」——中间没有任何障碍——而且没走
绑定就没做尺度归一化，固定尺寸的试玩角色和场景的比例会错得离谱。

原来这两种试玩产出的 verdict.json 长得一模一样，从结果上分不出来。
"""
from gwm.playtest.autopilot import discount_untrusted


def test_猜出来的槽位不计入通过率():
    v = discount_untrusted({"reached_goal": True, "playable": True, "trustworthy": False,
                            "binding": {"inferred_slots": ["player_spawn", "goal_volume"]}})
    assert v["playable"] is False, "判不了就不该算通过"
    assert v["playable_rule"] is True, "原始的规则判定要保留，不能丢信息"
    assert "untrustworthy_reason" in v


def test_正常绑定的判定原样通过():
    v = discount_untrusted({"reached_goal": True, "playable": True, "trustworthy": True,
                            "binding": {"inferred_slots": []}})
    assert v["playable"] is True
    assert "playable_rule" not in v


def test_旧的判定文件没有这一项时不动它():
    # 没有 trustworthy 字段的历史产物，行为保持不变
    v = discount_untrusted({"reached_goal": True, "playable": True})
    assert v["playable"] is True
    assert "playable_rule" not in v
