# 可编辑性实验

issue #13。论文主张我们的产物可以被读、被改、被重新组合，而三维重建和视频世界模型
做不到这件事。这个目录放把主张变成数字所需要的东西。

`instructions.json` 是二十条编辑指令，每条带机器可判的期望：该改哪个路径、改成什么值，
或者这条指令应当被拒绝。四条「应当拒绝」的用来检验模型会不会为了交差而编造改动。

跑法：

```bash
python scripts/run_editability.py
```

它对每条指令都从原始程序重新出发，互不影响，最后报两个数：可执行指令的通过率，
以及应当拒绝的有没有被正确拒绝。明细写到 `docs/results/editability.json`。

判定逻辑在 `scripts/run_editability.py` 的 `judge()`，它本身有单元测试
（`tests/test_pipeline_logic.py::test_editability_judge_catches_side_effects_and_fabrication`），
因为测量工具不可信的话跑出来的数字就没意义。

这里只放实验的定义，别放结果。结果统一在 `docs/results/` 下。
