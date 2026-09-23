# Candidate 自动 Repair

`gwm.compiler.repair.repair_candidate(program, evidence, diagnostics, config)` 返回三个值：独立的 Program 副本、带验证前后结果与逐项修改记录的报告、仍未解决的诊断。`diagnostics` 可传 `validate_candidate(program)` 的结果；传入的诊断必须与当前 Program 的重新验证结果一致。

`configs/default.yaml` 中的 `candidate_repair.mode` 默认为 `"off"`。`"report"` 会在副本上试算，并将建议写入 `proposed_repairs`，返回的 Program 不变；`"apply"` 最多执行 `max_passes` 轮验证、修复、再验证。Writer 仅在模式不为 `"off"` 时调用该入口。

自动规则限定为接近单位的有限四元数归一化，以及明确声明支撑关系的静态物体沿世界 Y 轴作小幅贴面修正。贴面还要求 Evidence 质量为 `proceed`、同一 clip、米制 Y 向上坐标、可靠且一致的对象几何与接触观测，以及唯一、近水平的静态支撑面。Evidence 的相对尺度不会用于米制贴面。其他诊断保留在 `unresolved`；每项实际修改均记录规则、原值、新值、Evidence 引用、置信度和原因。

例子：

```python
from gwm.compiler.validate import validate_candidate
from gwm.compiler.repair import repair_candidate

initial = validate_candidate(program)
fixed, report, unresolved = repair_candidate(program, evidence, initial, config)
```
