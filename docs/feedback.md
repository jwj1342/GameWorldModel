# 视觉反馈与 agent 编排

> 渲染比对如何变成 VLM 能执行的修订指令，循环怎么组织，用什么脚手架，以及为什么不做角色扮演式多智能体。证据来自 2026-09-18 的调研（`../survey/04`）。

## 1. 三条原则

1. **结构化胜过自由评论**。把参考截图与自身截图一起丢回给模型"自我修订"收益很小（Design2Code：GPT-4V 提升 3 个点，Claude 3 Opus 无变化）。有效的做法是可定位的差异清单：SEIG 的 checklist、Scenix 的逐物体 pass/fail 子句、VF-Coder 的元素级不匹配。
2. **数值先行，VLM 收尾**。先用免 LLM 的指标筛出失败对象，VLM 只对失败区域做定位与建议。M³-Verse 显示多模态模型在成对观测的状态变化检测上普遍薄弱，所以要给它叠加差异图、分块编号与 IoU 数值，而不是让它自由比较两张图。
3. **有界轮数、候选选优、允许回退**。OpenGame 在第三轮出现平台期，UI2Code^N 五轮提升 8 个点；BlenderAlchemy 与 BlenderGym 用成对比较选优并允许回退，低预算时多生成、高预算时多验证。

## 2. 渲染输出

与视频对齐的 K 个关键帧（K 取 8~16），每帧三个 pass：rgb、depth、id。实现见 `runtime.md` 第 4 节。

## 3. 数值层

| 指标 | 捕捉什么 | 工具 | 初始阈值 |
|---|---|---|---|
| 逐物体 mask IoU | 缺失、多余、位置、尺度 | 渲染 ID 图 vs SAM 3.1 掩码 | FAIL < 0.5 |
| 质心像素偏差与 3D 质心偏差 | 位置、运动相位 | ID 图质心 vs 掩码质心；Umeyama 对齐后 | FAIL > 0.5 m 或 > 5% 画幅 |
| 轨迹 ATE（动态物体） | 运动类型、周期、幅度 | 程序 `pose(t)` vs 证据 OBB 序列 | FAIL > 0.5 m |
| 深度 SILog / AbsRel（物体区域） | 尺度、前后关系 | 渲染深度 vs ViPE 深度，归一化 | FAIL > 0.3 |
| 相机重投影误差 | 相机轨迹 | 程序相机 vs ViPE 位姿 | 报告，不单独判 FAIL |
| DINOv3 全局余弦 | 布局与结构 | facebookresearch/dinov3 | 趋势指标 |
| DreamSim | 中层布局与姿态 | ssundaram21/dreamsim | 趋势指标 |
| 时序一致性 | 出现区间不匹配、ID 交换 | 程序出现区间 vs 观测区间 | 报告 |

不用 CLIP 作主指标（对布局不敏感），LPIPS / SSIM 只看趋势（渲染与真实风格差异大时噪声高）。

**子句生成**：每个对象、每项检查一条 `{object_id, check, status, evidence, hint}`；`hint` 由规则给出，如周期误差大提示 `suspect period`，IoU 低但质心对提示 `suspect scale`。

## 4. VLM 批评

- 输入：仅失败子句对应的对象；每个对象一组裁剪对（视频帧与渲染帧同一区域）、一张差异叠加图、指标数值、当前程序中该对象的节点。
- 输出限定为结构化 JSON：`{object_id, issue ∈ {missing, extra, pose, scale, timing, class}, evidence, suggested_edit}`，其中 `suggested_edit` 是 JSON Patch 片段。
- 禁止自由评语；提示词要求引用具体数值与区域。

## 5. 候选搜索与选优

每轮 Writer 生成 2~4 个候选补丁 → 全部编译渲染 → 数值分排序 → 前两名交给 VLM 成对比较 → 选优；若最优不如上一轮则回退并换 hint。默认两轮，上限三轮。

## 6. 编排

三个逻辑角色，一到两个模型，按阶段而非"职位"分解：

```
Perception aggregator（纯代码）
      │ evidence.json
      ▼
Writer（VLM）──► program.json ──► compile ──► render ──► metrics ──► clauses
      ▲                                                                 │
      └──────────── Critic（VLM，只看 FAIL 子句）◄──────────────────────┘
      ▼ 通过
Gameplay binder（规则 + 少量 VLM）
```

Writer 与 Critic 可以是同一个模型（SEIG 用一个模型兼任 generator 与 verifier）。Perception 与 binder 不用 LLM。

**脚手架**：Pydantic AI + 手写 evaluator-optimizer 循环。理由与对比：

| 框架 | 版本 | Claude API 与本地 vLLM 均可 | 工具返回图片给模型 | 评价 |
|---|---|---|---|---|
| **Pydantic AI** | 2.45 | ✓ | ✓（`BinaryContent` / `ToolReturn`） | **选用**；类型化输出契合结构化报告；API 面小 |
| OpenAI Agents SDK | 0.22 | ✓（chat completions 接 vLLM） | ✓ | 备选；多模态在非 OpenAI 端点"因提供商而异" |
| Claude Agent SDK | 0.2.x | 否，仅 Claude | ✓ | 只跑 Claude 的对照组 |
| LangGraph | 1.x | ✓ | 未核实 | 对三阶段回路过度工程 |
| CrewAI / MetaGPT | — | ✓ / 部分 | 未核实 / 否 | 角色扮演范式，恰是文献不推荐的 |
| smolagents | 1.26 | ✓ | 有 vision 教程 | CodeAgent 适合执行度量脚本；更新放缓 |
| 手写循环 | — | ✓ | 自行拼 | Anthropic 的建议：先用 prompt chaining / evaluator-optimizer 等可组合模式 |

可选：用 DSPy / GEPA 离线优化 Writer 与 Critic 的提示词。

## 7. 为什么不用多智能体

| 证据 | 结论 |
|---|---|
| Cemri et al., Why Do Multi-Agent LLM Systems Fail（NeurIPS 2025） | 七个框架失败率 41%~87%；四分之一失败源于验证缺失或错误；给 ChatDev 加高层目标验证提升 15.6% |
| How Generation Architecture Shapes Code Complexity（2026.06） | 六种架构通过率无显著差异；角色拆分让代码复杂度膨胀 50%~130%；有价值的是执行落地的 debugger |
| AgentCoder | 少角色 + 独立测试生成器优于多角色，token 少一半以上 |
| Anthropic 多智能体研究系统 | 并行广度搜索受益，但 token 约 15 倍；编码任务少有真正可并行的子任务 |
| OpenGame（2026.04） | 单 agent 六阶段流水线 + 模板库 + 有界修复，第三轮平台期 |
| GameCraft-Bench / GameDevBench（2026） | 查看渲染截图的 agent 更常成功；视觉反馈稳定提分（41% → 52%） |
| AVR-Agent（2025.08） | 迭代显著优于一次生成，但更多模态的反馈没有再提升；反馈可操作性是瓶颈 |

## 8. 相关工作的反馈机制

| 工作 | 信号 | 转化方式 | 轮数 | 收益 |
|---|---|---|---|---|
| SceneCraft（ICML 2024） | 渲染图 + 描述 → GPT-4V | 指出未满足的空间约束 | 内环多轮 | 去掉内环约束分 88.9 → 26.1 |
| BlenderAlchemy（ECCV 2024） | 渲染 vs 目标图 | 成对锦标赛选优，可回退 | 8×4 | 人类偏好 73% |
| BlenderGym（CVPR 2025） | 像素、N-CLIP、Chamfer | brainstormer + editor + verifier | 3×4 | 验证器查询越多越好 |
| SEIG（2026.06） | 渲染 vs 参考图 | checklist 式待办；几何 → 材质 → 构图 → 光照分阶段 | ≤ 13 | DINO .719 vs .622 |
| Scenix（2026.08） | BEV 代理图 | 逐物体子句 → 最小编辑 → 逐子句 pass/fail | 有界 | F1 +0.02~0.04 |
| Design2Code | 参考 + 自身截图 + 代码 | 自由文本修订 | 1 | 微弱 |
| UI2Code^N（2025.11） | GLM-4.5V verifier | round-robin 成对比较 | 1~5 | 66 → 74% |
| VF-Coder（2026.04） | 截图 + 可访问性树 + 像素色差 + 视觉评分模型 | 汇总为 bug 描述 | ≤ 10 | 21.7 → 28.3% |
| SpatialGrammar（2026.04） | 编译器约束 | 编译错误直接回喂 | 迭代 | 小模型接近大模型 |
| RLRF / cadrille / Visual-SDPO | 渲染或执行奖励 | GRPO / DPO | 训练期 | 显著优于 SFT |

## 9. 试玩层的反馈

见 `runtime.md` 第 6 节：轨迹回放 + 状态判定 + VLM 评审；不做实时游玩 agent。
