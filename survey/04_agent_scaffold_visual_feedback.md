# 调研 04：Agent 脚手架、多智能体分解与视觉反馈回路（2026-09-18）

> 由调研 agent 生成；版本号/日期来自官方 docs、GitHub releases 或 arXiv 页面。无法核实的条目标注 **未核实**。

## 0. 结论速览

- **脚手架**：原型要在 Claude API 与本地 vLLM Qwen-VL 之间切换，Claude Agent SDK 不合适（官方明确不支持路由到非 Claude 模型）。推荐 **Pydantic AI（v2.45.0，2026-09-17）**：原生 `VLLMProvider`/`AnthropicModel`，工具可通过 `ToolReturn(content=[BinaryContent(png)])` 把渲染截图回传给模型，MCP 支持，API 表面小。备选 OpenAI Agents SDK（v0.22.3）。编排层不必上 LangGraph，Anthropic 自己的建议是 evaluator-optimizer 等可组合模式先手写。
- **多智能体**：文献不支持角色扮演式多智能体（ChatDev/MetaGPT 类）；支持的是按阶段分解 + 执行/渲染落地的验证器（generator–verifier）。建议 3 个逻辑角色（Perception → Program writer → Render-critic）串成 evaluator-optimizer 回路。
- **视觉反馈**：单纯把截图丢回给模型"自我修订"收益很小（Design2Code）；有效的做法是结构化、可定位的差异清单（SEIG 的 checklist、Scenix 的逐物体 pass/fail 子句、VF-Coder 的元素级不匹配）+ 数值指标（DINO/DreamSim/mask IoU/depth）作为门控与排序，2–5 轮即饱和（OpenGame 第 3 轮平台期，UI2Code^N 5 轮 +8pt）。

## 1. Agent 脚手架

| 框架 | 当前版本/状态 | 模型无关（Claude + vLLM Qwen-VL） | 工具返回图片给模型 | 子智能体 | 沙箱执行 | MCP | 评价 |
|---|---|---|---|---|---|---|---|
| Claude Agent SDK (Py/TS) | Python 0.2.156；SDK 1.0 于 2025-09-29 | **否**，仅 Anthropic API / Bedrock / Vertex / Foundry | 是（in-process MCP 工具返回 image 块） | 是 | 权限系统 + 内置 Bash/文件工具 | 是 | 最成熟的"编码 agent 即库"，但绑死 Claude；可作仅 Claude 路径的对照组 |
| OpenAI Agents SDK (Py) | v0.22.3（2026-09-17） | 是（`OpenAIChatCompletionsModel` 接 vLLM；litellm 扩展） | 是（`ToolOutputImage`） | 是（handoffs） | sandbox 模块 **未核实** | 是 | 轻量；多模态在非 OpenAI 提供商上"varies by provider" |
| **Pydantic AI** | v2.45.0（2026-09-17） | **是**（`AnthropicModel`、`OpenAIChatModel`+`VLLMProvider`） | **是**（`BinaryContent` / `ToolReturn`） | agent-as-tool / pydantic-graph | 无内置 | 是 | **推荐主选**：类型化输出天然契合结构化错误报告 |
| LangGraph | 1.0 GA 2025-10-22 | 是 | ToolMessage 图片块 **未核实** | 子图 | 无 | 是 | 持久化强，但对 3 阶段回路过度工程 |
| Microsoft Agent Framework | 1.0 于 2026-04-03；AutoGen 2025-10 进维护模式 | 主要 OpenAI/Azure | 未核实 | 是 | 无 | 是 | 企业/.NET 取向 |
| CrewAI | 1.15.17 | 是（LiteLLM） | 未核实 | 是 | 无 | 是 | 角色扮演范式，恰是文献不推荐的方向 |
| MetaGPT | 70.5k★；README 新闻停在 2025-03 | 部分 | 否 | 是 | 无 | 未核实 | 学术参考价值 > 工程价值 |
| smolagents (HF) | v1.26.0（2026-05-29） | 是 | 有 vision agent 教程 | 是 | 是（E2B/Modal/Docker） | 是 | CodeAgent 适合"执行图像度量脚本"，更新放缓 |
| Google ADK | v2.9.2（2026-09-18） | 是（`LiteLlm` 接 vLLM） | 未核实 | 是 | 无 | 是 | Gemini 优化 |
| LlamaIndex Workflows | 2.24.0 | 是 | 未核实 | 事件驱动 | 无 | 是 | 生态偏 RAG |
| DSPy / GEPA | GEPA（ICLR 2026）比 GRPO 高至 20%、rollout 少 35× | 是 | `dspy.Image` **未核实** | – | – | – | 用途单一：优化 critic/writer 的 prompt |
| 手写循环 | – | 完全可控 | 自行拼 messages | – | – | – | Anthropic《Building effective agents》：最成功的实现"weren't using complex frameworks" |

来源：https://code.claude.com/docs/en/agent-sdk/overview ；https://openai.github.io/openai-agents-python/ ；https://pydantic.dev/docs/ai/ ；https://github.com/pydantic/pydantic-ai/releases ；https://github.com/microsoft/agent-framework ；https://github.com/huggingface/smolagents/releases ；https://adk.dev/agents/models/vllm/ ；https://dspy.ai/tutorials/gepa_ai_program/ ；https://www.anthropic.com/engineering/building-effective-agents

**推荐**：Pydantic AI 作为唯一脚手架，`Agent[Deps, SceneProgramPatch]` 多个实例共享同一套工具（`render(program) -> ToolReturn[截图+指标JSON]`、`compare(frame_id)`、`run_metrics()`）；模型通过配置在 `anthropic:claude-*` 与 `openai:qwen3-vl@vllm` 间切换。沙箱：three.js 无头渲染放在独立 Node/Playwright 子进程（Slurm 作业内）。

## 2. 多智能体是否有益

| 证据 | 结论 | 数字 |
|---|---|---|
| Cemri et al. "Why Do Multi-Agent LLM Systems Fail" (NeurIPS 2025) https://arxiv.org/abs/2503.13657 | 7 个框架失败率 41%–86.7%；三类：系统设计 43.8%、智能体间错位 31.95%、任务验证 24.25% | 给 ChatDev 加高层目标验证：+15.6% |
| "How Generation Architecture Shapes Code Complexity" (2026-06) https://arxiv.org/html/2606.00308 | 6 种架构 HumanEval 84–92% 无显著差异；角色拆分让代码复杂度膨胀 50–130%；有价值的是执行落地的 debugger | – |
| AgentCoder https://arxiv.org/abs/2312.13010 | 少角色 + 独立测试生成器优于多角色 | token 56.9K vs 138.2K vs 183.7K |
| Anthropic 多智能体研究系统 | 并行广度搜索 +90.2%，但 token ≈15×；编码任务少有真正可并行的子任务 | – |
| **OpenGame** (2026-04, v2 2026-09) https://arxiv.org/abs/2604.18394 | 单 agent 六阶段流水线；有界修复 ≤5 轮，第 3 轮平台期 | 零样本 BH 58.4 → 全流程 72.4；模板库 +5.8 |
| **GameCraft-Bench** (2026-06, Godot 4) https://arxiv.org/abs/2606.17861 | 测 7 个单 agent CLI；查看渲染截图的 agent 更常成功 | 最佳 Opus-4.7 仅 41.46% |
| **GameDevBench** (2026-02, Godot 4, 333 任务) https://arxiv.org/abs/2602.11103 | 编辑器截图 MCP + 运行时视频反馈 | GPT-5.4 加视觉反馈 41.1%→52.0%；Gemini 3 Pro 53.8% |
| AVR-Agent (2025-08) https://arxiv.org/abs/2508.00632 | 迭代生成显著优于一次生成；高质量资产与 omni-modal 反馈没有带来进一步提升，反馈可操作性是瓶颈 | 终版胜初版 0.647 |
| SceneConductor (2026-06) https://arxiv.org/abs/2606.08402 ；VULCAN (2025-12) https://arxiv.org/abs/2512.22351 | 3D 场景侧"多智能体"实为 planner + 局部专家 + verifier；VULCAN 强调用 MCP 函数级 API 代替脆弱的代码改写 | – |

**我们管线的合理分解**：按阶段而非"职位"拆：(1) Perception aggregator（纯代码）；(2) Program writer（VLM，输出 DSL）；(3) Render-critic（VLM + 数值指标，输出结构化差异清单）；(4) Gameplay binder（模板匹配，规则为主）。(2)(3) 构成 evaluator-optimizer 回路，同一模型可兼任（SEIG 用一个 Claude Opus 4.7 做 generator 与 verifier）。

## 3. 视觉反馈机制

| 工作 | 反馈信号 | 转化为文本/奖励 | 迭代 | 报告收益 |
|---|---|---|---|---|
| SceneCraft (ICML 2024) https://arxiv.org/abs/2403.01248 | 渲染图 + 场景描述 → GPT-4V | 指出未满足的空间约束 → 修改约束函数 | 内环 N 轮 | 去掉内环：约束分 88.9→26.1 |
| BlenderAlchemy (ECCV 2024) https://arxiv.org/abs/2404.17672 | 渲染图 vs 目标图 | VLM 成对锦标赛选优，tweak/leap 交替，可回退 | b=8, d=4 | 人类偏好 73% |
| BlenderGym (CVPR 2025) https://arxiv.org/abs/2504.01786 | PL、N-CLIP、Chamfer | brainstormer + code editor；verifier 成对选择 | 3×4 树 | 低预算多生成，高预算多验证 |
| **SEIG "Thinking in Blender" (2026-06)** https://arxiv.org/abs/2606.02580 | 渲染 vs 参考单图 | verifier 输出 checklist 式可操作待办；阶段：几何(5轮)→材质(3)→构图(3)→光照(2) | ≤13 轮 | DINO .719 vs .622 |
| **Scenix (2026-08)** https://arxiv.org/abs/2608.07012 | BEV 度量代理图 vs 从输入合成的俯视假设图 | Critic → 逐物体子句 → Editor 最小编辑 → Verify 逐子句 pass/fail | 有界预算 | 回路 F1 +0.016~+0.039；主要收益来自资产落地阶段 |
| Agentic 3D Scene Gen (2025) https://arxiv.org/abs/2505.20129 | 场景画像 + 语义点云 + 超图 | VLM 读写结构化空间上下文 | 迭代 | 未核实 |
| SceneWeaver (NeurIPS 2025) https://github.com/Scene-Weaver/SceneWeaver | 物理/视觉/语义自评 | reason-act-reflect 选工具 | 迭代 | – |
| Design2Code https://arxiv.org/html/2403.03163v3 | 参考截图 + 自身截图 + 代码 → 自由文本修订 | – | 1 | 收益微弱：85.8→88.8；Claude 3 Opus 无变化 |
| **UI2Code^N (2025-11)** https://arxiv.org/html/2511.08195v2 | GLM-4.5V 作 verifier，round-robin 成对比较 | – | 1→5 | 66→74%；VLM rewards 优于 CLIP |
| **VF-Coder (2026-04)** https://arxiv.org/html/2604.19750 | 沙箱截图 + 可访问性树 + 像素色差 + 训练的视觉评分模型 | Planner 汇总为 bug 描述给 Fixer | ≤10 轮 | 21.68→28.29% |
| Visual-SDPO (2026-06) https://arxiv.org/abs/2606.10334 | 渲染缺陷作为教师特权信息 | 缺陷回溯到代码语句加权 + GRPO | 训练期 | +10pt |
| RLRF (SVG, 2025) https://arxiv.org/abs/2505.20793 | L2、Canny、DreamSim、CLIP 加权和 | GRPO 奖励 | 训练期 | SSIM 79.4→95.1 |
| cadrille (CAD) https://arxiv.org/html/2505.22914v1 | 10×IoU − 10×invalid | Dr.CPPO | 训练期 | IoU 87.1→90.2 |
| SpatialGrammar (2026-04) https://arxiv.org/abs/2604.27555 | 编译器反馈（碰撞/约束） | DSL 编译错误直接回喂 | 迭代 | DSL + 编译校验让 104M 小模型接近大模型 |
| 3DCodeBench (2026-06) https://arxiv.org/abs/2606.01057 | 渲染 + 人类成对偏好 | – | 多轮 | 主要失败是 API 不匹配 |

## 4. 具体工具

**(a) 图像对齐指标**
- DINOv3（Meta 2025-08）https://github.com/facebookresearch/dinov3 ：全局 CLS 余弦 + patch 级相似度图，对布局敏感。
- DreamSim https://github.com/ssundaram21/dreamsim ：中层布局/姿态差异；SEIG 与 RLRF 采用。
- LPIPS/SSIM/PSNR：渲染与真实视频风格差异大时噪声高，仅做趋势。CLIP：仅作语义门控。
- 实例掩码 IoU：SAM 3 / 3.1（2026-03）https://github.com/facebookresearch/sam3 与 three.js 逐物体 ID 通道对比。
- 深度：Depth Anything 3（2025-11）https://github.com/ByteDance-Seed/Depth-Anything-3 ；SILog / AbsRel。
- 轨迹 ATE：跟踪掩码质心 vs 渲染物体投影，Umeyama 对齐后 RMSE。

**(b) VLM 结构化批评**
- 开源：Qwen3-VL https://arxiv.org/pdf/2511.21631 、InternVL3.5、GLM-4.5V（UI2Code^N verifier 先例）。
- 证据：BlenderGym 显示多次采样的 InternVL2-8B verifier 可超过单次 GPT-4o；M³-Verse https://arxiv.org/abs/2512.18735 显示 LMM 在成对观测的状态变化检测上普遍薄弱 → 不要依赖 VLM 自由比较两图，要给它叠加差异图 / 分块编号 / 掩码 IoU 数值。
- API：Claude Opus 4.7（SEIG、GameCraft-Bench）、GPT-5.5、GPT-5.4（GameDevBench）、Gemini 3 Pro。

**(c) 可改造仓库**：BlenderAlchemyOfficial（锦标赛逻辑）；SceneWeaver（工具接口）；gamedevbench（编辑器截图 MCP）；blender-mcp。three.js 专用 render-compare-revise 仓库未找到，建议自建几百行的 Playwright harness。

## 5. 游戏试玩智能体

| 项目 | 接口 | 要点 |
|---|---|---|
| VideoGameBench | 截图 + 键鼠；ReAct 带 5–10 帧记事本 | 实时模式推理延迟致状态过期；Lite 模式暂停模拟器 |
| lmgame-Bench / GamingAgent | Gym-style API | 感知/记忆/推理 harness 可开关 |
| BALROG | 文本或视觉观测 | 多数模型给图像后反而更差 |
| **GameCraft-Bench** | 回放 JSON 键鼠轨迹 → 录像 → 2fps 抽帧 → VLM 评审 | 确定性、便宜；适合 gameplay 层验收 |
| OpenGame-Bench | headless 浏览器自动游玩 + 帧熵/运动启发式 + VLM | 面向 web 游戏，与 three.js 最贴近 |
| Voyager | 代码即动作 | 让 agent 写测试脚本而非逐帧操作 |

## 6. 最终推荐

**(i) 脚手架与分解**
1. Pydantic AI（模型层）+ 手写 evaluator-optimizer 循环（编排层）；不上 LangGraph/CrewAI。
2. 三个逻辑角色、一到两个模型：Writer（DSL 生成/修补）、Critic（VLM，只输出结构化 JSON：`{object_id, issue∈{missing,extra,pose,scale,timing}, evidence:{iou,depth_err,dino}, suggested_edit}`），Perception 与 Gameplay-binder 为纯代码。
3. 用 DSPy/GEPA 离线优化 Critic 与 Writer 的 prompt。
4. 借鉴 OpenGame：模板/骨架库 + "debug skill" 记录高频错误；借鉴 SpatialGrammar：DSL 编译期约束检查先于渲染。

**(ii) 视觉反馈设计（baseline）**
1. 渲染输出：颜色帧、深度、物体 ID 掩码，与视频对齐的 K≈4–6 个关键帧。
2. 数值层（免 LLM）：每物体 mask IoU、SILog 深度误差、质心 ATE、全局 DINOv3 余弦、DreamSim；生成逐物体 pass/fail 子句（Scenix 风格）。
3. VLM 层：仅对 fail 子句所在区域裁剪/叠加差异图后询问 Critic，输出 checklist（SEIG 风格），禁止自由评语。
4. 搜索：每轮 Writer 生成 b=2–4 个候选补丁，用数值分 + VLM 成对比较选优并允许回退；轮数 2–3 轮。
5. gameplay 层：GameCraft-Bench 式轨迹回放录像 + 抽帧 VLM 评审，而非实时游玩 agent。

## 7. 未核实清单
OpenAI Agents SDK sandbox 模块；LangGraph ToolMessage 图片块；MAF 对 Anthropic/OpenAI 兼容端点支持；DSPy 当前版本；MetaGPT 2026 维护状态；CADCrafter/CAD-Coder 细节；Gemma 3/Kimi-VL/MiMo-VL 双图差异能力；OpenGame/AVR-Agent 代码链接；GamingAgent 本地模型支持；"4D 程序"类直接对应工作。
