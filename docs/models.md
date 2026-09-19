# 模型层

> 系统中每个用到模型的位置、候选模型对视觉输入的支持程度、在 L40S 上的部署方式、许可与成本，以及 baseline 之后训练要用的框架与数据。定价与模型 ID 为 2026-09-18 查询并对照官方参考核对的结果。

## 1. 模型层总览

| 角色 | 输入 | 输出 | 基线选择 | 备选 |
|---|---|---|---|---|
| 程序生成（Writer） | 8~16 张关键帧 + evidence.json + DSL schema + 错误报告 | program.json / JSON Patch | `claude-opus-5`（API） | `claude-fable-5-1`（难例）；Qwen3-VL-32B-Instruct FP8（自托管） |
| 批评（Critic） | 失败子句对应的裁剪对 + 差异叠加图 + 指标 | 结构化 checklist | 同 Writer 的模型 | GLM-4.6V / InternVL3.5（UI2Code^N 与 BlenderGym 有用开源模型做 verifier 的先例） |
| 类别候选 | 首帧 | 名词短语列表 | Writer 的模型 | Molmo 2（视频指点） |
| 试玩评审 | 2 fps 抽帧序列 + state 日志 | 可玩性评分 | `claude-sonnet-5` 或 Writer 的模型 | — |
| 感知 | 视频 | 位姿、深度、掩码、轨迹 | ViPE、SAM 3.1、SpatialTrackerV2 | 见 `perception.md` |
| 对齐指标 | 渲染帧 vs 视频帧 | 相似度 | DINOv3、DreamSim | CLIP、LPIPS |
| 素材检索 / 生成 | 裁剪图 + 类别 | GLB | CLIP / SigLIP 检索；TRELLIS.2-4B | 见 `assets.md` |

## 2. API 模型

| 模型 | 模型 ID | 视觉输入 | 上下文 / 最大输出 | 价格（每百万 token，输入 / 输出） | 备注 |
|---|---|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | 图像（视频需抽帧） | 1M / 128K | $5 / $25 | **基线默认**；adaptive thinking 默认开启；结构化输出用 `output_config.format` |
| Claude Fable 5.1 | `claude-fable-5-1` | 图像 | 1M / 128K | $10 / $50 | 难例；thinking 常开；不支持强制 tool_choice |
| Claude Sonnet 5 | `claude-sonnet-5` | 图像 | 1M / 128K | $2 / $10 | 试玩评审等轻任务 |
| GPT-5.5 / 5.6 | — | 图像 | — | — | 日期与视频输入未核实 |
| Gemini 3.1 Pro | — | **原生视频与音频**，单次最多 900 图 | 1M | — | 唯一原生吃视频的 API；若关键帧抽样不够可作对照 |

Claude 系列不接受视频文件，需抽关键帧作为多图输入；这与本系统"关键帧 + 证据 token"的设计一致。

## 3. 开源 VLM（可用 vLLM 自托管）

"视频"一列指 vLLM 是否支持原生多帧视频输入。显存按 BF16 权重估算，KV cache 另计。

| 模型 | 发布 | 尺寸 | 许可 | 图像 | 视频 | L40S 48 GB 部署 | 评价 |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL** | 2025-09~10 | 2B / 4B / 8B / 32B；30B-A3B、235B-A22B MoE | Apache-2.0 | ✓ | ✓ 原生，默认 2 fps，时间戳对齐 | 8B → 1 卡；32B → 2 卡（FP8 1 卡紧）；30B-A3B FP8 → 1 卡 | **自托管基线 32B FP8；微调对象 8B** |
| **Qwen3.5** | 2026-02 | 0.8B ~ 27B 稠密；35B-A3B、122B-A10B | Apache-2.0 | ✓ | ✓（需 vLLM main） | 9B → 1 卡；27B → 2 卡或 1 卡 FP8 | 原生多模态；Scenix 用 4B / 9B SFT 出场景程序 |
| InternVL3.5 | 2025-08 | 1B ~ 38B；MoE | MIT（代码） | ✓ | ✓ | 8B / 14B → 1 卡；38B → 2 卡 | 备选 critic |
| GLM-4.6V / Flash | 2025-12 | 106B-A12B / 9B | MIT | ✓ | ✓ 128K 长视频 | 9B → 1 卡；106B FP8 → 3 卡 | UI2Code^N 用 GLM-4.5V 作 verifier |
| Gemma 3 27B / Gemma 4 | 2025-03 / 2026 | 27B；2B ~ 31B | Gemma 条款 | ✓ | Gemma 3 否；Gemma 4 未核实 | 27B → 2 卡 | — |
| Kimi-VL-A3B | 2025-06 | 16B-A3B | MIT | ✓ | 模型卡支持，vLLM 表未列 | 1 卡 | — |
| MiMo-VL-7B | 2025 | 8B | MIT | ✓ | ✓ | 1 卡 | — |
| Molmo 2 | 2025-12 | 8B / 4B | Apache-2.0 | ✓ | ✓ 视频指点与跟踪 | 1 卡 | 适合当证据器而非程序员 |
| Llama 4 Scout | 2025-04 | 109B-A17B | Llama 4 | ✓ | 否 | INT4 → 2 卡 | 不选 |
| Pixtral 12B | 2024-09 | 12B | Apache-2.0 | ✓ | 否 | 1 卡 | 不选 |

**代码专用文本模型**（可选的二阶段"程序员"：VLM 出场景规格 → 文本模型出代码）：Qwen3-Coder-30B-A3B（Apache-2.0，256K；2 卡 BF16 或 1 卡 FP8）；DeepSeek-V3.2 / V4-Flash（MIT，文本；能否塞进 4 卡未核实）。baseline 采用"VLM 直接出 DSL，编译器出代码"，不需要这一阶段。

## 3b. OpenRouter（MVP 实际采用）

OpenRouter 提供 OpenAI 兼容接口（`https://openrouter.ai/api/v1`），计算节点经 squid 代理可达，一把密钥可切换所有模型。2026-09-19 查询到的带图像输入、约 100B 量级的候选（价格为每百万 token 输入 / 输出）：

| 模型 | 规模 | 价格 | 上下文 | JSON Schema 约束 | 备注 |
|---|---|---|---|---|---|
| `z-ai/glm-4.6v` | 106B-A12B | $0.30 / $0.90 | 131k | 支持 | **默认**；MIT 权重 |
| `qwen/qwen3.5-122b-a10b` | 122B-A10B | $0.26 / $2.08 | 262k | 支持 | 原生多模态，备选 |
| `qwen/qwen3-vl-235b-a22b-instruct` | 235B-A22B | $0.21 / $1.90 | 262k | 支持 | 更大 |
| `moonshotai/kimi-k2.5` | 1T MoE | $0.45 / $2.25 | 262k | 支持 | 带图像输入 |
| `qwen/qwen3.5-27b` | 27B | $0.195 / $1.56 | 262k | 支持 | 与自托管方案同款 |
| `anthropic/claude-opus-4.8` | — | $5 / $25 | — | 支持 | 难例 |

客户端差异：OpenRouter 不接受 vLLM 的 `chat_template_kwargs`；推理开销用统一的 `reasoning: {effort: low}`；`response_format: json_schema` 被拒时客户端自动改为把 schema 内联到系统提示并重试。

## 4. 调用方式

- 脚手架 Pydantic AI：`AnthropicModel` 与 `OpenAIChatModel + VLLMProvider` 通过配置切换；工具返回 `BinaryContent` 把渲染截图回传给模型；输出用 Pydantic 类型约束（程序 JSON、Patch、批评 JSON）。
- Claude 调用：使用官方 SDK；`claude-opus-5` 默认 adaptive thinking；长输出用流式；结构化输出用 `output_config.format` 而不是 prefill。
- 计算节点经 squid 代理可达 `api.anthropic.com` 与 `api.openai.com`（实测），agent 循环可整个放在作业内。
- 自托管：vLLM 起 Qwen3-VL-32B-Instruct-FP8 于 2×L40S，OpenAI 兼容端点；Vulcan 自带的 Aleph 推理服务也可作备选端点。

## 5. 成本估算（粗估，待阶段三实测）

单条片段、三阶段生成、两轮反馈、每轮 2 个候选：

| 项 | 估算 |
|---|---|
| 每次调用输入 | 12 帧 × 约 1.5k token + evidence 5k + schema 与 few-shot 8k ≈ 30k token |
| 调用次数 | 3 阶段 × (1 + 2 轮 × 2 候选) ≈ 15 次 Writer + 4 次 Critic |
| 输入合计 | 约 0.6M token → `claude-opus-5` 约 $3 |
| 输出合计 | 约 60k token → 约 $1.5 |
| 单条片段 | 约 $4~5；十条片段一轮实验约 $50 |

Prompt caching 可以显著降低：schema、few-shot 与 evidence 放在前缀并打缓存断点。自托管 Qwen3-VL 只有 GPU 时费。

## 6. baseline 之后：训练栈与数据

**SFT**：LLaMA-Factory 或 ms-swift（Qwen3-VL、InternVL3.5、GLM-4.6V 均支持，LoRA / QLoRA）。

**RL（非可微的执行与渲染奖励）**：

| 框架 | 多模态 | 算法 | 自定义奖励 | 备注 |
|---|---|---|---|---|
| TRL GRPOTrainer | ✓ | GRPO | 可调用或异步协程奖励函数，多奖励加权 | 最易接"渲染 → 比对"服务 |
| EasyR1（基于 verl） | Qwen2/2.5/3-VL | GRPO / DAPO / RLOO / GSPO | Python 奖励函数 | 7B 全参 4×40 GB，LoRA 2×32 GB |
| verl | ✓ | PPO / GRPO，LoRA | `custom_reward_function` | vLLM / SGLang rollout |
| OpenRLHF | ✓，含多轮 VLM RL | PPO / GRPO / RLOO | HTTP 远程奖励 | Apache-2.0 |

4×L40S 上 7~9B VLM + LoRA + GRPO 可行：2 卡 vLLM rollout（FP8）+ 2 卡训练；奖励服务用 Playwright 渲染 three.js 并算掩码 IoU、深度误差、程序可执行性与 DSL 合法性。先例：RLRF（SVG 渲染奖励）、cadrille（CAD 执行 IoU 奖励，ICLR 2026 Oral）、SimWorlds（4D Blender 程序 + 确定性验证器）。

**微调对象**：Qwen3-VL-8B-Instruct 或 Qwen3.5-9B（Apache-2.0、原生视频、单卡推理、全部框架支持）。

**数据集**：

| 数据集 | 类型 | 规模与标注 | 备注 |
|---|---|---|---|
| Kubric MOVi-E/F | 合成动态多物体视频 | 每集约 9.75k 段；24 帧；实例分割、深度、光流、逐帧相机、物体位姿与速度、3D 框 | 物体位姿序列可直接编译成 DSL 程序做 SFT |
| OmniWorld-Game | 游戏引擎录制 | 96k 段 1280×720@24fps；深度、位姿、光流、前景掩码 | 许可未核实 |
| XScene（Scenix） | 室内静态场景程序 | 约 110k 场景；程序 = 房间蓝图 + 物体类别 / 中心 / 尺寸 / 朝向 / 支撑 | 发布未核实；schema 值得借鉴 |
| ViPE 标注集 | 真实视频伪真值 | DynPose-100K++ 约 99.5k 段 | HF nvidia/* |
| GF-Minecraft | 游戏 + 键鼠 | 70 小时 | 许可未核实 |
| CARLA | 可自录 | 位姿、深度、分割、3D 框 | MIT |
| 自建 VidProg-Synth | three.js 程序化生成 | 真值程序 + 逐帧渲染 | RP 4.1；阶段一的内核与编译器可直接复用为数据引擎 |
