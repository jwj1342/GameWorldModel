# 设计笔记

这份文档记录每一层为什么是现在这个样子：当时有哪些候选、比较了什么、最后选了谁。调研的原始报告在 `../survey/`，这里是结论。

实际用了哪些、跑出来什么样，看 [status.md](status.md)。系统怎么搭的，看 [architecture.md](architecture.md)。

## 感知

> 从单目视频得到 `evidence.json` 的模型选择与算法。模型对比数据来自 2026-09-18 的调研（`../survey/05`），硬件按 L40S 48 GB 估算。

### 1. 目标

把视频变成 VLM 能直接使用的结构化证据：相机内参与逐帧位姿、静态结构候选、每个物体的逐帧有向包围盒、运动类型猜测、接触关系、置信度，以及供 VLM 看的关键帧索引。证据字段定义见 `architecture.md` 5.1。

### 2. 相机位姿、深度与动态掩码

| 模型 | 动态物体 | 输出 | 速度 / 显存 | 许可 | 评价 |
|---|---|---|---|---|---|
| **ViPE**（NVIDIA，v1.2.0，2026-06） | 处理：GroundingDINO → SAM → XMem 传播掩码，掩码取反约束 BA | 内参（针孔/鱼眼/360）、位姿、近度量稠密深度、动态掩码；深度先验可选 Depth Anything 3 / MoGe-2 | 3~5 FPS @640×480；300 帧约 1~2 分钟；内存有界 | Apache-2.0（UniK3D 组件 CC BY-NC-SA） | **主选**：工程最稳，一次给全 |
| **VGGT-Omega**（Meta，CVPR 2026） | 声称支持，未见动态掩码输出 | 位姿、内参、深度与置信度；1B 参数 | 纯前馈；500 帧 43 GB，300 帧@512 可在 L40S 跑 | 未核实 | **备选**：秒级前馈；需叠加 SAM 3 掩码 |
| MegaSaM（CVPR 2025） | 处理 | 内参、位姿、视频深度 | 约 0.7 FPS；依赖较旧 | Apache-2.0 | 兜底 |
| MoRe（CVPR 2026） | attention forcing 分离动静 | 位姿、深度、点云、运动掩码；基于 VGGT | 未给出 | 未标注 | RP 3.2 提到的候选；许可未明前不进主路径 |
| VGGT / VGGT-Long / VGGT4D | 否 / 否 / 免训练挖掩码 | 位姿、深度、点图 | 200 帧 40.6 GB | 非商用 + 申请制 | 不进主路径 |
| MonST3R / CUT3R / STream3R / Pi3 | 是 / 弱 / 是 / 否 | 点图、位姿、深度 | 各异 | 均非商用 | 不进主路径 |
| Depth Anything 3 | 静态；流式长视频 < 12 GB | 深度、射线 → 位姿与内参 | 轻 | Small/Base/Metric/Mono Apache-2.0 | 作 ViPE 的深度先验 |
| MapAnything（3DV 2026） | 否 | 度量点图、位姿、内参；可注入已知量 | 重 | 有 Apache-2.0 版 | 备选 |

### 3. 实例分割、跟踪与检测

| 任务 | 选用 | 备选 | 说明 |
|---|---|---|---|
| 实例分割 + 跨帧 ID | **SAM 3.1**（2026-03；848M；文本短语 / 点 / 框 / 示例提示；SAM License 允许商用，HF 需申请） | Grounded-SAM-2（SAM 2.1 + Grounding DINO 1.0，全 Apache-2.0） | 一个模型替代检测 + 分割 + 跟踪；单次前向可跟踪 16 个物体 |
| 类别候选 | VLM 对首帧列出名词短语 → 作为 SAM 3.1 提示 | OWLv2（Apache）、Florence-2（MIT）、Qwen3-VL grounding | Grounding DINO 1.5/1.6 与 DINO-X 仅 API |
| 点轨迹 | **SpatialTrackerV2**（3D 轨迹 + 动静标签；CC BY-NC） | CoTracker3（CC BY-NC）；TAPIR / BootsTAPIR（Apache，商用换这个） | 每个实例掩码内取点 |
| 记忆式 VOS | 不需要 | DEVA、Cutie（MIT） | 已被 SAM 3 覆盖 |

### 4. 证据提取算法

**逐帧 OBB**：实例掩码 × ViPE 深度 → 反投影到世界系 → 统计去离群 → 用静态平面法向对齐竖直轴 → 水平面 PCA 或最小面积矩形 → 中心、四元数、尺寸 → 沿时间做 SE(3) 平滑（掩码缺失帧插值并降低置信度）。

**运动类型分类**（对 OBB 的 SE(3) 序列做 Chasles 螺旋拟合）：

| 判据 | 类型 |
|---|---|
| 位移与旋转均低于阈值 | `static` |
| 旋转量 ≈ 0、平移沿固定方向 | `prismatic` |
| 轴方向稳定、沿轴平移 ≈ 0 | `revolute`（轴与枢轴由拟合给出） |
| 位置或角度的自相关有明显峰 | `periodic_translate` / `periodic_rotate`（周期、幅度、相位由拟合给出） |
| 轴稳定且角速度近常数、无平移 | `spin` |
| 其余 | `trajectory`（保留关键帧） |

拟合结果带残差与置信度，作为 VLM 的先验而非最终答案；VLM 可以否决（RP 第 8 节的风险应对）。2026 年的两项工作（Articulation in Prime、Track Articulate Act）用稠密 3D 轨迹刚性聚类做同类事，代码尚未放出，思路可借鉴。

**接触关系**：OBB 底面到静态深度或其他物体 OBB 顶面的距离低于阈值且持续若干帧，记为 `contacts[]`，附时间区间。

**静态结构候选**：对静态掩码内的深度做 RANSAC 平面拟合，输出地面与墙体候选（法向、偏移、覆盖比例）；地形起伏大时输出高度图候选。

**关键帧选择**：按相机运动量与物体事件（出现、消失、运动类型切换）均匀抽样 8~16 帧，记录选择理由。

### 5. 资源预算（300 帧、512 px、1×L40S）

| 步骤 | 估算 |
|---|---|
| ViPE | 1~2 分钟，显存有界 |
| SAM 3.1（约 10 个物体） | 1~2 分钟 |
| SpatialTrackerV2 | 1~3 分钟 |
| 证据提取（CPU） | < 1 分钟 |
| 合计 | 约 5~8 分钟/条；批量处理时一次作业跑多条 |

### 6. 环境与许可

- 虚拟环境用 `python/3.11.5` 模块建在 `/project`，权重预下载到 `/project`，`HF_HOME` 指向 `$SCRATCH`；作业内经代理可访问 HuggingFace 与 GitHub（见 `cluster.md`）。
- 论文用途下 SpatialTrackerV2 与 CoTracker3 的非商用许可可接受；若需商用换 TAPIR。ViPE、SAM 3.1、Depth Anything 3 Small/Base 均可商用。
- Kubric MOVi 片段带真值，用来校验 ViPE 尺度、OBB 精度与运动分类；见下面模型一节的数据集表。

### 7. 与 4D 重建的关系

MoSca、Shape of Motion、4DGaussians 等动态 Gaussian 方法输出不可编辑的高斯，不是本系统的证据来源，而是残差层与对比基线：静态背景一份 splat，每个动态实例一份 splat，由程序驱动变换，用 Spark 在 three.js 中渲染。4DGaussians 有逐时刻 PLY 导出脚本，Shape of Motion 可小改导出。

## 模型

> 系统中每个用到模型的位置、候选模型对视觉输入的支持程度、在 L40S 上的部署方式、许可与成本，以及 baseline 之后训练要用的框架与数据。定价与模型 ID 为 2026-09-18 查询并对照官方参考核对的结果。

### 1. 模型层总览

| 角色 | 输入 | 输出 | 基线选择 | 备选 |
|---|---|---|---|---|
| 程序生成（Writer） | 8~16 张关键帧 + evidence.json + DSL schema + 错误报告 | program.json / JSON Patch | `claude-opus-5`（API） | `claude-fable-5-1`（难例）；Qwen3-VL-32B-Instruct FP8（自托管） |
| 批评（Critic） | 失败子句对应的裁剪对 + 差异叠加图 + 指标 | 结构化 checklist | 同 Writer 的模型 | GLM-4.6V / InternVL3.5（UI2Code^N 与 BlenderGym 有用开源模型做 verifier 的先例） |
| 类别候选 | 首帧 | 名词短语列表 | Writer 的模型 | Molmo 2（视频指点） |
| 试玩评审 | 2 fps 抽帧序列 + state 日志 | 可玩性评分 | `claude-sonnet-5` 或 Writer 的模型 | — |
| 感知 | 视频 | 位姿、深度、掩码、轨迹 | ViPE、SAM 3.1、SpatialTrackerV2 | 见上面感知一节 |
| 对齐指标 | 渲染帧 vs 视频帧 | 相似度 | DINOv3、DreamSim | CLIP、LPIPS |
| 素材检索 / 生成 | 裁剪图 + 类别 | GLB | CLIP / SigLIP 检索；TRELLIS.2-4B | 见下面素材一节 |

### 2. API 模型

| 模型 | 模型 ID | 视觉输入 | 上下文 / 最大输出 | 价格（每百万 token，输入 / 输出） | 备注 |
|---|---|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | 图像（视频需抽帧） | 1M / 128K | $5 / $25 | **基线默认**；adaptive thinking 默认开启；结构化输出用 `output_config.format` |
| Claude Fable 5.1 | `claude-fable-5-1` | 图像 | 1M / 128K | $10 / $50 | 难例；thinking 常开；不支持强制 tool_choice |
| Claude Sonnet 5 | `claude-sonnet-5` | 图像 | 1M / 128K | $2 / $10 | 试玩评审等轻任务 |
| GPT-5.5 / 5.6 | — | 图像 | — | — | 日期与视频输入未核实 |
| Gemini 3.1 Pro | — | **原生视频与音频**，单次最多 900 图 | 1M | — | 唯一原生吃视频的 API；若关键帧抽样不够可作对照 |

Claude 系列不接受视频文件，需抽关键帧作为多图输入；这与本系统"关键帧 + 证据 token"的设计一致。

### 3. 开源 VLM（可用 vLLM 自托管）

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

### 3b. OpenRouter（MVP 实际采用）

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

### 4. 调用方式

- 脚手架 Pydantic AI：`AnthropicModel` 与 `OpenAIChatModel + VLLMProvider` 通过配置切换；工具返回 `BinaryContent` 把渲染截图回传给模型；输出用 Pydantic 类型约束（程序 JSON、Patch、批评 JSON）。
- Claude 调用：使用官方 SDK；`claude-opus-5` 默认 adaptive thinking；长输出用流式；结构化输出用 `output_config.format` 而不是 prefill。
- 计算节点经 squid 代理可达 `api.anthropic.com` 与 `api.openai.com`（实测），agent 循环可整个放在作业内。
- 自托管：vLLM 起 Qwen3-VL-32B-Instruct-FP8 于 2×L40S，OpenAI 兼容端点；Vulcan 自带的 Aleph 推理服务也可作备选端点。

### 5. 成本估算（粗估，待阶段三实测）

单条片段、三阶段生成、两轮反馈、每轮 2 个候选：

| 项 | 估算 |
|---|---|
| 每次调用输入 | 12 帧 × 约 1.5k token + evidence 5k + schema 与 few-shot 8k ≈ 30k token |
| 调用次数 | 3 阶段 × (1 + 2 轮 × 2 候选) ≈ 15 次 Writer + 4 次 Critic |
| 输入合计 | 约 0.6M token → `claude-opus-5` 约 $3 |
| 输出合计 | 约 60k token → 约 $1.5 |
| 单条片段 | 约 $4~5；十条片段一轮实验约 $50 |

Prompt caching 可以显著降低：schema、few-shot 与 evidence 放在前缀并打缓存断点。自托管 Qwen3-VL 只有 GPU 时费。

### 6. baseline 之后：训练栈与数据

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

## 视觉反馈与编排

> 渲染比对如何变成 VLM 能执行的修订指令，循环怎么组织，用什么脚手架，以及为什么不做角色扮演式多智能体。证据来自 2026-09-18 的调研（`../survey/04`）。

### 1. 三条原则

1. **结构化胜过自由评论**。把参考截图与自身截图一起丢回给模型"自我修订"收益很小（Design2Code：GPT-4V 提升 3 个点，Claude 3 Opus 无变化）。有效的做法是可定位的差异清单：SEIG 的 checklist、Scenix 的逐物体 pass/fail 子句、VF-Coder 的元素级不匹配。
2. **数值先行，VLM 收尾**。先用免 LLM 的指标筛出失败对象，VLM 只对失败区域做定位与建议。M³-Verse 显示多模态模型在成对观测的状态变化检测上普遍薄弱，所以要给它叠加差异图、分块编号与 IoU 数值，而不是让它自由比较两张图。
3. **有界轮数、候选选优、允许回退**。OpenGame 在第三轮出现平台期，UI2Code^N 五轮提升 8 个点；BlenderAlchemy 与 BlenderGym 用成对比较选优并允许回退，低预算时多生成、高预算时多验证。

### 2. 渲染输出

与视频对齐的 K 个关键帧（K 取 8~16），每帧三个 pass：rgb、depth、id。实现见下面运行时一节 第 4 节。

### 3. 数值层

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

### 4. VLM 批评

- 输入：仅失败子句对应的对象；每个对象一组裁剪对（视频帧与渲染帧同一区域）、一张差异叠加图、指标数值、当前程序中该对象的节点。
- 输出限定为结构化 JSON：`{object_id, issue ∈ {missing, extra, pose, scale, timing, class}, evidence, suggested_edit}`，其中 `suggested_edit` 是 JSON Patch 片段。
- 禁止自由评语；提示词要求引用具体数值与区域。

### 5. 候选搜索与选优

每轮 Writer 生成 2~4 个候选补丁 → 全部编译渲染 → 数值分排序 → 前两名交给 VLM 成对比较 → 选优；若最优不如上一轮则回退并换 hint。默认两轮，上限三轮。

### 6. 编排

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

### 7. 为什么不用多智能体

| 证据 | 结论 |
|---|---|
| Cemri et al., Why Do Multi-Agent LLM Systems Fail（NeurIPS 2025） | 七个框架失败率 41%~87%；四分之一失败源于验证缺失或错误；给 ChatDev 加高层目标验证提升 15.6% |
| How Generation Architecture Shapes Code Complexity（2026.06） | 六种架构通过率无显著差异；角色拆分让代码复杂度膨胀 50%~130%；有价值的是执行落地的 debugger |
| AgentCoder | 少角色 + 独立测试生成器优于多角色，token 少一半以上 |
| Anthropic 多智能体研究系统 | 并行广度搜索受益，但 token 约 15 倍；编码任务少有真正可并行的子任务 |
| OpenGame（2026.04） | 单 agent 六阶段流水线 + 模板库 + 有界修复，第三轮平台期 |
| GameCraft-Bench / GameDevBench（2026） | 查看渲染截图的 agent 更常成功；视觉反馈稳定提分（41% → 52%） |
| AVR-Agent（2025.08） | 迭代显著优于一次生成，但更多模态的反馈没有再提升；反馈可操作性是瓶颈 |

### 8. 相关工作的反馈机制

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

### 9. 试玩层的反馈

见下面运行时一节：轨迹回放 + 状态判定 + VLM 评审；不做实时游玩 agent。

## 运行时与游玩层

> 编译目标为什么是 three.js，运行时内核如何组织，玩法模板与槽位怎么定义，无头渲染与确定性步进怎么做，试玩验收怎么做。版本号为 2026-09-18 查询结果。

### 1. 运行时选型

| 引擎 | 版本 | 生态（stars / npm 周下载） | 无头渲染 | 物理 | 结论 |
|---|---|---|---|---|---|
| **three.js** | r186 | 115.6k / 14.8M | headless Chromium 直接可跑 | 外接 Rapier / Jolt / cannon | **选用**：LLM 训练数据最丰富；纯库无编辑器；深度与 ID pass 可在 JS 层完全控制 |
| Babylon.js | 9.27 | 26.1k / 314k | `NullEngine` 只跑逻辑不出图，渲染仍需 Chromium | 内建 Havok | 唯一值得考虑的替代，生态差一个量级 |
| PlayCanvas | 2.22 | 16.8k / 62k | 同 three | 内建 ammo | 场景 JSON 与云端编辑器强绑定 |
| Godot Web | 4.7 | 117k / – | Web 导出需 SharedArrayBuffer；`--headless` 剥离渲染 | 内建 | 生成目标是 GDScript + .tscn；无法从外部注入深度/ID pass；适合最终导出而非中间目标 |
| Phaser | 4.2 | 40.3k / 308k | 同 three | 2D | 纯 2D，不匹配 |

旁证：WorldCoder-Bench（2026）以 three.js 为浏览器 3D 世界生成的唯一基底；Rosebud 的 3D 内核 Roseblox 与 Bitmagic 都选 three.js；npm 下载量 three 与 babylon 之比约 47 : 1。

不用 react-three-fiber：它的 JSX 像声明式 DSL，但引入 React 运行时与 hooks 语义，无头调试与帧级确定性更绕，配套库更新也落后 Rapier 主库。

### 2. three.js 游戏栈

| 类别 | 选用 | 版本 | 备选 | 说明 |
|---|---|---|---|---|
| 物理 | `@dimforge/rapier3d-compat` | 0.20.0 | jolt-physics 1.1（three 官方 addon，LLM 熟悉度低）；cannon-es（2022 停更） | WASM；`KinematicCharacterController` 内建自动上台阶、贴地、坡度限制；跨机确定性用 `-deterministic-compat` |
| 角色控制 | Rapier KCC | — | `three-mesh-bvh` shapecast 示例 | ecctrl 仅 R3F |
| 导航 | `@recast-navigation/three` | 0.43.1 | three-pathfinding（只做 A*） | 运行时从几何生成 navmesh 与 crowd；可行走面推断直接复用 |
| 游戏 AI | yuka | 0.7.8 | 自写 | steering、FSM、goal-driven；引擎无关 |
| ECS | miniplex | 2.0.0 | bitecs 0.4、koota 0.6 | 简单、LLM 易读；Roseblox 采用 |
| 后处理 | postprocessing | 6.39.5 | — | 与 r186 兼容；可选 |
| Gaussian 残差 | `@sparkjsdev/spark` | 2.2.0 | GaussianSplats3D（停滞） | three.js 原生对象、MIT、流式 LOD、多 splat 对象 |
| 相机 | 自写 spring-arm + lerp；OrbitControls 调试用 | — | camera-controls | 第三人称跟随无主流独立库 |
| 动画 | `AnimationMixer` + `SkeletonUtils.retargetClip` | 内建 | — | KayKit 角色自带动画；Mixamo 不可再分发 |
| 输入 | 自写 keydown / keyup 状态表 | — | — | 无头时由 `__game.input` 注入 |
| 随机 | seedrandom | 3.0.5 | alea | 确定性 |

### 3. 内核设计

**固定代码，生成侧不碰。** 沿用 Roseblox 的模式：Resource → Setup → Runtime systems，systems 按优先级区间执行。

| 区间 | systems |
|---|---|
| 10–20 | input（读取按键集合） |
| 20–30 | motions（对每个动态物体求 `pose(t)` 并写入运动学刚体） |
| 30–40 | movement（玩家 KCC、敌人 steering） |
| 40–45 | physics（Rapier `world.step()`，固定步长） |
| 45–50 | events（接触、体积触发、收集计数、胜负判定） |
| 50–65 | animation（AnimationMixer 更新） |
| 70–75 | camera（回放模式跟随关键帧；游戏模式 spring-arm） |
| 90 | render（rgb；按需 depth / id） |

**控制面 `window.__game`**：`step(dt)`、`render(pass)`、`state()`、`input(keys)`、`reset(seed)`、`mode(m)`，定义见 `architecture.md` 5.3。

**状态 schema 固化**：`state()` 返回的字段与类型写死在内核里，生成侧不能增删。WorldCoder-Bench 的分析指出这类系统的主要失败是"状态 schema 漂移与交互链断裂"，而非缺场景元素。

**确定性**：物理固定步长；随机数用 seedrandom；`step` 完全绕过 requestAnimationFrame；需要跨机器逐位一致时换 `rapier3d-deterministic-compat`。

**免打包**：`index.html` 用 importmap 指向 `vendor/`，产物不需要 npm；VLM 可以直接读产物排错。

### 4. 无头渲染

**启动参数**（Playwright 1.63 + Chromium 1243，已在 Vulcan 实测，见 `../RUNNING.md`）：

- 通用：`--headless=new --no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage --disable-gpu-sandbox --disable-background-timer-throttling --disable-renderer-backgrounding`
- CPU 软渲染：`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist`
- GPU 硬件加速：`--use-gl=angle --use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan --ignore-gpu-blocklist`
- 不要用 `--use-gl=egl`（回退 SwiftShader 且丢上下文），不要自己传 `--use-gl=swiftshader-webgl`。

**三个 pass**：

| pass | 实现 | 读回 |
|---|---|---|
| rgb | 正常渲染到 `WebGLRenderTarget` | `readRenderTargetPixelsAsync` 或 `page.screenshot`（需 `preserveDrawingBuffer: true`） |
| depth | `scene.overrideMaterial = MeshDepthMaterial({ depthPacking: RGBADepthPacking })`，或挂 `DepthTexture` | 同上；解包为线性深度 |
| id | 渲染前把每个物体的材质临时换成 `MeshBasicMaterial({ color: idColor })`，24 位 ID 编码到 RGB，渲染后恢复（官方 GPU picking 示例做法） | 同上；ID 图直接与 SAM 掩码算 IoU |

**帧步进**：由 `page.evaluate(() => __game.step(dt))` 驱动，关键帧时刻调用 `render`。备选是 Playwright `page.clock`（`goto` 前 `install()`，`pauseAt` + `runFor(16.67)`）。

**Node 路线不选**：headless-gl 主打 WebGL1 而 three.js 已要求 WebGL2；WebGPU/Dawn 的 Node 方案仍属实验性。

### 5. 玩法模板与槽位

每个模板是内核内一组固定 system，只暴露槽位；DSL 的 `binding` 段填槽。

| 模板 | 槽位 | 核心 system | 状态 |
|---|---|---|---|
| `platformer_3p` 第三人称平台跳跃 | `player_spawn`、`goal_volume`、`walkable`、`hazards[]`、`collectibles[]` | KCC 移动与跳跃、动态平台承载、收集计数、危险物重生、到达目标判胜 | 第一阶段实现 |
| `topdown_survival` 俯视生存 | `player_spawn`、`arena_bounds`、`enemy_spawns[]`、`enemy_paths[]`、`pickups[]` | 俯视相机、yuka 巡逻与追击、生命与计时 | 可选 |
| `racing` 赛车 | `track_spline`、`start_grid[]`、`checkpoints[]`、`obstacles[]` | 载具控制、检查点计时 | 草案 |
| `tower_defense` 塔防 | `enemy_paths[]`、`tower_slots[]`、`base_volume`、`wave_schedule` | 路径跟随、放置、波次 | 草案 |
| `shooter_simple` 简单射击 | `player_spawn`、`enemy_spawns[]`、`cover[]`、`ammo_pickups[]` | 第一或第三人称射击、命中判定 | 草案 |
| `puzzle_escape` 解谜 | `player_spawn`、`switches[]`、`doors[]`、`keys[]`、`exit_volume` | 开关与门的触发链、钥匙与锁 | 草案；直接复用 DSL 的 `trigger_on_enter` |

**绑定规则（平台跳跃）**：recast 对静态结构与静止物体生成 navmesh；出生点取 navmesh 上距视频首帧相机最近的可站立点；目标体取 navmesh 上离出生点最远的连通点，或提示中指定的物体；类别为金币、宝石、钥匙的进收集物；类别为岩浆、尖刺、水的进危险物；运动物体保持其 motion 作为动态平台。视频中没有玩家时由模板生成角色（KayKit CC0 角色与动画）。

### 6. 试玩验收

不做实时游玩 agent：VideoGameBench 的经验是推理延迟主导失败；BALROG 发现给图像后多数模型反而变差。采用 GameCraft-Bench 的做法：

1. 由 navmesh 路径生成一条出生点到目标的按键序列（含跳跃时机），写成 `trajectory.json`。
2. 无头运行，逐帧 `__game.input` 注入，记录每帧 `state()`，2 fps 抽帧存图。
3. 规则判定：是否到达目标、是否卡死（位置长时间不变）、帧率是否达标、收集物计数是否变化。
4. VLM 评审抽帧序列：可玩性、视觉一致性、明显缺陷，输出结构化评分；作辅助而非主判据。

### 7. 参考仓库

| 仓库 | 用途 |
|---|---|
| rosebudai/roseblox（MIT，2026-09 活跃） | 内核模式：three + Rapier + miniplex，systems 优先级区间，buildless |
| isaac-mason/sketches | three + Rapier + recast 的现代栈样例，质量高 |
| swift502/Sketchbook（已归档） | 第三人称角色与载具控制参考 |
| pmndrs/racing-game | 赛车模板参考（R3F + cannon，需现代化） |
| Casmo/tower-defense | 塔防模板参考 |
| dylanebert/VibeGame | 面向 LLM 的声明式标签 + ECS 思路 |
| heagandev/threejs-agent-starter | 面向 agent 的 three.js 起步模板 |

### 8. 为什么不用现成的声明式场景格式

three.js 自带的 JSON Object/Scene format 只描述静态层次且几何内联冗长；glTF 是资产交换格式，可用 vendor extension 挂元数据（Needle Engine 的做法），但不适合承载运动与绑定；A-Frame 的 HTML 实体组件语法对 LLM 友好但绑定 DOM 与 WebXR；VibeGame 最接近但近一年未更新。结论是自研 JSON DSL，资产用 GLB 承载并以 URI 引用。

## 素材

> 视频中检测到的物体如何变成 three.js 里的几何：三层回退、CC0 素材库、检索配方、生成模型、许可。数据来自 2026-09-18 调研（`../survey/03`）。

### 1. 三层回退

| 层 | 做法 | 何时用 |
|---|---|---|
| 参数化 primitive | 门、平台、金币、树、矿车等各写一个参数化生成器，接受 extent 与颜色 | 永远可用；第一阶段到第五阶段的默认 |
| CC0 GLB 库 + 检索 | 1,000~2,000 个低多边形 GLB，CLIP 检索 + VLM 重排 | 第六阶段起的默认 |
| 单图生成 | TRELLIS.2-4B（MIT）或 SAM 3D Objects 对 hero 物体生成 | 可选；每条片段 1~3 个，失败回退 |

不建 Objaverse 级大库：CC0 仅 3.5K，质量噪声大，检索需 Objaverse++ 级过滤，与 baseline 目标不匹配。

### 2. CC0 素材库来源

| 库 | 规模 | 格式 | 许可 | 获取 | 品类 |
|---|---|---|---|---|---|
| **Kenney** | 4,812 个 GLB / 49 套 kit（镜像 github.com/shorepine/kenney，附 kits.tsv） | GLB | CC0 | git clone | Nature 329、Platformer 153（金币、平台）、Car 50、Train 103、Racing 112、City、Castle 76、Pirate 72、Food 200、Furniture 140、Space 153、Characters、Weapon |
| **KayKit** | Dungeon 200+、Adventurers 4 角色（75 动画）、City Builder | FBX / glTF | CC0 1.0 | GitHub org `KayKit-Game-Assets` | 地牢、角色、城市 |
| **Quaternius** | 数千模型，60~70% 免费 | FBX / OBJ / glTF | CC0 | Google Drive 逐包；多数已镜像到 Poly Pizza | 角色（含动画）、载具、自然、建筑、道具 |
| Poly Pizza | 10,700+ | OBJ / FBX / glTF | 混合，API 可按 CC0 过滤 | API v1.1，免费 key；**计算节点代理封禁，只能在登录节点抓** | 全品类低多边形 |
| Poly Haven | 约 520 模型 | glTF | CC0 | 公共 API | 写实 PBR 扫描，非低多边形 |
| Sketchfab | 700K+ CC | glTF | 混合 | 下载需终端用户 OAuth，署名随行 | 不适合打包 |
| Mixamo | 角色与动画 | FBX | 免版税但禁止独立再分发 | 无 API | 不打包；用 KayKit 角色替代 |

Kenney + KayKit + Quaternius 全部 CC0、全部有 glTF，合计超过 5,000 个 GLB，品类正好覆盖平台跳跃与低多边形世界。

### 3. 库构建流程

1. 登录节点抓取（网络操作，轻量）：Kenney 镜像按 kits.tsv 选 Nature、Platformer、Car、Train、City、Castle、Pirate、Food、Furniture、Characters 等约 1,500 个；KayKit 与 Quaternius 免费包；Poly Pizza 按 `license=CC0` 补漏。
2. 归一化：居中、Y-up、单位包围盒、记录原始尺寸、meshopt 或 Draco 压缩；记录来源、kit、许可。
3. 计算节点批量渲染每个资产 4~6 个视图的缩略图（three.js 无头渲染即可，复用 harness）。
4. 嵌入：CLIP 或 SigLIP 图像嵌入 + 用 kit 名、文件名、VLM 描述做文本嵌入，入 FAISS。
5. 存放：GLB 与嵌入放 `/project`（约 0.5~2 GB GLB + 0.2~0.4 GB 缩略图 + 十几 MB 嵌入）。

### 4. 检索配方

Holodeck（CVPR 2024）的配方去掉几何项再加上图像查询：查询 = 视频中该物体的最佳裁剪帧（图像到图像）+ 类别文本（文本到图像与文本）→ 加权融合 → top-k → VLM 重排（尺寸先验：候选原始尺寸与 evidence 中 extent 的比值；是否 rigged；风格一致性）→ 按 extent 缩放放置。命中分低于阈值时回退 primitive。

对不超过两千个资产的小库，渲染缩略图 + CLIP 已足够；OpenShape / Uni3D 等 3D 原生嵌入是可选增强。

### 5. 生成模型

| 模型 | 发布 | 输入 | 输出 | 显存 / 时延 | 许可 | 评价 |
|---|---|---|---|---|---|---|
| **TRELLIS.2-4B**（Microsoft） | 2025-12 | 单图 | mesh + PBR → GLB | ≥ 24 GB；H100 上 512³ 约 3 秒、1024³ 约 17 秒；L40S 估 2~3 倍 | MIT | **首选**：自带 decimate、remesh、UV unwrap，最接近 game-ready |
| **SAM 3D Objects**（Meta） | 2025-11 | 单图 + 掩码 | 形状、纹理、位姿与布局；3DGS 与 mesh | 未核实 | SAM License（HF 需登记） | 擅长遮挡与杂乱场景，最贴近"视频帧裁剪"输入；额外给位姿 |
| Hunyuan3D 2.1 | 2025-06 | 图像 | mesh + PBR | 约 29 GB | Tencent Community License：不适用于 EU / UK / 韩国，禁止用输出训练其他模型 | 质量口碑好但许可受限 |
| Hunyuan3D-Omni | 2025-09 | 图像 + 包围盒 / 骨架等控制 | mesh + PBR | 10 GB | 同族 | 包围盒控制对"按视频尺寸生成"有用 |
| Step1X-3D | 2025-05 | 单图 | 水密 mesh + 纹理 | 27~29 GB；约 150 秒 | Apache-2.0 | 训练与数据管线全开源 |
| TripoSG / Direct3D-S2 / PartCrafter | 2025 | 图像 | mesh（无纹理 / 高分辨 / 多部件） | 8~24 GB | MIT | 几何为主 |
| Hunyuan3D 2.5 / 3.0 | — | — | — | — | 无开源仓库，仅 API | — |

没有开源模型原生输出美术风格低多边形加干净 UV；统一后处理：decimate → xatlas UV → 纹理烘焙 → glTF-Transform 压缩。

商业服务一行速览：Meshy（Pro $20/月，API 在 Pro 以上，Meshy-6 有 low-poly 模式）、Tripo（有公开 API）、Rodin/Hyper3D（Business $120/月含 API）、Sloyd（API 约 $0.13~0.33/模型，参数化 game-ready）。

### 6. 关节与部件

门、抽屉、轮子这类关节物体，baseline 用参数化模板最稳（门 = 框 + 板 + 绕 Y 轴 pivot；矿车 = 箱体 + 四个圆柱轮）。若要从生成 mesh 自动拆关节：Hunyuan3D-Part（P3-SAM）与 Particulate（2025-12，前馈秒级，代码未核实）是最现实的选项；Articulate-Anything（ICLR 2025）能从视频生成 URDF，依赖 PartNet-Mobility 检索。2026 年的 PAct、ArtLLM、URDF-Anything+、MonoArt 均为单图或 mesh 到部件与运动参数，代码状态未核实。

### 7. 视频到规范 mesh

截至 2026-09 没有开箱即用的工具：Shape of Motion、MoSca 输出动态高斯而非 mesh；Mesh4D（2026-01）最贴题但代码未核实。实用做法是选帧（面积最大、遮挡最少）→ SAM 3 掩码 → SAM 3D Objects 或 TRELLIS.2 单图生成。

### 8. 许可规则（面向论文发布）

- 只打包 CC0（Kenney、KayKit、Quaternius、Poly Haven、Smithsonian），可随代码整体分发。
- 引入 CC-BY 必须附 attribution 清单；排除 NC / ND / SA。
- 不打包 Mixamo、Sketchfab 下载物、ShapeNet、Toys4K、3D-FUTURE，改为提供获取脚本。
- 若生成资产将用于训练数据，选 MIT / Apache 系生成模型（TRELLIS.2、TripoSG、Direct3D-S2、PartCrafter、Step1X-3D），避开 Hunyuan3D 系的地域与训练限制。
