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

## 从场景到玩法：角色是怎么定下来的

视频拍下来的是一个世界，不是一个游戏。视频里没有玩家、没有目标、没有输赢、没有交互规则。

所以这套系统实际在做两件性质不同的事。一件是**还原**，把看到的世界变成能跑的场景，这件事有对错，可以和视频比对，指标见 `gwm/feedback/gt_metrics.py`。另一件是**发明**，在这个世界上架一套玩法，这件事没有对错，只有合不合适。这一节讲的是第二件。

### 为什么不能靠类别名

最早的做法是拿物体的 class 名去匹配两份写死的词表，收集物匹配 coin、gem、key、star，危险物匹配 lava、spike、water、fire。

这套只在我们自己用游戏术语手写的合成场景上管用。真实视频里感知给出的类别是开放词表，叫 toy_train、conveyor belt、metal arm、cardboard box，一个都匹配不上，所以两段真实视频的收集物和危险物列表全是空的，玩起来就是在平地上走到角落，中间什么都不发生。

把词表加长解决不了问题，真实世界里的物体名字是无穷的。

### 改成看属性

现在看的是物体本身的几个量，它们和物体叫什么名字无关：

| 量 | 怎么算 | 为什么管用 |
|---|---|---|
| 顶面面积 | 包围盒的长 × 宽 | 决定站不站得住人 |
| 最大边长 | 包围盒最长的那条边 | 决定捡不捡得起来 |
| 块状度 | 第二长边 ÷ 最长边 | 分开块状物和条状结构件 |
| 高度 | 位置的 y 分量 | 决定够不够得着 |
| 运动类型 | `motion.type` 是不是 static | 分开会动的和不会动的 |

阈值一律以**玩家**为基准，不以场景尺度为基准。绑定会把最大静态足迹缩放到 24 米的游戏尺度，而玩家胶囊是固定大小不跟着缩。同一扇门在大场景里占 5%、小场景里占 30%，但它能不能被捡起来跟场景多大没关系。玩家胶囊半径 0.35、半高 0.55，所以直径约 0.7 米、总高约 1.8 米，阈值都从这两个数推出来。

### 规则

判断顺序是有讲究的，站得住人的优先当平台：

| 顺序 | 角色 | 条件 |
|---|---|---|
| 1 | 危险物 | 类别或材质里出现危险物关键词 |
| 2 | 会移动的平台 | 顶面 ≥ 0.45 平米，而且在动 |
| 3 | 平台 | 顶面 ≥ 0.45 平米，静止 |
| 4 | 收集物 | 站不上去、最大边长 ≤ 1.4 米、块状度 ≥ 0.3、高度够得着 |
| 5 | 移动障碍 | 在动、站不上去、高度够得着 |
| 6 | 装饰 | 其余 |

顺序先判平台是实测踩出来的：楼梯一级级尺寸都不大，先判收集物的话整段楼梯会变成一串金币。

0.45 平米这个数贴着玩家足迹（0.7 × 0.7 ≈ 0.49），角色控制器有自动上台阶和贴地，所以略小于足迹也站得住。设成 1.0 会把一级台阶判成站不住。

块状度这个判据是用来分开硬币和门框的。硬币 0.5 × 0.5 × 0.06，块状度 1.0；门框 1.79 × 0.22 × 0.08，块状度 0.12。用长宽比做不到这点，它把扁片和长条混为一谈——硬币长宽比 8.3，门框 22，中间没有干净的分界。

### 一个诚实的边界

**危险物推不出来。** 熔岩、水、火这些本质上是语义判断，光看尺寸形状没法知道一个东西危不危险。所以这一项仍然保留一份可配置的关键词表，并且在输出里把它标成语义判断、置信度低。要做对需要接一个语义模型，那是后话。

每条判断都带一句理由字符串，产物里能直接看到为什么某个物体被判成某个角色，方便人工审查。

几何尺寸按形状读取：球体使用直径，圆柱使用直径与沿其轴向的高度，锥体使用直径和高度；不能把圆柱当球体而丢掉高度。缺失或非有限尺寸不默认成小收集物，而保守标为装饰并说明原因。这仍是局部尺寸启发式，不证明旋转后的可站顶面或支撑关系。

已有 `binding.slots.collectibles`（包括空列表）优先于自动推断。绑定只为最终有效收集槽位添加缺失的接触消失事件，不给被排除的对象新增事件；输入中已经手写的事件保持不变。因此空列表阻止的是自动添加，并不删除已有事件。绝对高度可达性、实例级角色和完整理由持久化仍是待完善边界。

### 实际效果

| 场景 | 推断出来的角色 |
|---|---|
| 手写合成场景 | 升降平台 → 会移动的平台；矿车 → 会移动的平台；门 → 移动障碍；金币 → 收集物；熔岩坑 → 危险物 |
| 玩具火车（真实视频） | 两段火车 → 会移动的平台 |
| 传送带（真实视频） | 皮带、金属臂、纸箱 → 平台；没有收集物 |

手写场景的结果和原来按词表匹配出来的一致，但现在是从属性推的。

传送带那段推不出收集物，这是诚实的结果而不是失败：那段视频里确实没有玩家能捡起来的东西。玩具火车那段拿到了之前完全没有的玩法元素——火车可以站上去被载着走。

### 玩法模板的选择（还没实现）

现在只有第三人称平台跳跃一个模板，所以没有选择可言。等模板多起来之后，选哪个可以由场景内容决定，这比固定一个模板更贴近「从视频恢复游戏」这个命题：

| 场景里有什么 | 适合的玩法 |
|---|---|
| 高度差 + 会移动的平台 | 平台跳跃 |
| 多个小的离散物体 | 收集 |
| 周期性运动的障碍 | 躲避 |
| 可推动物 + 目标位置 | 推箱 |
| 明确的路径（比如玩具火车的轨道） | 竞速 |

这一块先只写设计不实现。为一个模板写选择器没有意义，等真有几个模板了再说。

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

## 证据与验证门禁

目前接通的是 Evidence → 候选 Program → 校验与保守修复 → 编译 → 渲染验证 → 试玩。
主动感知请求、定向 observation 和 Evidence Update 保留在独立功能分支，待 Updated Evidence
能喂回重新合成后再单独集成；本管线不输出尚无后续消费阶段的请求文件。

### Evidence 契约与质量评估

正式契约位于 `gwm/perception/schema/evidence.schema.json`，统一入口是
`validate_evidence()`。旧格式数据经 `adapt_legacy_evidence()` 适配，缺失的观测、属性和数值置信度
保留为 `unknown` 并产生警告，不通过默认值伪造证据。帧顺序、引用、track 身份、范围及必需字段
使用结构化诊断报告。二维可见状态与数值可见比例分别记录，不相互替代。

`assess_evidence_quality()` 分别记录逐对象覆盖、bbox/mask/depth/可见性实测覆盖、track 缺口、
相机覆盖、属性 unknown、身份歧义和运动支持。结构损坏与关键引用错误为 `block`，
信息不足通常为 `warn`；质量决策与诊断会传递给生成器，并保存为 `evidence_quality.json`。
通过契约检查不等于具备充分的三维重建证据。

文件与接口使用语义化名称，格式版本仍由 Schema 内部的 `schema_version` 约束为 `2.0`；
此次命名整理不改变数据格式、旧数据适配行为或诊断代码。适配器的公共名称统一为
`adapt_legacy_evidence()`，模块导入和包导出均已更新，不保留旧编号名称的别名。
验证入口的适配开关同样命名为 `adapt_legacy`，默认仍为 true；显式关键字调用需使用
`validate_evidence(data, adapt_legacy=False)`，位置参数调用不受影响。

### 多帧关联与运动接口

`DetectionProvider.detect()` 提供检测记录，`associate_detections()` 独立关联。单关键帧模式保留
首帧／中帧回退；多关键帧模式允许晚出现与遮挡恢复，身份不可靠时记录候选，不强制合并。
主分支的 `NullSegmentationBackend` 回退以及 `gwm.taxonomy` 的公共类别和 ID 规则继续使用。

运动统一入口为 `estimate_motion(times, positions, quaternions, config, ...)`，输入分别为
`[N]` 时间、`[N,3]` 位置和 `[N,4]` 的 xyzw 四元数。相机空间输入需提供相机位姿，转换到
Y 向上的世界坐标后拟合静态、平移、绕轴和周期假设；证据不足返回未知或竞争候选。
返回值包含 `motion_guess` 和 `hypothesis`，不再仅返回旧运动字典。

API 变更：移除只为旧 yaw 调用提供转发的 `evidence.classify_motion()` 与
`motion.classify_motion_legacy()`。生产路径和主分支的数值测试均直接使用 `estimate_motion()`。
调用方若只有绕 Y 轴的角度，应自行转换；单轴批量角度使用显式列向量，兼容 SciPy 1.13.1 和 1.17.0：

```python
from scipy.spatial.transform import Rotation

quaternions = Rotation.from_euler("y", yaw.reshape(-1, 1)).as_quat()
result = estimate_motion(times, positions, quaternions, config, coordinate_space="world")
guess = result["motion_guess"]
```

标量角度转换不需要改动。此 API 简化不影响旧格式 Evidence 数据适配。

旋转拟合对绕固定轴的有符号角度执行解缠绕，避免总转角超过 180° 后失效。
这要求采样足够密集：相邻帧无法唯一确定超过半圈的旋转，遗漏的整圈也无法从四元数恢复。
相邻观测恰为半圈时返回 unknown，不猜旋转方向。固定轴模型仍不等价于任意三维旋转。

朝向不可靠时保留位置支持的静止候选，但旋转残差与朝向状态标记 unknown。
`perception.motion.unreliable_orientation_static_confidence_cap` 的内置默认值为 0.50，
可按需覆盖；这是拟合评分上限，不是实测观测置信度。默认低于主假设门槛 0.55，
因此输出 unknown 并保留位置支持候选，而非丢弃信息或宣称物体没有转动。
显式放宽配置后可能输出低置信度 static，朝向未知标记仍保留；位置平移证据不因此清零。

### Candidate Preflight 与保守 Repair

`validate_candidate(program)` 检查 Schema、引用和几何，包括非有限变换、四元数、穿透及支撑接触。
`repair_candidate(program, evidence, diagnostics, config)` 返回独立副本、repair report 和 unresolved。
传入诊断必须与当前 Program 的重新验证一致。

默认 `candidate_repair.mode: off`。`report` 在副本上试算，将建议写入 `proposed_repairs`，返回
未修改的 Program；`apply` 最多执行 `max_passes` 轮 validate → repair → validate。Writer 仅在
模式不为 `off` 时调用。每次修改记录规则、字段路径、前后值、Evidence 引用、置信度与原因。

规则只允许接近单位的有限四元数归一化，以及明确支撑关系的静态物体沿世界 Y 轴小幅贴面。
贴面要求质量为 `proceed`、同一来源 clip、米制 Y 向上坐标、可靠且一致的几何和接触观测，以及
唯一且近水平的静态支撑面。相对尺度、动态物体、合法悬浮物体、非有限坐标、零四元数、
歧义支撑、较大位移或语义冲突保持 unresolved。修复有界且幂等。

```python
from gwm.compiler.validate import validate_candidate
from gwm.compiler.repair import repair_candidate

initial = validate_candidate(program)
fixed, report, unresolved = repair_candidate(program, evidence, initial, config)
```

### 渲染验证与配置分组

最终绑定后的 Program 重新渲染 RGB/depth/ID，通过 `validate_render_result()` 检查图像存在、
解码、尺寸与空白、ID 注册、对象可见性、动态变换、时间戳及浏览器执行错误。
结果保存为 `render_validation.json`，不使用单一总分。默认 `report` 记录问题继续原流程；
`enforce` 遇到阻断诊断时停止在试玩前，不自动修改 Program。

`configs/default.yaml` 保留影响证据判定及安全边界的阈值，按功能分组：

| 配置组 | 用途 |
|---|---|
| `perception.detection_keyframes` / `association` | 检测采样与多帧关联 |
| `perception.motion` | 最小证据、假设拟合残差与置信度 |
| `evidence_quality` | 实测覆盖、缺口与歧义门禁 |
| `candidate_repair` | 模式、轮数及安全修复容差 |
| `render_validation` | 图像、可见性、运动与执行验证 |

主动感知和定向观测专属配置随功能拆出；旧 yaw 分类器移除后不再使用的自相关、相机漂移
和运动一致性阈值一并移除。仍参与判定的阈值保持配置化，重复轴稳定性配置保留原实际生效值 0.9。
渲染评分默认的 `feedback.passes: [rgb, id]` 与
最终验证所需的 RGB/depth/ID 三通道是两个不同用途，保持 PR #1 的评分修复。

CPU 单元测试不需要模型或权重。依赖矩阵使用两个独立环境，固定相同的 NumPy 1.26.4，分别
安装 SciPy 1.13.1 与 1.17.0 后运行 `python -m pytest -q tests`。通用依赖不设置旧版本上限。

### 直接代码宿主的几何与位姿契约

`describe(THREE)` 中的 `object3D` 必须是独立根节点。根的 position/quaternion 是初始世界位姿；
`pose(t)` 返回绝对世界位姿，替换而非叠加到初始根变换。根 scale 和子节点变换属于局部几何。
宿主克隆输入，将根刚体位姿移到 registry.group，只应用一次；静态根位姿也进入导出状态。
生成物理碰撞体前，以单位根位姿计算局部包围盒，使用物理内核支持的完整 `extent` 与局部 `offset`。
因此根旋转不会被重复编码到包围盒，再被刚体旋转一次。

包围盒是局部轴对齐盒近似，不是逐三角形或凹面碰撞体；零厚度轴沿用 0.1 米碰撞厚度。
带未声明父变换、重复 ID、空/非有限几何、无效位姿以及 static 搭配 pose 明确拒绝，不静默跳过。
根和所有子节点必须启用 `matrixAutoUpdate` 与 `matrixWorldAutoUpdate`；构建前遍历检查并拒绝
手动矩阵模式。否则修改 position/quaternion 不会可靠更新实际矩阵，可能重新产生重复位移或
使用过期世界坐标。不自动改写这些开关，也不猜测手动矩阵的语义。
已有生成代码需要复核这一约定；若初始朝向需要保留，必须将其包含在 pose 返回的 quat 中。
手写示例同步迁移，并有回归检查；旧模型产物不自动批量改写。

`tests/test_code_scene_host.py` 使用已安装的 Node/Three.js/Rapier 和手写合成描述验证真实 CPU 物理。
通过 `GWM_NODE_BINARY`（或 PATH）与 `GWM_NODE_MODULES` 配置依赖，缺失时显式 skip，不自动下载。
测试临时编译包只替换模块解析路径，不改物理算法。覆盖大地面玩家站立、绝对位姿、旋转/缩放/偏移
和非法输入；这不是浏览器图像验证，也不是模型重建效果实验。

直接入口先用统一 Evidence 验证器校验输入，保留旧生产者实有的帧注册表并重新校验，不制造缺失帧。
从 `meta` 读取 clip/duration，并按 keyframes 声明顺序、图片引用和时间发送全部帧，不按目录排序猜测
或默认截断至8张。引用必须位于 `--keyframes` 目录；兼容相对图片目录和旧生产者相对运行目录的
路径，但两种解析指向不同合法文件时拒绝，要求明确引用。迁移路径需要显式 `--allow-relocated-keyframes`。
迁移只证明找到了指定名字的文件：无声明图片哈希时标明未验证对应关系；有哈希则核对所选图片字节。
原图 image_sha256 不能冒充重新编码的 file_small 哈希。图片必须可解码，重复、缺失或越界时间拒绝。
宿主相机来自 Evidence poses 与实有内参，不补虚构相机；不完整到无法构造摘要或合法Program时拒绝。

报告保存 Evidence/图片/代码/提示词哈希、声明时间和帧索引、相机政策与安全的预算配置；
不写入密钥。视频哈希仅记录已有声明，视频来源和PTS没有在此入口重新核验，明确为 not_checked。
兼容 `meta.source_video_sha256` 与 `meta.video_sha256`；声明必须是合法SHA-256，二者冲突时拒绝。
传入的配置用于打包，不丢弃CLI覆盖。输入、客户端、调用、响应、语法和打包失败保留明确状态；
尝试数须为正，非空输出目录拒绝覆盖。

此入口不执行浏览器。打包完成是 `generation_ok=true / status=packaged_unverified`，
`runtime_ok=null`、`runtime_status=not_checked`，兼容字段 `ok=null`，不能等同于成功重建。
CLI退出码0只表示生成/打包完成，1表示失败，非法参数或非空输出目录为2；后续消费者须检查
所需阶段字段。不同于旧 `ok=syntax_ok` 的语义，旧保存报告不自动修改。
共享运行时不能证明相机或预算公平，fairness_status仍not_established；未执行浏览器验证或模型实验。
PR25 的修补尚未传播到此分支，旧编辑成功率不自动有效；旧 pilot 结果不能作为正式重建优劣结论。

玩法角色推断的三个参照系要说清楚，否则同一个场景换个写法就换一套角色。

够不够得着从玩家站的那一层量起，不拿世界坐标的绝对高度。场景落在哪一层由尺度对齐
怎么定原点决定，是观测不到的规范自由度，整体抬高十米不改变玩家和物体的任何相对关系。
站立面取静态几何里水平投影最大的那块的顶面；物体声明了 `support` 就从所支撑节点的
顶面量起，这样三米高台子上的硬币仍然算够得着，因为玩家能爬上台子。

一个物体的多个 `instances` 全都要算，取最宽松的那个，不只看第一个。只认 `instances[0]`
会让角色跟着数组顺序走：同样五枚硬币，够得着的那枚写在开头还是末尾，整组的角色就不一样。

能不能站上去要把姿态算进去。盒子三组对面各自算水平投影面积和法线偏离竖直的余弦，
只要有一组面积够大又不太陡就算站得住。只看局部 `extent[0] * extent[2]` 分不出躺着和
立着——一块 2 x 0.1 x 2 的板子绕 x 轴转九十度，extent 一个字没变，顶上已经没法站人。
三组面都要比，不能只挑法线最接近竖直的那一组：板子倾斜五十度时窄边比大面更接近水平，
但人踩的是那块四平米的大面。坡度上限 `max_slope_deg` 默认 55 度，跟 `kernel/physics.js`
里 `setMaxSlopeClimbAngle` 的设置对齐，再陡角色控制器就爬不上去了。

原来的 `scene_scale()` 没有调用者，也和本模块「阈值一律以玩家为基准，不以场景尺度为
基准」的前提相反，一并删掉。四元数转旋转矩阵提成 `gwm/compiler/geometry.rotation_matrix`，
几何预检和绑定共用一份，免得两处对姿态的读法漂开。

# Cone dimensions in role inference

Cone extents accept `radius` and explicit `radius_top`/`radius_bottom`, matching
the cylinder radius fallback. The largest declared radius sets horizontal
diameter; height remains the y extent as in the renderer. This fixes valid
top/bottom-radius cones incorrectly becoming decorations. Support-relative
height, multiple instances and rotated standing surfaces are handled separately
in the author's PR42; this change does not implement those policies.

## Ground-truth metric protocol

`gt_metrics.evaluate()` retains `full`/`holdout` metric keys, but the reference
state record alone fixes the time window. Instances use runtime `name` with the
parent object ID retained for motion labels. Identity assignment uses only raw
samples before the reference's tail cutoff and is frozen for both windows;
threshold-constrained matching maximizes valid pairs before minimizing distance.
No endpoint extrapolation is allowed. Each frozen pair is evaluated only on the
intersection of its measured support and the unchanged reference window.
`trajectory.coverage` reports supported duration / requested duration per object;
`evaluated_window_s` reports that intersection (null when there is no positive
overlap). An unmatched identity has null coverage, not measured zero coverage.
`gt_evaluation.min_time_coverage` defaults to 0.9 and can be overridden by the
optional `evaluate(..., min_time_coverage=...)` argument. Below-threshold pairs
have no trajectory error and retain an `unavailable` reason; missing intervals
never contribute zero error. Accepted but incomplete coverage has `partial`
status even if all objects have usable errors. Coverage measures endpoint support,
not interior sampling density; interpolation still assumes the existing track
model. Recall/precision describe frozen identity matches, not
temporal coverage. Objects first recorded after the cutoff cannot be identified
from the prefix and are explicitly reported as unavailable. Legal empty-object
records give zero recall; missing or malformed records and state IDs absent from
their own Program are errors, not default-static motion credit. Only `object`
entries are counted by default, not static scene geometry.

The tail is a diagnostic, not proof that reconstruction inputs excluded it.
Position metrics do not measure rotation, camera or resolve the separate dynamic
physics evaluation policy. Historical `docs/results/gt_metrics_sensitivity.json`
uses the old protocol and has not been re-recorded by this fix; do not reuse it
as validation of the corrected protocol. `check_gt_metrics.py` now records
pass/fail/not-applicable checks and exits nonzero unless all checks pass;
an inapplicable or unmeasurable perturbation is not a successful sensitivity test.

### 可编辑性判定边界

`scripts/run_editability.py` 的 `judge()` 不将命中路径等同于编辑成功。
指令集里的 `expect_ops` 是只提供给判定器的参考补丁，不传入编辑器。
它作用于原始 Program，形成完整参考结果；判定比较对象 ID、值、数组顺序和其他字段，
因此删错对象或删除时顺带改动其他对象也会失败。数值允许 1e-6 绝对误差。
不同补丁写法只要得到相同结果即可通过；未实现任意物理等价表示的判断。
例如不同但运动等价的相位或保留多余静态参数，不会被自动视为等价。
参考指令集必须绑定 `program_sha256`：对解析后的 JSON 排序字典键、保留数组顺序后计算 SHA-256。
格式空白不影响哈希，但对象顺序、身份和初值变化会使基线失效。CLI 在初始化客户端之前拒绝
不匹配或缺失的参考基线；`evaluate_cases()` 也不调用客户端，保留 `input_mismatch` 记录。
直接调用 `judge()` 时，参考案例也必须带基线哈希。结果文件保存实际输入哈希。
旧指令仍支持 `expect_paths` + `expect_value`，但须有 `before`；从目标值构造完整参考结果，
宽泛父路径不能放过同层副作用。缺少原始 Program、只有路径无目标值或目标字段无法核对时，
明确标记无法评测，不算成功。该兼容方式支持修改已有字段；新增/删除使用 `expect_ops`。

`apply_edit()` 保留原返回字段，额外返回 `before` 和 `status`。
只有通过响应 Schema、补丁应用及 Program 校验才是 `applied`；明确拒绝必须提供
`refused: true`、空操作和非空原因。旧格式的空操作保留为 `no_change`，不猜测拒绝。
`execution_failed`、`invalid_response`、`invalid_patch` 和 `invalid_program` 单独记录；
调用失败保留在批次分母中，不冒充正确拒绝。已满足要求的指令允许无操作，
但仍比较完整结果，禁止重复追加事件。输入完整提供给编辑器，不截断运动或遗漏几何/位姿。

修订后的示例明确资产尺寸和相位，纠正材质路径，并同时修改门的范围和实际调度角度。
它不是原指令集的同条件重复实验；旧保存结果及其成功率需要按新协议复核或重新运行，
不能因合成单元测试通过就宣称模型成功率改善。本轮测试不调用模型，也不验证渲染或资产外观。
# Editing judgment precision

Reference edits still compare the complete resulting Program, preserving IDs,
array order and unrelated fields. The default numeric absolute tolerance is
1e-6. A case may specify `tolerance` as a JSON Pointer to absolute tolerance map;
only finite, non-negative tolerances on numeric fields explicitly added or
replaced by `expect_ops` are accepted. The phase case uses 0.001 radians only at
`/objects/0/motion/phase`; other parameters retain the default tolerance.
This avoids penalizing a rounded representation of -pi/2 without hiding edits
to other fields. Tolerances are offline judgment metadata, not model input.
The lift-width instruction now specifies x only, leaving y and z unchanged.
Historical experiment success rates are not re-evaluated by these unit tests.

## Non-no-op event editing case

The instruction set now contains 21 cases, including the original satisfied
coin-contact request and a real event addition. `trigger_door_on_lift` adds a
`trigger_on_enter` event on lift targeting door, enables the door's motion trigger,
and replaces its schedule with a 1.5-second opening from trigger time zero.
The lift is an object entry processed by platformer events; the static door frame
is not, so using the frame without runtime changes would misrepresent support.
Current runtime records target trigger time; it does not independently dispatch
the `action` string. Opening is implemented by the target's triggered schedule.
Offline reference/negative tests check structure, target identity, all three edits
and side effects, not measured model performance or real browser contact behavior.
The previous 20-case experiment rates cannot be reused for this 21-case set.

# Dynamic runtime contract

`motion.mass` is the total mass of an object instance. For compound colliders it
is distributed in proportion to collider volume (uniform density over parts),
not applied in full to every part. Without an explicit mass, Rapier's default
collider density is retained. Overlapping parts are counted independently.

When a scene contains `dynamic` motion, `step(dt)` accepts only the runtime's
fixed `1/60` second step. Unsupported, non-finite or non-positive steps fail
before advancing state. Existing fixed-step callers are unchanged. `seek(t)`
requires a finite non-negative time and samples the nearest fixed physics frame;
it returns that frame's actual time. Repeated requests for the same quantized
frame do not advance the simulation; backward frame requests reset and replay.
Script-only scenes retain exact-time seek. This is not variable-step integration
or a guarantee of cross-platform determinism.

Recording and render indexes use `t`/`actual_t` for the actual simulated state
time and preserve `requested_t` for the sampling request. For encoded recordings,
frame `i` has nominal video time `i/fps = requested_t`; these times may differ
from the simulated state by at most half a physics step. Sample at an aligned
rate such as 30 or 60 fps when exact frame-time correspondence is required.
Render filenames still use requested times. Render indexes declare
`time_sampling` as `exact`, or `nearest_physics_frame` with `dt=1/60` for dynamic
scenes. Validation compares `requested_t` with external requests, checks actual
`t` against the nearest physics frame and `actual_t`, and retains the existing
state-time check. The declaration cannot relax validation for script-only scenes
or unsupported steps. Legacy indexes without the declaration retain exact-time
validation. Duplicate actual times still fail the distinct-frame requirement.

Dynamic velocity/mass/restitution must be finite, with positive mass and
restitution in `[0, 1]`. `dynamic` with `trigger=true`, or a `trigger_on_enter`
event targeting a dynamic object, is rejected by Program validation: activation
semantics are not implemented. Scripted triggers remain supported. This change
does not redesign the motion enumeration or define dynamic ground-truth metrics.

Offline regression: `node --experimental-default-type=module --test
tests/test_physics_runtime.mjs` (Node 20). Tests use real CPU Three.js/Rapier and
the runtime control methods, with renderer/template/browser surfaces stubbed;
they do not certify browser rendering or real-video reconstruction.

规范对齐的启动用投票，不用质心。

对齐本身要先建立起一批匹配才能解析求解，而匹配又要先对齐——破这个循环的是粗对齐。
原来用两边质心之差，有两个毛病。整体转了九十度时，光靠平移永远够不到配对阈值，
一对都建不起来，后面估偏航那步根本启动不了。多一个假物体或少检出一个真物体时，
质心被拽走，原本对得上的也散了。

改成枚举四个轴向偏航，再拿单对物体的隐含平移去投票，取阈值内对上最多的那组。
单对的隐含平移不受别的物体影响，投票天然忽略少数离群的。四个轴向够不够精确不重要，
这一步只要够得着，精确偏航由 align_gauge 在匹配建立之后解析求出。非轴向的大角度
整体旋转仍可能启动失败，这是已知边界。

报告里记 `gauge_alignment.coarse`，出问题时能分清是粗对齐没起来还是精对齐偏了。
