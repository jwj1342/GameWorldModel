# 设计笔记

## 掩码颜色观测（Issue8第一档）

`observe_colors()` 用已关联帧及二值全帧mask采集sRGB像素，逐通道取中位数；
外层按帧遍历，每个有有效mask的帧只解码一次RGB，并依次测量各对象，内存中不缓存整段视频。
图片解码次数不再乘以对象数，但mask重采样及像素中位数的计算仍随对象数增长。
原 `observe_color()` 保留为单对象兼容包装。
每帧等权，再对逐帧中位数取中位数。下采样全帧mask用最近邻恢复到图像尺寸，
不靠bbox补mask、不修改几何/运动。`perception.color_min_pixels`默认4；缺图、
空mask、尺寸比例不符或像素不足时记录原因，无有效测量则为unknown。
`perception.color_aspect_tolerance` 默认0.02，以相对长宽比检查图像和mask，
允许VGGT按patch尺寸取整（如518×294与960×540），明显不匹配仍拒绝。
这一检查不能证明mask没有同长宽比裁剪，完整帧来源仍由调用方保证。
对象可选appearance记录颜色、frame_index/t、图像与mask引用、像素数和失败原因，
Schema与语义门禁检查颜色格式及帧引用；置信度保持unknown，不能把像素数当置信度。
已尝试颜色提取、有mask观测却没有任何有效颜色时，质量门禁输出
`color_measurement_unavailable` warning及失败原因；旧Evidence没有appearance时不补造失败。
直译静态结构和物体优先写material.base_color，模型路径通过Evidence摘要与提示接收；
旧Evidence无此字段或未知时保留原类别材质兜底，不冒充实测颜色。
实测底色只替换调色板底色，保留粗糙度、金属度、透明度及自发光等原先验；
Python的非颜色参数表与JS运行时调色板通过回归测试核对，类别先验不等于实测属性。
模型摘要仅对有实测颜色的对象加一行，颜色代理说明只写一次；模型是否遵循提示仍需实测。
这是观察到的画面颜色代理，不是去光照后的albedo，不能据此推断粗糙度、金属度、
纹理或光照。压缩、mask边缘污染、遮挡、混色和视角会影响数值。
缓存重建可复用下采样mask，但原图片已删除时不能恢复颜色；旧缓存不自动补字段。
本次不构建真实三视频对比或声明视觉保真提升，需后续固定版本实验验证。
颜色提示与本次材质处理会改变输入/输出口径，#3应分别固定合并前后commit、配置、输入哈希与seed。
不得把已使用本PR的实验当作合并前基线。#55的渲染在环颜色拟合不在本改动范围。
## 配置快照与身份摘要的统一脱敏

`redact()` 保守地匹配字段名中的 key/token/secret/password/authorization/credential
子串，只对五个精确名称 `keyframes`、`keyframe_max_side`、`detection_keyframes`、
`max_tokens`、`hazard_keywords` 放行。`apikey`、`api_keys`、`authtoken`、`keystore` 和 camelCase
凭据拼写均隐藏，不再需要名称规范化。新增实验字段若被误遮，需审核后加入白名单，
不能为了保留设置放宽凭据规则。白名单容器内部仍递归脱敏。
凭据 `_file` 路径继续隐藏：它可能暴露个人目录/认证部署信息，且旧配置身份算法
已排除这些路径；统一规则可保留凭据文件轮换的恢复行为。不读取凭据文件。
`configuration_identity()` 对同一脱敏快照按原有规范化 JSON 算 SHA-256；
本次保留 `hazard_keywords` 会使曾将它遮掉的配置摘要变化，旧恢复门禁会拒绝失配；
改变危险词表也会改变新摘要。测试逐份检查已发布 YAML，仅允许显式凭据字段被遮，
防止新增普通设置被静默误伤；不读取凭据文件。旧规则未遮掉的非标准字段（如 `apikey`），
以及现在被保守遮掉的非白名单设置，可能改变摘要，此时恢复仍按原门禁拒绝，
不重写历史或自动放行。密钥只放在被忽略的凭据文件中，不内联进 YAML。
这修复未来配置记录，不能恢复旧报告中已经被 `<redacted>` 替换的采样值；
也不是任意字符串中凭据的通用扫描器，不应将含认证信息的 URL 等写进公开配置。

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

评测入口 `scripts/eval.py` 的四件事，都属于不会报错、只会让表里的数悄悄变成另一个意思的那一类。

录制缓存按输入内容认，不按文件在不在认。缓存键是真值或候选 Program 的 SHA-256、
时长、帧率、画幅，加上 `kernel/` 下所有 js 的哈希，以及 `harness/` 下 mjs、
package.json 和 `scripts/node_harness.sh` 的摘要。内核或录制时间规则改了，都要重录；
旧缓存缺少录制器摘要时自然失效，无须删除已有材料。复用前要求 `record.json` 的帧数、
`gt_states.jsonl` 实际行数及请求的预期帧数三者一致。预期帧数按录制器的
`Math.round(duration * fps)` 计算，正数使用 `floor(duration * fps + 0.5)`，
不能使用 Python 的半整数偶数舍入。录到一半断掉的不算完整结果。缓存目录名带上身份摘要，否则
`out/clipA/run1` 和 `out/clipB/run1` 会共用同一份预测缓存。

真值按名字定位，但名字不构成来源证据。每一行记下匹配是靠目录名还是靠 `meta.clip`
（`truth_link`），报告末尾单独列出所有靠名字定位的片段。有多份候选同时匹配的直接
拒绝，不按 glob 顺序挑第一个——选错一份真值，算出来的误差是假的，而且从表上完全
看不出来。等 #29 那条线上的 `ground_truth_source` 带上生成工具之后，这里改成调用它，
本地这套轻量记录就退役。

评测入口的 JSON 和运行目录清单显式使用 UTF-8 读写，不依赖 Windows 系统编码或
`-X utf8` 开关。此处修正评测入口，不代表所有生产者或编译器的编码已逐一审计。

执行失败和没有真值是两件事。`evaluate()` 判不了的时候返回 `error`，这种行既不能进表
也不能当成跳过；一条都没评出来还退 0 的话，作业脚本会把一次全盘失败当成成功，所以
有失败或者零有效行就退 1。

分组按配置内容分，不按文件名分。`config_label()` 用 run.json 里的 `config_identity`
作后缀，两份都叫 `no_feedback.yaml` 但内容不同的不会被并进同一行。重复给出的运行目录
按解析后的绝对路径去重，并在报告里记下去掉了几条。

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

样式那一半的字段进了 DSL：颜色与光照。

核心主张第四条说同一份程序里样式和行为分成两半写，两分是字段层面的——`geom` 与
`material` 属于样式，`motion` 与 `events` 属于行为。但样式那一半此前只有一个 19 个词
的平色调色板，光照干脆写死在 `kernel/scene.js` 里。

`material` 现在有三种写法：调色板词、`#rrggbb`、或者一个写全表面参数的对象
（`base_color` 必填，`roughness`、`metalness`、`emissive`、`emissive_intensity`、
`opacity` 可选）。对象那条是给「颜色从视频里量出来」留的——调色板那 19 个词是为合成
场景写的，真实物体很少对得上。调色板同时从「任意 32 字符字符串」收紧成闭集：原来写错
一个词，内核查表查不到就落回灰色，不报错，样式那一半的分就这么悄悄丢了。

光照成了 `style.lighting`：`ambient` 的天空色、地面色、强度，`key` 的颜色、强度、方向、
开不开阴影。缺省值与原来写死的那一组逐项相同，所以不写 lighting 的老程序渲染结果不变，
`tests/test_style_kernel.mjs` 盯着这一条。写死的真正代价不是难看——是 5.8 的生成器随机化
了光照之后真值程序里没有地方记它，每条合成样本的样式那一半按构造就是有损的。造数据集
排在这之后，所以这组字段要先落地，否则数据集造出来就是对不上第四条主张的。

调色板有两份——schema 的 `$defs.materialName` 和 `kernel/scene.js` 的 `MATERIALS`。
`tests/test_style_half.py` 比对两边的键，漂开就失败，否则 schema 放行的词内核渲染成灰色，
正是上面那个洞。

贴图和网格几何不在这一轮。贴图要先定素材从哪来、怎么随程序走，网格是另一件事；
加一个运行时不认的字段比没有这个字段更糟。RP §6 缺口 14 要跟着改一半，它在
`docs/research-proposal` 分支上。

数值字段的有限性在校验的第一步统一查，不分样式还是行为。NaN 跟任何上下界比较都是假，
schema 的 `minimum`/`maximum` 拦不住它；`1e400` 这种合法 JSON 字面量解析出来是 inf，
没写上界的字段也拦不住。两种都会原样写进 `program.json`，浏览器 `JSON.parse` 读到裸的
NaN/Infinity 直接 SyntaxError，而编译这一步会报成功——产物是坏的，日志是干净的。
所以 `validate.py` 递归走整棵程序树，不逐个列字段：列字段的写法得跟着 schema 一起改，
漏一个就是一个无声的洞。`bundle.py` 写 program.json 时再用 `allow_nan=False` 兜一道。

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

### 奖励实验来源清单生成与人工审核

`python -m scripts.create_ground_truth_association` 根据显式指定的三个文件计算
视频、独立参考 Program、生成/参考记录的 SHA-256，复用来源契约生成清单。
所有输入必须位于清单所在目录之下，使用可移植相对路径；输出必须是新文件，不覆盖旧清单。

```bash
python -m scripts.create_ground_truth_association \
  --video out/source_trial/video.mp4 --program out/source_trial/reference.json \
  --record out/source_trial/generation.json --out out/source_trial/associations.json \
  --association-id source_trial --duration 6 --kind synthetic_generation \
  --review-note "待人工核对生成来源和时间对应"
```

这里只计算文件身份和检查所声明时长，不推断文件名对应，不证明视频由 Program 生成，
不解码 PTS，也不能识别输入是否实际上是当前候选 Program。操作者必须提供独立参考。
生成的 `review.status` 永远是 `pending`，奖励入口会拒绝。人工检查来源记录、视频关联、
时间覆盖与独立性后，在审核副本填写理由并明确改为 `confirmed`，才可传给
`check_reward_agreement.py --associations`。实际文件变化仍由现有哈希门禁拒绝。
非零时间偏移和速率变换尚未支持；不得以生成清单为由宣称完整评测协议已验证。

录制器保留上游 `t/actual_t` 真实仿真时间与 `requested_t` 采样请求，同时保留默认关闭的
`--include-endpoint`。奖励实验要求完整且实际时间与请求一致的闭区间；固定步长场景中
不对齐物理帧的请求将被拒绝，而不是改写真实时间。默认录制仍保留量化时间。
可选额外终点若不在常规帧率网格上，不能直接当作普通编码帧时间；需独立核对解码 PTS。

### 评测录制缓存与终点采样

`scripts.eval.states_of(..., include_endpoint=True)` 可显式请求补终点，默认仍为
`False`，批量评测默认采样策略不变。该开关同时进入 `record_key()` 并传给录制器。
缓存校验读取 `record.json.include_endpoint`，要求它与请求一致；按录制器的
`Math.round`、空序列先采零点及 `1e-9` 终点容差计算帧数，再核对状态行数。
字段只接受布尔值，不把字符串解释为开关。旧 metadata 缺字段按默认关闭处理，
旧缓存身份缺此开关会失配并重录，不改写旧轨迹来冒充新模式。
这只修复缓存身份和完整性，不证明视频 PTS、物理轨迹或真值来源正确。

### 合成数据集的场景生成器

生成器把一个场景拆成三路独立采样：**行为**（场景结构、几何、运动脚本）、**样式**
（背景、光照、每个节点的材质）、**观测**（相机轨迹）。三个种子分开给，所以可以固定
两路只变一路。这不是为了代码好看——「同行为不同样式」这组配对就是核心主张第四条在
数据这一侧的证据，也是可编辑性实验的标准答案：一次只改样式，应该且只应该改到
`style` 子树和各节点的 `material`。

这条分离比看上去难守。第一版里材质和 `style` 子树共用一个随机源，而材质的抽取次数
等于节点数，节点数又是行为那边定的，于是只换行为种子会把光照也带着改。配对测试当场
就挂了。现在样式那一路拆成两个流：`style_seed` 给 `style` 子树，`style_seed` 异或一个
常数给材质，谁都不受对方抽取次数的影响。

**相机四档里没有「固定俯视」。** RP 5.8 原本列了这一档，但机位不动就没有基线，而
合成数据必须跑一遍真实感知管线（否则训练时用真值证据、推理时用带噪感知，模型在真实
视频上直接崩），没有基线的位姿估不出来。留着这一档等于往训练集里塞一批建立在估不出的
位姿上的样本。换成 `overhead_dolly`：仍然是俯视，但机位沿直线平移，基线有了，俯视这个
因子也保住了。四档各自都有一条测试盯着首末机位的距离，手持那一档的平移量按「方向加
模长」抽而不是三个分量各自抽，就是因为后者会抽出模长接近零的那一组。

**场景家族分成互不相交的两组**：训练用 `room`／`platform`／`corridor`，测试从
`yard`／`track` 生成。不这么分的话主表全是同分布结果，说明不了泛化。

**运动类型的抽样比例两套，分开说明。** 测试集八类等权，因为表 1 的混淆结构要求每一格
都有样本；训练集按常见度配（静止和周期运动多、自由落体少）。测试那套有一条测试卡住
每类占比不低于 6%，训练那套只要求每类都不为零。

**类名池故意比绑定词表宽。** 数据集最容易被审稿人抓住的坑是「DSL 表达不了的东西在
数据里不会出现，分数自然好看」，其中最具体的一条就是场景里恰好叫 `coin` 和 `lava`，
而绑定规则的词表里恰好有 coin 和 lava——等于自己跟自己对答案。现在大部分类名
（crate、bollard、spool、girder……）落在词表之外，少部分落在里面，命中个数逐条写进
manifest。这个比例是要报出来的数，不是埋着的坑。

**生成事实不进程序。** 家族名、相机档、profile、种子都只进 manifest。程序是模型要
预测的答案，把这些写进 `meta` 等于在答案里留小抄，模型会学着背出来，而真实视频上
没有这一行。`meta` 的键被一条测试锁死在 clip/fps/duration/units/up/notes 六个之内。

**事件这一列暂时还是空的，原因和原来想的不一样。** 原以为只是生成器产不出带事件的
样本，实际更深一层：`despawn_on_contact` 和 `trigger_on_enter` 都要靠玩家接触才触发，
而 `template.update()` 只在 play 模式跑，录数据集用的 `record.mjs` 走的是 replay 模式
（`kernel/main.js` 里 `if (game.modeName === 'play') template.update(...)`）。所以即使
真值程序里写了事件，录出来的视频里也什么都不会发生——带事件的真值配上没有事件的画面，
是不可学的样本。要填这一列必须换成 play 模式加自动试玩来录，让接触真的发生。这件事
没有在生成器里做。

**`dynamic` 这一档是「可外推」那条主张的载体，所以测试集里不能少。** 八类运动里
`dynamic` 是唯一一个位姿不由时间的纯函数给出的——程序里写的是质量、恢复系数和初速度，
轨迹由物理引擎跑出来（`kernel/motions.js` 的 `isSimulated`）。这正是「样式与脚本分离」
之外的另一半：脚本那一半说「什么时候在哪」，物理那一半说「它是个什么东西」。
一个写成 `dynamic` 的场景可以往视频结束之后继续推，写成 `trajectory` 的推不动——
关键帧在视频末尾就没了。测试集八类等权，`dynamic` 占到约六分之一，这条主张才有数可报。

## 资产检索与直接操纵（2026-10-17）

### 为什么重建出来不像

把三段真实视频的产物翻出来逐层看了一遍，瓶颈不在模型也不在预算，在表示能力。链路上有五段天花板：检测词表只给了三个词、物体上限 6，所以一个塞满东西的房间只检出 4 个物体；物体几何是拟合到可见表面的有向包围盒，也就是一个盒子；`asset` 这种几何形式在内核里一律退回基本体；地面法向偏离竖直 34 度，整个场景是斜的；最后，反馈分数只对**已配对上的物体**求平均，场景里少了三十个物体一分都不扣。

决定性的证据是反馈循环的分数：三轮下来精确到小数点后十六位都没变。循环本身是对的（生成补丁、重渲染打分、提升不够就回滚），它只是找不到任何能改进的候选——权重一半压在掩码 IoU 上，而一个盒子和一列小火车的掩码 IoU 怎么挪都是零。优化器是好的，搜索空间里没有解。

### 资产这一层

按本文档前面那节定下的方案实现：CC0 的 Kenney 镜像（4812 个 GLB，50 个 kit），全库渲 4 视角缩略图（286 秒），DINOv2 嵌成图像索引（19248 条向量，自查 top-1 60/60）。

检索是**文字决定收不收、图像决定收哪个**。两个分数不混在一起算总分：DINOv2 比的是真实照片和低多边形渲染，余弦相似度天然偏低，掺进总分再去和为文字分调好的阈值比，会把本来正确的结果一起砍掉。重排也顺带挡住一类错误——没进文字候选的资产不会被图像分拉进来，所以不会出现「书架」匹配到「门洞」。

挑出来的资产写进 `geom.ref`。这是事后标注，不是模型要预测的字段，训练数据里不带它。

### 和可交互性的边界

`buildNode()` 返回 `{group, colliders}` 两份东西，碰撞体**永远**从包围盒算，哪怕这个节点挂了真网格。所以换外形不改变任何物体的可拖动性、可踩性和物理行为。这条边界是有意固定下来的，它就是论文里「表征与交互分写」那条主张在代码上的落点。

### 可交互性分档

| 档 | 是什么 | 实现 |
|---|---|---|
| 观测运动 | 复现视频里看到的运动 | `motion` 字段 + `kernel/motions.js` |
| 直接操纵 | 用鼠标抓住物体拖动，松手后物理接管 | `kernel/manipulate.js` |
| 关节交互 | 抽屉沿轴滑、柜门绕铰链转，带限位 | 未实现 |
| 玩法交互 | 角色走到目标、收集、受伤 | `kernel/templates/` |

拖动用速度驱动而不是直接改位姿：直接改位姿会穿过别的物体，速度驱动是把它推过去，碰撞、堆叠、把挡路的东西撞开都照常发生。

### 接触装配

直译出来的位姿直接取自包围盒中心，加上地面法向本身有误差，经常有物体陷进地里。`gwm/synthesis/contact.py` 把它们抬到地面上。这一步和 `gwm/compiler/repair.py` 是两件事：那边是保守修复，只在证据支持时才动位姿；这里是装配，明确承认「让物体落到支撑面上」是我们加的先验，所以每次移动都记下来。

### 当前还差什么

资产这层接通之后，瓶颈换成了摆位和尺度：物体之间离得太远、尺寸不对，因为包围盒是拟合到可见表面的，而且度量尺度是靠假设相机高度推出来的。这是 #7 和度量尺度那条线的事，不是资产层能解决的。

### 度量尺度与尺寸

单目视频恢复不出绝对尺度，总得有个锚。原来的锚是「相机离地 1.2 米」。俯拍桌面时，
被当成地面的其实是桌面，相机只高出半米左右，整个场景会被放大两三倍。

改成用物体尺寸当锚，先验有两个来源：命名那一步本来就问过每个物体的 `size_m`（答案一直
存在 `namer.json` 里，从没被用过），以及一张常见物体最长边的表。

实测这件事的结论比预期有意思：全局倍数只有 ×0.99，但**逐物体比例的离散度是 11**。
桌子 1.14、书架 0.83、纸箱 1.55 都还行，玩具火车却被量成了 1.14 米（真实约 0.2 米）、
台灯量成 0.99 米（真实约 0.45 米）。也就是说场景不是整体缩放错了，是**小物体被单独
撑大了**——包围盒拟合的是可见表面，而深度噪声是个绝对量，物体越小它占的比例越大。

所以分成两步，各记各的：

1. **全局尺度**只采信长边不小于 0.5 米的物体。让小物体参与投票等于让噪声最大的样本
   决定全场尺度。
2. **逐物体回收**：偏离先验超过两倍的物体，按先验把尺寸收回来，只改大小不改中心和
   朝向。这一步是先验压过测量，所以单独做、每条都记下来。

顺带修了 `scale_program()` 漏掉的两个量纲字段：相机关键帧的 `look_at`（只缩 pos 不缩它，
场景一放大相机就瞄到别处）和 `motion.linear_velocity`（场景放大后物体会显得变慢）。

### 场景完整度

旧的反馈分数只对**已配对上的物体**求平均，所以场景里少三十个物体一分都不扣——这就是
重建出五个盒子也能一路通过门禁的原因。`gwm/feedback/completeness.py` 按两轴报，刻意
不合成总分：合成之后，「什么都没重建出来但剩下那个能拖」会拿到不低的分，而那正是现在
最该避免的失败模式。

两轴里最要紧的一项是 `evidence_pixel_coverage`。在 plarail 那段视频上，`object_recall`
是 1.0——检出的九个物体全都重建了出来——但画面只有 **48.8%** 被解释。只看召回率会以为
一切正常，而「不像」主要就来自那没被解释的一半。

### 检测召回：画面覆盖的主要杠杆

完整度指标上线之后第一件事就是查「画面只有一半被解释」是从哪来的。在同一段视频上扫了
四组设置（`detection_keyframes` 是在几个关键帧上做检测，然后跨帧关联）：

| 设置 | 物体数 | 感知耗时 | 画面覆盖（逐帧平均） | 覆盖（最好的一帧） | 资产命中率 |
|---|---|---|---|---|---|
| 1 帧（原默认） | 9 | 49.5 s | 0.227 | 0.503 | 0.889 |
| 3 帧 | 31 | 73.9 s | 0.294 | 0.617 | 0.742 |
| **6 帧** | **62** | **107.7 s** | **0.529** | **0.815** | 0.742 |
| 6 帧 + 低阈值 | 190 | 240.6 s | 0.778 | 0.929 | 0.579 |

默认从 1 改成 6：物体数七倍、画面覆盖两倍多，代价是感知耗时两倍出头。再往下降阈值能把
覆盖推到 0.78，但资产命中率从 0.74 掉到 0.58——检出一堆对不上任何类别的碎片——而且耗时
接近五倍，所以阈值不动。

这项改动会直接影响 #21 的算力预算，感知那一档要按两倍重算。

**顺带修了完整度指标自己的一个缺陷。** 第一版的画面覆盖只统计了碰到的第一帧，于是物体数
从 9 涨到 62 的时候这个数纹丝不动（都是 0.488）——因为多关键帧检出的物体，掩码是从它被
检到的那一帧才开始有的。改成逐帧算再平均，同时也报最好的那一帧。一个在该动的时候不动的
指标比没有指标更糟，因为它会让人以为改动无效。

### 摆位：让场景自己稳下来

感知给的位姿是按**可见表面**拟合的包围盒中心，于是物体常常半悬空，或者和桌面、和别的
物体穿插。这种场景看着就不对，而且一旦用鼠标去拖，物体会从别的东西里被拽出来——可交互
性先要求物体本来就站在该站的地方。

做法是把编译好的场景放进我们自己的物理里跑一秒半（`harness/settle.mjs`），让它落到
支撑面上、把穿插解开，再把稳定之后的位姿写回程序。只动没有脚本运动的物体：有脚本运动
的物体位姿由 `pose(t)` 决定，不该被重力改掉。

采纳规则单独放在 `gwm/synthesis/stabilise.py`，和仿真分开。位移超过 0.8 米的不采纳——
那不是「稍微悬空了一点」，而是这个物体根本没有支撑或者尺寸估错了，让它一路掉下去只会
把一个看得见的错误变成一个看不见的错误。

有一个容差上的坑：仿真允许极小的穿透来保持接触稳定，而校验器按几何判定，两边对不上，
于是稳定之后反而报 `ground_penetration`。所以稳定之后要再过一次接触装配，把贴着地面的
物体精确推到表面上。万一仍然不过，整体回退到稳定之前那一版——稳定是锦上添花，不该把
整条链路挡死。

在那段真实视频上：63 个物体里 47 个被仿真挪动，最大位移 0.70 米，3 个因为超限被放弃，
稳定之后又有 24 个需要重新贴合。

### 运动估计为什么必须换信号（#7 的证据）

在五十条**有标准答案**的合成片段上量了一遍，结论是运动估计不是精度不够，是基本没出结果：
696 个证据物体里 515 个的 `motion_guess` 是 `unknown`（74%），对齐规范自由度之后的运动
类型准确率是 **0.137**（五十条片段共 50 个配对物体）。

> 这里原来写的是 0.000。那个数是量具糊掉的：准确率汇总到物体一级之后是一串 0 和 1，
> 而汇总表对每一列都印中位数，0/1 序列的中位数只有 0 和 1 两个取值。判对不到一半就
> 恒等于 0.000。真实值一直是 0.137。结论方向没变——七成四根本没出结果、最高候选置信度
> 中位数 0.058——但「五十条无一例外」这句话是错的，已改。

估计器自己记下的拒绝原因里，七成是「几个运动假设分不开」。但真正说明问题的是这个数：
**最高候选的置信度中位数只有 0.058**（阈值 0.55）。不是两个模型难以取舍，是每个模型都
拟合得很差。所以调阈值没有意义——调松了只会把置信度 0.058 的拟合填进程序，行为那一半
会从「空着」变成「写满了错的」，对训练数据而言后者更糟。

排除掉的解释（脚本固化在 `scripts/probe_motion_signal.py`，用真值感知跑，世界坐标系
和真值一致，不需要对齐规范自由度）：

| 假设 | 做法 | 结果 |
|---|---|---|
| 轨迹太短估不出来 | 按被看到的帧数分桶 | **不成立**。被看到 60–100% 帧的成功率 0.26，比只被看到 10–30% 的 0.40 还低 |
| 偏移是「只看得见近侧」 | 沿视线推半个深度尺寸 | **过校正**。误差 0.294 → 0.620 m |
| 偏移是个随视角转的刚性量 | 每个物体估一个与时间无关的沿视线标量（闭式解） | **只改善 8%**。静止物体漂移 0.154 → 0.142 m |
| 换一个逐帧统计量 | 点云中位数 / 均值替代包围盒中心 | **更差**。位置误差 0.244 →（中位数 0.541 / 均值 0.469） |

掩码完全准确时，包围盒中心的位置误差中位数是 0.244 m，其中约一半沿视线、一半垂直视线。
负的沿视线分量说明中心确实偏在靠近相机的一侧，机制对，但只解释一半。

综合起来：**可见点集的任何逐帧汇总量都给不出干净的轨迹**。包围盒中心是可见点在各轴上
3/97 百分位的中点，可见面一变两端就变；换成中位数或均值更糟，因为可见表面的重心本来
就落在表面上。信号必须来自跨帧的对应关系，而不是逐帧的统计量——这就是 #7 说的三维点
轨迹，它不是优化，是必要条件。

### 跨帧刚体配准：第一个正向结果

三条廉价修法失败之后，换信号的那一步做出来了：`gwm/perception/registration.py`。它不问
「这一帧的中心在哪」，只问「这一帧的点云相对之前转了多少、挪了多少」——相邻帧的可见面
重叠最多，所以这个问题是良态的。

路上有两处实测得来的设计：

**不逐帧链式配准。** 第一版是把每一步 `icp(当前帧, 上一帧)` 的结果连乘起来。每步残差
只有 0.026 m，看起来很好，但轨迹误差 0.298 m，**比包围盒中心的 0.220 m 还差**——每步的
小误差沿三十帧累积起来盖过了信号。改成维护一个「物体自身坐标系下的累积模型」，每来一帧
先用上一帧的位姿当初值再和模型配准，轨迹误差降到 0.203 m。

**模型只吸收前几帧。** 让累积模型一直长下去的话，静止物体会跟着可见面缓慢滑移。限制成
只有前六帧进模型之后，轨迹误差进一步降到 0.188 m。

二十条合成片段、2390 个样本（掩码取自真值渲染，世界坐标系与真值一致）：

| 信号 | 相对第一帧的位移误差中位数 | 真值静止物体漂移中位数 |
|---|---|---|
| 包围盒中心 | 0.253 m | **0.182 m** |
| **跨帧配准** | **0.188 m** | 0.286 m |

比的是**相对第一帧的位移**而不是绝对位置：配准给的是相对锚点的轨迹，锚点落在可见表面上
本来就偏，而运动估计用到的正是位移这个量。

轨迹误差好 26%，但静止物体的漂移还更差（只有 6 个样本，统计上不牢，但方向一致）。下一步
是把这个信号接进 `build_evidence` 并重跑五十条，看 `unknown` 比例和运动类型准确率动没动；
静止那一档可能需要单独处理——配准解出来的每步修正量接近单位阵时就是静止，这个判据比
看累积位置稳。

### 关节推断为什么默认关着

`gwm/perception/registration.py` 的 `infer_articulation()` 能从配准的逐步刚体变换里读出
铰链轴和滑轨行程——每一步都绕大致同一根轴转就是铰链，几乎不转而平移方向一致就是滑轨。
机制在合成点云上是对的（`tests/test_registration.py` 里造了已知的门和抽屉，轴、支点、
行程都解得回来）。

但在真实的感知结果上它不够准。四十条有标准答案的合成片段（`revolute` 和 `prismatic`
就是真关节，别的都不是），扫阈值的结果：

| 转角 | 轴一致性 | 行程 | 直线度 | 报出 | 查准 | 查全 | 类型判对 | **可用率** |
|---|---|---|---|---|---|---|---|---|
| 8° | 0.85 | 0.05 | 0.90 | 16 | 0.312 | 0.227 | 0.40 | **0.125** |
| 15° | 0.90 | 0.10 | 0.93 | 6 | 0.50 | 0.136 | 0.667 | **0.333** |
| 25° | 0.93 | 0.15 | 0.95 | 1 | 0.0 | 0.0 | — | 0.0 |
| 40° 及以上 | | | | 0 | — | 0.0 | — | — |

可用率是「报出来的东西里既是真关节、类型也判对」的比例。最好的一档只有 0.333，再收紧
就一个都报不出来——**没有哪组阈值能让它可信**。

真实视频上的表现和这组数一致：plarail 那段里它在「毯子」和「书架」上判出了铰链，三条
都刚好压在阈值线上。

所以默认关着（`perception.motion.infer_articulation: false`），代码和探针都留着。
报一个假关节比不报更糟：用户拖一下就会发现，而且写进训练数据会被模型当成答案学。
等运动信号再好一些（现在配准的轨迹误差 0.188 m，静止物体漂移还比包围盒中心大）再打开，
用同一套口径重扫一遍阈值。

### 每步位移分不开静止和运动

跨帧配准解出的**逐步**修正量，一度被我当成判静止最稳的信号：累积位置会被缓慢滑移带偏，
而真正没动的物体每一步修正都该接近单位阵。合成点云上确实如此（测试里造的静止物体，
可见面在变但每步修正很小）。

真实感知结果上不成立。二十五条有标准答案的合成片段：

| | 每步位移 / 物体对角线（中位数） |
|---|---|
| 真值静止的物体 | **0.165** |
| 真值运动的物体 | **0.179** |

两条分布基本重合。按原来 0.02 的阈值，静止物体上**一次都没触发**，运动物体上误触发
5.4%——净有害。而且这不是阈值没调好：0.165 和 0.179 之间没有任何一刀能把两类分开。

这个数还说明了一件更要紧的事：**配准的每步噪声是物体自身尺寸的 17%**。一步是 0.25 秒，
也就是说逐帧看过去，物体在自己身上「跳」了六分之一。难怪八种运动模型都拟合不上，
最高候选置信度只有 0.396。配准改善的是**累积轨迹**（相对位移误差 0.253 → 0.173 m），
逐步量仍然很噪。

所以静止判据默认关掉（`perception.motion.use_static_detector: false`），函数和测试留着。
下一步要降的是逐步噪声本身，而不是在它上面再加判据。

### 点到面 ICP 为什么更差

点到点 ICP 在平坦表面上会沿着面滑动，而我们的物体大多是盒子和平板，所以点到面看起来是
最该试的改进。实测相反：

| | 相对位移误差中位数 | 真值静止物体漂移 | 每步位移 / 对角线 |
|---|---|---|---|
| 点到点 | **0.173 m** | **0.231 m** | 0.165 / 0.179 |
| 点到面 | 0.293 m | 0.462 m | 0.317 / 0.358 |

原因想明白之后很直接：我们的点云是**单视角深度图**，一帧只看得见朝向相机的那一面，
所以这些点的法向几乎全都指向相机。点到面只惩罚沿法向的偏差，于是它只约束了深度方向，
把切向那两个自由度彻底放开——正好和「防止沿面滑动」的初衷相反。点到点至少对三个方向
一视同仁。

实现和测试留着（在合成的完整点云上它是对的，测试里那块薄板沿法向的位移解得出来），
默认不用。

**这组数还澄清了一件更要紧的事。** 上面这些探针跑的是真值感知，深度来自源程序的渲染，
掩码来自 ID 通道——**深度和掩码都是完美的**。所以 0.173 m 的轨迹误差和 17% 的每步噪声
都不是传感器噪声，而是「每帧只看得见一个面」这件事本身带来的：相邻两帧的可见面不同，
配准必须在两块不同的局部表面之间找对应，切向上本来就欠定。

换句话说，瓶颈不在深度估计的质量上，也不在配准算法的选择上，而在**单视角观测的欠定性**。
能真正压下去的方向是把多帧的观测联合起来解（而不是逐帧两两配准），或者利用已知的相机
轨迹去预测可见面的变化。

### 画面覆盖这个指标测的不是「场景被解释了多少」

我用 `evidence_pixel_coverage` 报了好几轮「画面只有一半被解释」。把真值感知和神经感知
在同一条片段上一比，这个说法站不住：

| | 掩码覆盖画面 | 物体数 | 最大的那个掩码 |
|---|---|---|---|
| 真值感知（掩码取自源程序的 ID 通道） | **0.046** | 10 | 2.1% |
| 神经感知 | **0.472** | 18 | **67%** |

真值感知覆盖的才是「场景里真的有什么」，只有 4.6%——这些合成场景里物体本来就小。
神经感知覆盖十倍，是因为它检出了两个巨大的东西：一个占画面 67% 的「slab」和一个
占 61% 的「ground」，也就是把地面当成了物体。

所以这个指标衡量的是「有多少像素被检测**声称**占有」，不是「场景被解释了多少」。大的
虚检会让它变好看，真实的重建质量反而看不出来。它仍然是个有用的诊断量（能看出检测在
声称什么），但不能当保真度指标用。

真正的保真度应该是**把重建出来的场景按输入的相机轨迹重渲染，和原视频逐帧比**。这件事
的零件都有（`gwm/feedback/metrics.py` 里算渲染图和原帧的 DINO 相似度就是这个），只是
还没接进完整度报告。

### 占画面太大的检测按结构件处理

上面那个 67% 的「slab」不只是把指标撑大了：它会进程序的 `objects` 列表，变成一块和整个
场景一样大、还能用鼠标拖动的板子。真值里最大的物体只占 2.1%。

现在掩码占画面超过 `synthesis.max_object_area_frac`（默认 0.35）的检测一律按结构件处理，
并在节点的 notes 里注明「按面积判为结构件，掩码占画面 67%」——不能悄悄降级，报告里要
看得出这个节点为什么跑到 static 去了。不丢掉是因为大而平的东西通常确实是结构。

### 重渲染相似度：第一个直接回答「像不像」的数

按输入视频的相机轨迹把重建的场景渲染出来，和对应时刻的原帧比 DINOv2 嵌入的余弦相似度。
这是整条链路一直缺的那个指标——此前所有指标都只在测侧面（掩码覆盖率测的是「有多少像素
被检测声称占有」，物体召回测的是「检出来的有没有重建」）。

**光报一个数没有意义**，所以一起报两条标尺：

| | 含义 | 实测 |
|---|---|---|
| 上限 | 同一段视频不同时刻两帧互相比。视角差异还在，重建再好也追不上 | 0.54 ~ 0.97 |
| 下限 | **另一条片段重建出来的场景**对本片段的原帧 | 0.026 ~ 0.030 |
| 重建 | | 0.080 ~ 0.195 |
| **归一化** | 落在下限和上限之间的位置，0 是场景完全没对上 | **0.056 ~ 0.057** |

也就是说，现在的重建只走完了「从完全不对到追平视角差异」这段路的 **6%**。

**下限的选法踩过一个坑，值得记。** 第一版用另一段**真实视频**的帧当下限，算出来归一化是
负的（−0.045 / −0.168），读起来像「重建出来的比一段无关视频还不像原视频」。但真实视频
之间共享「这是照片」这类低层统计，低多边形渲染没有，所以那条下限同时在测**领域差异**。
换成另一条片段的**重建场景**之后，下限从 0.118~0.210 掉到 0.026~0.030，可见这个混淆有多大，
而归一化也从负数变成了合理的 0.056。

一个没有正确对照的指标比没有指标更危险：它会给出看似精确的错误方向。

### 用重渲染相似度做的第一次消融

有了唯一直接衡量「像不像」的指标之后，第一件事是拿它回答：我加的这几级，哪些真的让
场景更像原视频。三条真实视频，每条跑四个变体，下限固定成同一条参照场景：

| 变体 | plarail | 传送带 | 牛顿摆 |
|---|---|---|---|
| full | 0.281 | 0.098 | 0.017 |
| 关掉资产检索 | 0.234 | 0.043 | **0.033** |
| **关掉尺度与尺寸回收** | **0.473** | 0.098 | 0.022 |
| 关掉物理稳定 | 0.224 | **0.110** | 0.025 |

**资产检索是正向的，两条片段上明显。** 传送带那条开着 0.098、关着 0.043，翻了一倍多；
plarail 0.281 对 0.234。只有牛顿摆是负的（0.017 对 0.033）——那是一段昏暗场景里的金属
摆锤，而库里配上去的是彩色低多边形游戏道具，风格差得太远。这支持「问题在资产库的风格
而不在资产这条路线」。

**尺寸回收是负向的，而且是整张表里最大的单项效应。** plarail 上关掉之后从 0.281 涨到
0.473。原因大概是：把物体按先验缩回去的同时，相机距离并没有跟着改，于是取景整个变了
——度量上更对，画面上更不像。所以默认关掉；要重新打开的话，得把相机和物体尺寸一起解。

**物理稳定两边都有。** plarail 上是正向的（0.281 对 0.224），另外两条片段上略负。那两条
场景里物体少、本来就没什么要稳的，稳定带来的位移反而把东西挪离了原位。

**消融本身踩过一个坑。** 第一版没固定下限那条参照——它是从「另一条已经重建好的片段」
里自动挑的，而随着各变体陆续产出，挑到的那条会变（实测 0.067 和 0.079 两个值）。变体
之间的差值只有 0.01 量级，和这个漂移同量级，整组数读不出任何东西。加 `--likeness-reference`
固定之后差值才显出来。对照组必须固定。

### 用指标驱动的两处改动，保真度从 0.281 涨到 0.445

消融指出两件事：尺寸回收是负向的（最大单项效应），资产在昏暗的金属场景上是负向的
（风格不匹配）。按这两条各改一处，再用同一套消融复测：

| | plarail（改前 → 改后） | 牛顿摆（改前 → 改后） |
|---|---|---|
| full | 0.281 → **0.445** | 0.017 → **0.057** |
| 关掉资产 | 0.234 → 0.389 | 0.033 → 0.066 |
| 关掉尺度 | 0.473 → 0.409 | 0.022 → 0.055 |
| 关掉物理稳定 | 0.224 → 0.397 | 0.025 → 0.028 |

plarail 上涨了 58%，牛顿摆涨了 3.4 倍。更要紧的是**每一级在 plarail 上都变成正向的**了：

| | 改前 | 改后 |
|---|---|---|
| 资产检索 | +0.047 | **+0.056** |
| 度量尺度 | **−0.192** | **+0.036** |
| 物理稳定 | +0.057 | +0.048 |

尺度那一项从严重负向翻成正向，是因为把「尺寸回收」（按先验强行缩物体）和「全局度量
尺度」拆开了：前者默认关掉，后者保留。原来两件事混在一个开关里，被前者的负面效应盖住。

牛顿摆上资产仍然略负（0.057 对 0.066），但差距从 0.016 缩到 0.009。

> **2026-10-18 更正：把这 0.007 归给色彩迁移是不成立的。** 后来直接量了重渲染相似度
> 对颜色的敏感度：把每个物体的色相旋转半圈（每条片段改 11 到 22 个物体，形状、摆位、
> 光照一个不动），六条片段的相似度中位数只从 0.5635 变到 0.5568，而且其中两条反而
> 涨了。也就是说**这个指标几乎看不见颜色**，幅度和片段间噪声同量级。
>
> 上面那 0.007 落在这个噪声里，不能说明色彩迁移起了作用——也不能说明它没起作用，
> 只能说这个指标回答不了这个问题。本节主要的那个增益（0.281 → 0.445）来自尺寸回收
> 那一处**几何**改动，那条归因不受影响。
>
> 后果更大的是：**样式那一半目前没有能验收的指标。** 要做颜色相关的改动（#55 的渲染
> 反馈闭环），得先补一个对颜色敏感的度量，比如掩码内颜色的逐物体误差——合成数据上
> 材质颜色是已知真值，可以直接算。复现：`scripts/probe_colour_headroom.py --corrupt`。

### 三条真实视频的完整消融（改动之后）

下限参照一律用**另一条**片段的重建场景（传送带那条第一次跑错过：参照用的是它自己，
「下限」变成了同一个场景的重建，归一化被压低）。

| 变体 | plarail | 传送带 | 牛顿摆 |
|---|---|---|---|
| **full** | **0.445** | **0.034** | **0.057** |
| 关掉资产检索 | 0.389 | **−0.011** | 0.066 |
| 关掉全局度量尺度 | 0.409 | 0.034 | 0.055 |
| 关掉物理稳定 | 0.397 | 0.049 | 0.028 |

逐级的贡献：

| | plarail | 传送带 | 牛顿摆 |
|---|---|---|---|
| 资产检索 | +0.056 | **+0.045** | −0.009 |
| 全局度量尺度 | +0.036 | 0（这条没有足够先验，没改尺度） | +0.002 |
| 物理稳定 | +0.048 | −0.015 | +0.029 |

**资产检索是最稳的正向项。** 传送带那条关掉资产之后归一化变成**负数**（−0.011），也就是
重建出来的还不如另一个场景的重建像——那条片段里几乎全是纸箱，没有资产就只剩几个灰盒子。

**物理稳定三条里两条正向。** 传送带上是负的，那条场景里物体少、本来没什么要稳的，
稳定带来的位移反而把东西挪离了原位。

**牛顿摆仍然是三条里最低的**，而且资产在它上面仍然略负。那是一段昏暗场景里的金属摆锤，
色彩迁移把彩色游戏道具往深灰拉了一截（负向从 −0.016 缩到 −0.009），但低多边形的造型
和真实摆锤的差距还在。那是资产库的问题，不是这条路线的问题。

### 多帧联合求解：两遍精化反而更差

逐帧两两配准的误差会沿帧累积（实测每步残差只有 0.026 m，轨迹误差却有 0.298 m），所以
自然的改进是把链断开：拿全部帧建一个完整模型，每帧**独立**对它配准。

实测反而更差，而且原因比结果有意思：**一帧只看得见一个局部表面，拿它去配准一个更完整
的模型时，这块局部可以滑到模型上好几个位置都说得通**；而相邻两帧的可见面大小相当、
重叠又多，反倒是良态的。部分对完整比部分对部分更欠定。

这也回头解释了之前扫模型帧数时「6 帧比 12 帧好、12 帧比全部帧好」那组数——同一个机制。

实现（`refine_poses`）留着，默认关（`refine=0`），并写了一条测试把这个事实钉住，免得
以后有人再想一遍「为什么不用全部帧建模型」。

**这条排除掉之后，#7 剩下的方向就很窄了**：既然「局部对局部」是良态的、「局部对完整」
不是，那压噪声就不能靠扩大模型，只能靠给轨迹本身加约束——比如联合拟合一个运动模型
（而不是先配准再拟合），或者用上我们已经精确知道、但完全没进入配准的相机轨迹。

## 可交互程度：从声明值改成量出来的（2026-10-17）

### 问题

完整度报告的第二条轴原来只数程序**写了**什么：`interaction.grab` 不是 false 就算一个
能拖的物体。二十条合成片段上这一项是 1.000——程序说 238 个物体全都能拖。

声明和能用是两回事。审稿人问「这个数是量出来的还是写上去的」，原来的答案是后者。

### 做法

`harness/interaction_probe.mjs` 在无头物理里把每个物体真走一遍：抓住、提起来、横着
拖、松手、等它落稳。不开浏览器、不要 WebGL、不加载 GLB，一条片段几秒钟，所以每次
生成都能顺手量一次。四项判据——提得起来、跟得住、松手之后停得下来、不掉穿地面。

点不点得到单独报。从程序开场那台相机看过去，在物体投影出来的方框里打 5×5 一片射线，
数有多少条先打中它自己。只打中心一条会把「中心被挡、边上露出来」误判成够不着。

### 量出来的数（二十条合成片段，238 个物体）

| | 声明 | 实测 |
|---|---|---|
| 能拖的比例 | 1.000 | **0.655** |
| 点得到的物体 | — | 127 / 238 |
| 点得到的面积（中位数） | — | 0.10 |

79 个自由拖动的物体没通过，分布是：提不起来 28 个，提不起来且跟不上 38 个，跟不上
10 个，另外 3 个是松手之后停不下来或掉穿地面。

### 根因：生成的时候就插在一起

探针在抓住那一刻量了一次插入深度（抓住会把物体变成动力学刚体，Rapier 这时候才给它
和周围固定物算接触——两个 fixed 刚体之间的接触窄相是不算的）。

| | 插入深度中位数 | 插超过 1 cm 的比例 |
|---|---|---|
| 拖得动的 | 0.005 m | 0.44 |
| 拖不动的 | **0.032 m** | **0.73** |

插着的时候一抓，接触求解先把它往外顶，于是你往上拖、它往下走（`lift_progress` 是
负的那 38 个就是这样）。物理稳定那一步跑了，176 个物体里改了 163 个位姿，但 1.5 秒
显然不足以把互相咬住的解开，而且稳定之后那次接触装配是按地面贴合的，会把刚分开的
物体再推回去。这条链路要单独改，先把数量出来。

### 顺带修掉的两个内核缺陷

**鼠标能穿墙。** `Manipulator.pick` 原来是「这个命中抓不动就跳过、接着往后找」，
于是隔着墙、隔着桌面也能把后面的杯子抓出来。改成取第一个属于某个节点的命中：抓得动
就抓，抓不动就说明被挡住了。

**推不动旁边的东西。** 场景里除了手上那个，其余物体默认是焊死的——`addEntry` 里
视频中没观测到运动的物体 `dynamic=false`，建成 fixed 刚体。于是拿杯子去撞旁边那个
杯子，对方纹丝不动。`Manipulator.wakeContacts` 让**真的被碰到**的物体转成动力学刚体：
推到了才活，没碰过的不动，所以场景不会一加载就整体塌下来。静态结构和有脚本运动的
物体不碰。

### 样式-行为解耦有数了（#52）

同一串操作跑两遍：一遍用基本体，一遍把每个物体的外形换成一个环面结——形状、尺寸都
和原来南辕北辙。碰撞体永远从基本体那一套算（`kernel/scene.js` 的 `buildNode`），
所以轨迹应该一丝不差。

二十条片段、合计 **55920 步仿真，最大位置偏离 0 m**。

用合成的环面结而不是真 GLB 是刻意的：真 GLB 的外形和基本体接近，区分力反而弱；
形状差得越远，碰撞体一旦有一丝从网格漏出来就越藏不住。

### 一个教训

探针第一版的 `dragDistance` 是从 `group` 的包围盒取尺寸的，于是换外形之后拖动距离
跟着变，解耦对照立刻报出 1.6 cm 的偏离。内核是干净的，是**量具自己**把样式混进了
行为。尺寸改成从碰撞体取之后归零。量解耦的工具本身必须站在行为那一半。

## 让生成的场景在物理上说得通（2026-10-17）

上一节把可交互程度量出来之后，拖不动的根因很清楚：抓住那一刻就和别的东西插在一起。
当前链路的二十条片段上，拖不动的物体插入深度中位数 0.058 m，拖得动的 0.001 m，
差了将近六十倍；拖不动的里八成插超过 1 cm。这一节是针对它的两处改动。

### 一、重力稳定之前先只解穿插

重力解不开互相咬住的物体。它只有竖直方向的分量，两个物体横向插在一起的时候，重力
没有把它们分开的方向可用，跑多久都还是插着。而接触求解本身会沿着接触法线把重叠顶开，
所以先关掉重力单跑一段让它只做这件事，再开重力落到支撑面上。两件事混在一个仿真里
会互相抵消：顶开一点，重力又压回去。

`game.settle(seconds, separateSeconds)` 现在分两段跑。扫参用的是
`scripts/sweep_separation.py`，它从每条片段的 `program_before_settle.json` 分叉，
不重跑感知，所以几个配置之间唯一的差别就是这一刀。

当前链路产出的二十条（223 个物体）：

| 分离秒数 | 真拖得动 | 汇总成功率 | 插入深度中位数 | 插超 1 cm |
|---|---|---|---|---|
| 0（现状） | 172 | 0.771 | 0.004 m | 0.354 |
| 1.5 | **179** | **0.803** | **0.001 m** | **0.289** |

更早那批产物（238 个物体）上先扫过一轮，结论一致：0.655 → 0.706，0 / 0.5 / 1.5 / 3.0
四个值里 1.5 最好，3.0 反而略降。两批数据上都是 +3 到 +5 个百分点。

顺带否掉了一个猜测。一开始少跑了管线里「稳定之后再过一次接触装配」那一步，看到插超
1 cm 的比例掉到 0，于是怀疑是那次按地面的贴合把刚分开的物体又推回去了。补上这一步
重测，加上一组刻意跳过贴合的对照，结果是：贴合在开与不开两种条件下，分离都给出同样
量级的改善，而且少了贴合有四到五条片段过不了校验。贴合不是主因，它甚至略有帮助。

### 二、class 是地面的节点永远是固定结构

当前链路产出的二十条里还有两个 `class=ground` 的节点躺在 `objects[]` 里：一个
1.79 x 0.22 x 1.15 米，一个 1.94 x 0.39 x 1.53 米且带着 `motion: trajectory`——一块
一米九的地板沿着轨迹在飞。那是相机在动，而配准把相机的运动算到了地面头上。

「占画面太大的检测按结构件处理」那条拦不住它们：掩码只占画面 10% 上下，够不着 35%
的阈值。但地面这个类别本身有硬先验。它和 `is_static_class` 那一组的强度不一样：轨道
判出在动未必是错的（可能真是传送带），地面判出在动一定是错的。所以这一类不看运动
假设，直接按固定结构处理。

用完全相等而不是子串或分词匹配。子串会把 playground 一起吃掉；分词看着更聪明（能认出
「wooden floor」），但同样会把「floor lamp」判成地面。这条规则强度很高——它不看运动
假设直接改归属——所以宁可漏不可错。

### 端到端

同一批二十条片段，同一条链路，只差这两处改动：

| | 基线（main） | 两处改动之后 |
|---|---|---|
| 真拖得动的物体 | 170 / 223 | **182 / 221** |
| 实测可交互成功率 | 0.762 | **0.824** |
| 点得到的物体 | 129 | 132 |
| `class=ground` 躺在 objects[] | 2 | 0 |
| 最大遮挡者占被挡物体的比例 | 0.63 | 0.57 |
| 拖得动的接触点数中位数 | 3 | 1 |

一处退步：松手之后掉穿地面的从 0 个变成 1 个。分离会把深陷在地面里的物体往外顶，
顶出来之后可能落在没有支撑的位置。一个物体不足以判断是系统性的，记下来下次复测时盯着。

点得到的面积中位数从 0.399 降到 0.36，而点得到的物体数是涨的。两个数方向相反：
移走挡在前面的地板之后，原来完全点不到的物体露了出来，但只露出一小部分，拉低了
中位数。涨的那个是要的，降的那个说明露得还不够多。

### 为什么这两件事算在「可交互性」底下

因为点不到就拖不到，而挡住视线的往往就是这些东西。被挡住的物体里，62% 是被**同一个**
物体挡的（十三条有遮挡的片段的平均）。点不到不是场景太挤，是每条片段里有一两个本不该
在那儿、或者本不该那么大的东西横在最前面。

## 指标本身会骗人：这一轮踩到的四个口径问题（2026-10-18）

这一节记的不是某个模块怎么做的，是**量东西的时候踩到的坑**。四个都让我报出过错的结论，
而且每个都是「指标看着正常、但换个场景就不等价」。

### 一、带长度量纲的阈值不是尺度不变的

真值评测的物体配对容差 `max_dist_m` 默认 3 米，是绝对量。手持取景的场景只有 3.2 米
见方，容差是场景尺寸的 94%，等于「什么都能配上」；关卡尺度的场景 21 米，3 米是紧的。

我据此报出过「换取景约定之后召回 0.354 → 0.885」，同口径下真实值是 0.385 → 0.500，
而精确率根本没改善。`scripts/eval.py` 现在有 `--max-dist`，跨尺度比较必须按场景一起缩。

同一个坑在运动分类那边也有：`static_pos_thresh_m: 0.05` 和
`translation_min_displacement_m: 0.08` 都是绝对米制。

**做法**：任何涉及尺度变化的实验，动手前先 grep 一遍 `_m`、`_thresh`、`max_dist`，
把带长度量纲的常量列出来，逐个决定要不要跟着缩。

### 二、重渲染相似度几乎看不见颜色

把场景里每个物体的色相旋转半圈（每条片段改 11 到 22 个物体，形状、摆位、光照一字不动），
六条片段的相似度中位数只从 0.5635 变到 0.5568，其中两条反而涨了。

后果：样式那一半长期没有能验收的指标，而且我据此把「色彩迁移让牛顿摆上资产差距从
0.016 缩到 0.009」写进了文档——那 0.007 落在噪声里。已更正。

**做法**：用一个指标去验收某个维度之前，先测它对那个维度敏不敏感。测法很便宜——
把那个维度刻意改错，看指标掉多少。

### 三、相对比值在暗场景上会放大

颜色闭环报两个数：每通道绝对误差，和亮度比（渲出来的亮度除以视频里量到的）。
两条真实视频的终态：

| | 每通道误差 | 亮度比 | 材质最亮通道均值 |
|---|---|---|---|
| conveyor_boxes | 3.00 / 255 | 1.000 | 184 / 255 |
| newtons_cradle | **2.00 / 255** | 0.953 | 89 / 255 |

绝对误差更小的那条，亮度比更差——因为它是暗场景。我一度把 0.953 当成「没收敛好」，
去查粗糙度和高光，查完发现两条片段的材质参数完全一样。

**做法**：绝对量和相对量一起报，不要只看一个下结论。

### 四、样本小的时候「看起来越过阈值」和「真的越过」分不开

关节档那条 issue 的第一条建议是「两个估计器同时同意时更可信」。二十四条片段（七十八个
物体）上量到可用率 0.67，越过了「过半才值得开」那条线。扩到五十条（一百五十一个物体）
之后，**报出的个数纹丝不动，还是 3 个**——两者分别报出 21 和 17，随机重合的期望是 2.4。
一致性就在随机水平，那个 0.67 是三个样本里的噪声。

同一类问题这一轮出现了四次：0.885 的召回（11 条片段）、颜色拟合的默认轮数（6 条）、
关节合取判据（n=3）、逐短语检测的粗比（11 条对 7 条）。

**做法**：合取判据、交集、子集比较天然样本小，最容易踩。下结论前先问「这个数是几个
样本算出来的」，以及「扩一倍样本它会不会变」。

## 两份排除清单（2026-10-18）

这两份清单的价值和正面结果相当：它们说明现在的方法选择是排查出来的，也挡住了后来者
重走这些路。

### 过检（假阳性占预测的 83%）：八条

真阳性和假阳性之间，没有任何现有信号能分开：

| 信号 | 配上 | 没配上 | 结论 |
|---|---|---|---|
| 检测置信度 | 0.2998 | 0.3395 | 反相关 |
| 被看到的帧比例 | 0.4062 | 0.4062 | 完全相同 |
| 掩码占画面 | 0.0097 | 0.0047 | 两倍，重叠 |
| 资产检索成功率 | 0.69 | 0.51 | 1.35 倍，重叠 |
| 三维尺寸变异系数 | 0.105 | 0.066 | 方向反了 |
| 逐帧中心跳变 | 0.075 | 0.112 | 1.5 倍，完全重叠 |
| 图像检索最高分 | 0.540 | 0.549 | 几乎相同 |
| **逐短语检测**（换方法，非过滤） | — | — | 精确率 +0.002，耗时 +60% |

前七条是过滤，第八条是换检测的提问方式。最后一条最有说服力：换了方法过检纹丝不动。

**机制**：前六个信号都只描述检测自身，而真假之别在于它对不对应一个真实物体，那是
检测自身不含的信息。图像检索分是唯一来自外部参照的，它也失败说明**低多边形素材库的
视觉特异性不足以当裁判**——背景碎片和某个 Kenney 资产的相似度，跟真物体与它最像的
那个资产的相似度，是一样的。

### 运动类型判不出来：十二条

尺度、阈值、置信度定义、漂移沿视线、往复次数判据、静止占比判据、可见表面偏置、
点到面 ICP、模型累积帧数、用整个背景估公共漂移、相机位姿、抬高足迹长短边比的门槛
——全部排除，每条都有数（见 #7）。

**单一瓶颈是轨迹质量**：同一个分类器，在真值感知的好轨迹上置信度 0.640、准确率 0.277；
在神经感知的坏轨迹上 0.017、0.030。配准的参数空间已经扫完（点到点最好、累积六帧最好、
平滑窗口 15 最好），再往前要换方法不是换参数。

> **这里有两套「运动类型准确率」，不能混着比。** 上面这组（0.277 / 0.030）出自
> `scripts/probe_motion_signal.py`，它直接把轨迹喂给分类器，不经过程序合成、不做规范
> 对齐、不做物体配对，衡量的是「分类器在给定轨迹上能不能判对」。端到端评测
> （`scripts/eval.py`）那一组（0.137 / 0.182 / 0.158 / 0.200）要先把重建的物体和真值
> 物体配上对，判错和配不上是两回事，分母也不同。我曾经把探针的 0.277 和评测表的数
> 并排放进同一张表当作「全真那一档」——那是错的，两列不同口径。要比消融档位，四档必须
> 全部出自同一套评测、同一个容差。

#### 根因：ICP 在真值点云上无中生有地解出转动

把 `track_rigid` 单独拎出来量（`probe_motion_signal --mode icp`，真值深度 + 真值
掩码渲出的点云，不经过分类、配对、阈值），12 条片段：

| 真值运动 | 个数 | 真实转角范围 | ICP 解出的范围 | 角度 RMSE | 与真值的相关 |
|---|---|---|---|---|---|
| static | 5 | 0.000° | 19.85° | 12.53° | — |
| periodic_translate | 6 | 0.000° | 73.05° | 41.41° | — |
| prismatic | 3 | 0.000° | 39.09° | 20.20° | — |
| spin | 4 | 114.05° | 72.89° | 51.47° | **0.095** |
| periodic_rotate | 8 | 46.94° | 47.10° | 30.02° | 0.398 |

两个方向都错：**根本不转的物体被解出几十度转动**（纯平移那类最离谱，真值严格 0，
解出七十三度）；**真的在转的物体解出来的序列和真实的不相关**（spin 的相关系数
0.095）。`periodic_rotate` 那行范围对得很好（46.94 对 47.10）但 RMSE 30°、相关
0.398——**幅度对上是巧合，时间过程是错的**，只看范围会误判成差不多对。

这解释了之前所有没解释的观察：r2 分布为什么是反的（静止物体的角度序列是 ICP 自己
造的游走，游走容易碰上像正弦的一段）、#63 里真值静止的物体为什么报出 76.5°、
四档消融里换真值为什么不动、分类器为什么赢不过常数基线。

> 疑点：`revolute` 三个的**真值**转角范围量出来是 0.000°，不对。可能是 schedule
> 落在跟踪帧窗之外，也可能是生成器只绕支点公转不改自身朝向。n=3，没继续追，
> 那一行暂不作为证据；另外三类的真值 0° 是可信的，结论靠它们就够。

下一步怀疑**自遮挡**：单目看一个物体永远只看得见朝向相机的一半，两帧之间可见的那
半换了，ICP 会把「可见面换了」解成「物体转了」。这和两个错误方向都吻合。验法是拿
真值程序渲出的**完整**点云（直接从网格采样，不走深度图）重跑同一个 `track_rigid`。

#### 第一个修好的地方：把二分从七分类的闸底下救出来

十八条排除之后，第一个「有东西可以修」的点。两道闸
（`min_primary_confidence`、`min_hypothesis_margin`）是为**七分类**设的，这个取舍
本身对——分不清 spin 还是 periodic_rotate 时，往程序里塞一个猜错的运动模型，用户
一看就穿帮。**但二分陪葬了**：七分类过不了闸，连「它会动」也一起丢掉。

所以 `motion.py` 在两道闸之前单独算一个 `moves`，`_unknown` 那条路也带着走。
`type` 可以是 unknown 而 `moves` 是 True——知道它在动，不知道它怎么动。

端到端验过（十条真值感知片段、92 个物体、42 个真会动）：

| | 个数 |
|---|---|
| `moves=True` 真会动 | **37** |
| `moves=True` 其实不动 | 20 |
| `moves=False` 其实会动 | **3** |
| `moves=False` 真不动 | 29 |
| 说不准 | 3 |

**会动查全 0.19 → 0.925**，说会动的命中率 0.649（瞎猜 0.449，差 +0.200）。

> **代价**：二十个误报，每说三个会动的有一个其实不动。所以这一版只写进节点的
> `notes`，**不影响渲染**。要让它真正驱动交互，得先按用途分开定：用来**标注**
> 够了；用来**驱动物理**不够（五分之一的误动很明显）；用来**挑可交互物体**可能够。
> 下一步值得看误报是不是集中在足迹接近正方形的那些（朝向通道在那上面退化，
> yaw 跨度中位 73.5°）——如果是，一道只针对误报的过滤可能能在不太伤查全的前提下
> 把命中率提上去。

#### 最低的那级台阶也没上：分不清一个物体动没动

七分类贡献精确为零之后，退一步只看「会不会动」这个二分。**也不成立。**

| 产物 | 配对 | 真值会动 | 判会动 | 判对 | 说会动的命中率 | 瞎猜 | 差 | 会动查全 |
|---|---|---|---|---|---|---|---|---|
| 全真感知 | 26 | 22 | 6 | 4 | 0.667 | 0.846 | **−0.179** | 0.182 |
| 真值感知 | 26 | 22 | 5 | 3 | 0.600 | 0.846 | **−0.246** | 0.136 |
| 几何全真 | 25 | 21 | 6 | 5 | 0.833 | 0.840 | −0.007 | 0.238 |
| 当前主线 | 24 | 18 | 3 | 3 | 1.000 | 0.750 | +0.250 | 0.167 |

八个配置六个为负——说一个物体「会动」比闭眼瞎猜「会动」还不准。两个正的都是
主线，命中率 1.000 但**分子是 3**：从七成五都在动的池子里摸三个全中，概率 0.75³
= 0.42，不是证据。查全只有 0.11–0.24：二十来个会动的，链路只说出来三到五个。

> **这一列原来是按「二分准确率 vs 恒答不动」报的，八个配置差全为正，看着像有信号。**
> 那是类别不平衡造出来的：会动的占七八成，基线本来就低，乱猜「会动」也能蹭到。
> 要看的是 P(真会动 | 判会动) 减 P(真会动)。这是这一轮第六个同类的量具问题，
> 而六个的方向全都是让结论看起来比实际干净。

**后果**：可交互性四档里，第一档（观测运动）要的就是这个二分，它不成立；
第三档（关节）从来没产出过（#63）；第二档（直接操纵）不需要理解场景。
所以**现在真正立得住的只有第二档**——而那正是审稿人会问「和把每个检测框变成
可拖动盒子的基线有什么区别」的那一档。

#### 更硬的那句话：它一次都没判对过任何一个会动的物体

把「碰到转动一律说 unknown」在七份评测产物上重算（`scripts/probe_motion_baseline.py`，
只读已有混淆矩阵，不重跑）：压掉转动预测之后，**每一个配置的准确率都精确等于常数
基线，差全是 +0.000**。只要有任何一次平移类判对了，这个差就会是正的。

**161 个配对物体、七个配置，运动分类一次都没有正确识别出任何一个会动的物体。**
它拿到的每一分都来自「把静止的叫静止」，每一次非 static 的答案都是错的
（全真那档的 −0.095 就是把两个真静止的判成了 spin）。

而瓶颈不在分类、也不在转动信号，在**更上游**：会动的物体根本没进到分类器面前。
真值里位移最大的两类（`trajectory` 4.9 m、`periodic_translate` 4.2 m）在混淆矩阵里
一个都没出现——它们没被配上对，因为 `match_objects` 的代价是整段逐时刻距离的平均，
而我们把它们写成了 static。**这是个闭环**：运动判错 → 配不上 → 不进分母 →
分母里只剩不怎么动的 → 「恒答 static」成为最优解。

所以 `evaluate` 加了 `match_on="start"`：按观察窗起点的位置配，和运动无关。
物体是不是同一个，和我们有没有把它的运动猜对，本来是两件事。

#### 先看这一行：六个配置没有一个赢得过「恒答 static」

运动类型准确率要和**常数基线**并排看，否则读不出区分力。基线的定义很简单：
一个「不管看到什么都答 static」的程序能得多少分，也就是配对物体里真值本来就是
static 的比例。

| 档位 | 配对物体 | 运动类型准确率 | 恒答 static | 差 |
|---|---|---|---|---|
| 当前主线 | 24 | 0.250 | 0.250 | **+0.000** |
| 全神经 | 22 | 0.182 | 0.182 | **+0.000** |
| 只有相机真值 | 19 | 0.158 | 0.158 | **+0.000** |
| 几何全真 | 25 | 0.200 | 0.160 | +0.040 |
| 真值感知 | 26 | 0.115 | 0.154 | **−0.038** |
| 全真 | 21 | 0.095 | 0.190 | **−0.095** |

**没有一个配置赢得过常数基线。** 三个精确相等，两个比基线还差，一个 +0.040
（n=25，在噪声里）。当前主线那一档的混淆矩阵最直白：24 个配对物体判对 6 个，
**六个全是 `static->static`**；剩下三次给出非 static 的答案，三次全错。

所以「运动类型准确率 0.250」不是「准确率偏低」，是**这个指标此刻对我们的方法没有
任何区分力**。这一列已经做进 `gwm/feedback/report.py` 的表里，不靠人想起来去算。

#### 四档分解：感知整条都不是瓶颈

同一套评测、同一容差（`--max-dist 0.45`）把感知逐层换成真值：

| 档位 | 片段 | 配对物体 | 运动类型准确率 | 标准误 | 物体召回 | 物体精确率 |
|---|---|---|---|---|---|---|
| 全神经 | 10 | 22 | 0.182 | 0.082 | 0.429 | 0.211 |
| 只有相机是真值 | 8 | 19 | 0.158 | 0.084 | 0.464 | 0.194 |
| 几何全真（相机+深度） | 10 | 25 | 0.200 | 0.080 | 0.500 | 0.258 |
| 全真（再加真值掩码） | 9 | 21 | 0.095 | 0.064 | 0.571 | 0.333 |

检测随真值进来单调变好（召回 0.429 → 0.571），运动类型纹丝不动，全真那档甚至最低。
全真那档的混淆矩阵里，**21 个配对物体有 16 个（76%）被写成 static**。给了完美的
相机、深度、掩码，链路还是把四分之三的东西写成不动的。

#### 不是「运动本身太小」，除了两类

配对物体的轨迹误差中位 0.138 m，真值里会动的 38 个物体整段位移中位 **2.549 m**，
信噪比 18 倍。但按类别拆开，`periodic_rotate` 只有 0.307 m、`revolute` 只有
0.185 m——贴着噪声底，而它们正是塌成 static 的主力。理由是几何的：**绕自身轴转的
物体中心几乎不动**，靠中心轨迹原理上看不见它，必须看朝向。

（顺带量出另一件事：位移最大的两类 `trajectory` 4.9 m 和 `periodic_translate`
4.2 m 在混淆矩阵里一个都没出现——它们根本没被配上对。动得最多的物体是**配不上**
的那些，不是判错的那些。那是物体配对那一侧的问题。）

#### 朝向通道的退化：相关系数 −0.906

`obb_from_points` 的 yaw 是 xz 点云 PCA 的主轴。拿**真值静止**的物体量它的噪声底
（静止物体朝向本该恒定）：

| 底面长短边比 | 个数 | yaw 跨度中位 | 最大 |
|---|---|---|---|
| 1.2 – 1.5（接近方） | 6 | 73.5° | 252.4° |
| 1.5 – 2.5 | 9 | 34.1° | 260.7° |
| 2.5 以上（细长） | 34 | 5.1° | 17.5° |

log(长短边比) 与 log(yaw 跨度) 的相关系数 −0.906，是 PCA 主轴退化的典型签名。

> **范围要说清楚：这个退化影响的不是主路径。** 运动估计主路径用的朝向是
> `_registered_quats(q_a[0])`——配准解出的朝向接在第一帧包围盒朝向上，PCA 只以
> 常数偏移的形式进来，造不出假转动。当前二十条产出里 153 个物体走配准、38 个走
> 包围盒中心，所以 PCA 乱转只影响那两成，外加任何直接读 `obb[].quat` 的下游。
> `continuous_footprint_yaw` 修的是后者，对端到端运动类型准确率的提升预计很小。
>
> #63 里那个「真值静止的物体报出 76.5 度转角」，追下去是 `track_rigid` 的累积旋转，
> 也就是**配准解出了不存在的旋转**，不是 PCA。我一开始归错了，已公开更正。这条线索
> 比 PCA 那条重，因为八成物体走配准——而且它和四档分解自洽：真值点云上仍然解出假
> 旋转，说明问题在配准这一步本身，不在输入质量。对称点云上 ICP 的旋转本来就欠定
> （正方体绕中心转 90 度和没转，点云完全重合），所以 `orientation_reliable` 用足迹
> 比去闸配准朝向，道理上是通的。

**想当然的修法（把 `yaw_min_aspect_ratio` 从 1.3 抬到 2.5）不行**：十四个真会转的
物体会被挡掉九个，因为真转动的足迹比也集中在 1.5–2.5，和噪声最大的静止物体是同一个
带。幅度同样分不开，两边的 yaw 跨度完全重叠。这是第十二条排除。

实际做的是**消掉主轴的规范自由度**（`continuous_footprint_yaw`）：特征向量的符号
（180 度）在任何形状上都任意，一律消；长短轴互换（90 度）只在足迹比低于 1.6 时消，
否则会把细长物体真实的直角转动抹掉。

还有一条原理上的限制：足迹主轴只定义到 180 度，长方形转过半圈和没转是同一个足迹，
所以转超过半圈的 `spin` 靠足迹 yaw 读不出来，要换信号。

### 配准轨迹的时间平滑：把最后一处退步也补上了

配准换掉包围盒中心之后，有一项一直是退步的：真值静止物体的漂移 0.231 m，比包围盒中心的
0.170 m 还大。原因是配准的**每步**噪声是物体尺寸的 17%，而包围盒中心那条路本来就过一道
时间平滑（`smooth_series`），配准这条之前漏了。

补上之后（二十五条有标准答案的片段，掩码与深度都是真值）：

| 信号 | 相对位移误差中位数 | 真值静止物体漂移 |
|---|---|---|
| 包围盒中心（原来的） | 0.216 m | 0.170 m |
| 配准，不平滑 | 0.173 m | **0.231 m** |
| 配准 + 窗口 5 | 0.152 m | 0.148 m |
| 配准 + 窗口 9 | **0.140 m** | 0.115 m |
| **配准 + 窗口 15** | **0.140 m** | **0.084 m** |

窗口 9 和 15 的轨迹误差一样，但 15 把静止漂移压到 0.084——**比包围盒中心好一倍**，所以
取 15。至此配准在两项上都优于原来的信号：轨迹误差好 35%，静止漂移好一倍。

这条之所以一直没被发现，是因为我换信号的时候只替换了「喂给运动估计的位置」，没注意到
旧路径上还挂着一道平滑。替换一条链路的中间环节时，要连它周围的处理一起看。

### 评测来源与实测分母

- `similarity` 的尺度提示和对象配对一样，只能使用观察前段的源样本。
  相机在两侧共同时间支撑内按相同时刻插值；不借留出帧插值，不外推。
  缺少两个有效相机样本、共同时间跨度或有效运动时，保留单位尺度及不可估原因。
  默认 `rigid` 不变；相似对齐仍是诊断口径，不代表真实米制尺度恢复正确。
- 运行实例名称用于独立配对，`track.object_id` 用于回查父 Program 的类别、尺寸及形状。
  `objects.geometry_per_object` 保存每对的数值或 `missing_geometry`，`geometry_n` 只数实测项。
  无几何的匹配不算形状成功或失败，不拿 `matched` 代替测量数。
- 汇总的 `scale_ratio` 从逐对象原始比值计算；旧报告仅存片段中位数时，
  放在单独的 `scale_ratio_run_medians`，不能伪装成逐对象统计。
  缺测量数的旧形状/尺度比例不进入对象汇总，并列于 `geometry_unavailable_runs`；
  有明确 `geometry_n` 的历史摘要可恢复二值通过率，但不能恢复逐对象尺度中位数。
- 直接翻译保留旧 Evidence 已明确的运动类型。`moves=True` 或明确的非静态类型
  都阻止仅凭结构类名将对象静止化；缺少 `moves`、甚至与它冲突的 `False`，不抹掉明确运动。
  原有地面和大面积结构规则不在此变更范围内。未知运动并不因此获得新的运动结论。

以上是离线正确性约束，不宣称提升真实视频重建质量；旧实验数值需按固定输入重新评测才能比较。

## 两个创新点接上了（2026-10-23/24）

> **先读这条限定（2026-10-24 补）。** 下面整节的数都是在**真值掩码 + 真值深度**
> 上测的。换成真实 VGGT 几何之后，残差挑资产从 0.297 掉到 0.041（随机 0.030），
> 正确网格当模板也不再领先基线。详见本文末「真实几何把表面级判据吃掉了」。
> 这一节保留，因为它说明**在干净几何下这条链是成立的**——约束在几何，不在链本身。


这一轮之前，两个创新点是并列的两件事：一边是「怎么把稠密几何变成对离散程序选择的
监督」，一边是「会动的物体上位姿解不出来」。量完之后它们是一条链，而且每一环都有数。

### 稠密信号确实能定位资产，只是管线从来没问过它

把视频里物体的裁剪图嵌入，在全库 4812 个资产里按图像相似度排序，本片段真值资产的
中位名次是 **365**（四分位 105 / 1064，最好第 1），进前 32 名的比例 0.176；换成
别片段真值集合的大小匹配零假设是 **1117 / 0.031**。对照之下，管线按文字选中的那个
在图像分上排第 **860**——比正确答案还靠后。信号在，只是没被问过。

顺带更正一条之前引用过的数：「拿真值类名当查询词 top-1 0.689」**是循环的**。真值
场景用资产库搭之后，类名由资产文件名拆词得来，那等于拿资产自己的名字查自己。
文字那条路的上限不能用这个数。

管线侧有两处阻塞，都修了。一是图像塞进来的候选被接在文字命中表的**尾巴**后面，
而融合重排只动前 24 个——122 条决策里只有 8 条候选表真变了，一条都没改选中，
那个 A/B 什么都没测到。二是收录判据只看文字分，而图像塞进来的候选文字分恒为 0，
必然被 0.45 的阈值挡掉。两处一起改之后稠密信号第一次真的参与定夺（7/122 条决策
改由图像候选胜出），形状距离从 0.0719 降到 0.0667。

但精确命中率仍是 0/71。原因在数里：走图像门的 22 条名次中位 62，而正确资产的名次
中位是 365，门卡在前 32 或 128 只捞得到尾巴。**瓶颈不是信号，是「怎么从前几百名里
挑出那一个」。**

一条设计教训：只开候选不配收录判据是**净倒退**。admit256 比不开少选中 7 个资产——
图像候选挤掉了文字候选的第一名，然后自己因为文字分 0 被挡掉，退回基本体。

### 配准残差能挑对资产，这是创新点一的字面验证

把「提候选」和「在候选里挑」拆开。候选 = 真值资产 + N 个干扰项，问**配准残差**
（把候选网格缩到观测点云的逐帧中位对角线上，ICP 配一次）能不能把真值挑出来。
不用类名取候选，因为那是循环的。

assetgt60（60 条片段，74 个带真值 ref 的物体），按样本加权：

| 候选集 | 挑对比例 | 随机猜 | 倍数 |
|---|---|---|---|
| 真值 + 8 个随机干扰 | 0.459 | 0.111 | 4.1× |
| 真值 + 32 个随机干扰 | 0.284 | 0.030 | 9.4× |
| **真值 + 32 个形状最接近的干扰** | **0.243** | 0.030 | **8.0×** |

（上表是**单帧**版。后来把用几帧扫了一遍：1 / 3 / 7 / 15 / 全部帧 =
0.149 / 0.243 / 0.297 / 0.284 / 0.311，**平台在 7 帧**，静止物体上平台
**0.625**。另外换一帧差 63%——只用首帧 0.149、只用中间帧 0.243，首帧的可见面
最偏。再往上两条杠杆都试完了：加初始偏航反而更差，见下。）

第三行是密度匹配的零假设——随机资产长得和目标完全不像，而真实方法面对的候选集
个个都像。换成难干扰项之后只从 0.284 掉到 0.243，**说明残差不是靠排除「长得完全
不像的」在赢，它真的能定位到具体那一个。** 真值静止的物体上，难干扰项 33 选 1
挑对 0.500（16.5 倍）。

这就是创新点一的字面实现：连续稠密几何监督离散的符号选择。梯度传不过去不要紧，
「哪个网格配得上这团点」这个判据不需要梯度。

分桶有个规律：静止 0.500、prismatic 0.500、revolute 0.286，而 periodic_rotate
是 **0.000**、spin 0.125。**选择机制在物体不转时很强，一转就垮**——和下面那条线
指向同一个根因。

### 挑对资产有真实的下游价值，而且评价不循环

「数据集评价不了资产检索」那个死结是：收益指标循环（重排正是在优化它），代价指标
是噪声。位姿给出了第三条路——**选对资产 → 位姿解得准**，这个信号不循环。

assetgt60 配对口径，**同一批 8 个物体**（真值静止、带资产 ref），只换模板来源：

| 臂 | range | 四分位 | RMSE |
|---|---|---|---|
| 基线（局部互配） | 25.28° | 14.51 / 53.87 | 12.85° |
| 按观测包围盒搭的盒子 | 23.86° | 11.37 / 30.72 | — |
| **真值资产网格** | **9.25°** | **5.95 / 12.18** | **6.41°** |

gtref 的上四分位低于基线的下四分位。但真正要看的是**判别比**（静止的假转动 对
spin 解出的转动，同一批物体）：

| 臂 | 静止 | spin | 比值 |
|---|---|---|---|
| 基线 | 25.28° | 65.63° | 2.60 |
| 按包围盒搭的盒子 | 23.86° | 23.78° | **1.00** |
| **真值资产网格** | 9.25° | 39.89° | **4.31** |

**盒子模板没用的原因是它把静止和转动压成了一样**——假转动降一点，真转动降同样多。
换成正确的网格，判别比升 66%。所以「检索挑对」不是锦上添花，它是第三、四档
（关节、玩法）能不能走的前提：那两档卡的正是「动没动」「关节真不真」，靠的就是
幅度能不能分开。

**边界要写清楚：时间过程不改善。** 转动物体的角度 RMSE——spin 基线 44.39 / gtref
41.32 / 盒子 39.01，periodic_rotate 基线 27.52 / gtref 30.39 / 盒子 31.89——谁都
不赢。收益只在「静止物体的假转动」和「两类分不分得开」这两件事上。

口径提醒：gtref 只覆盖带资产 ref 的物体（静止桶 67 个里只有 8 个，其余是结构节点，
真值用的是基本体）。**别拿 gtref 的 9.25° 和全体 67 个的基线 19.62° 比。**

### 模板路线那条否定要收窄

之前记的是「模板配准在真实数据上不成立，两条能落地的模板来源都比现状差六倍」。
查下来那组数里有一半是实现 bug：模板按**所有帧并起来**的点云搭，而那团点云随视角
扫动——实测并帧对角线是逐帧中位对角线的 **1.890 倍**。而受控扫描把容差线扫出来了，
模板相对它要配的那片点云：

    长方体  x0.7: 5.0°  x0.8: 4.9°  x0.9: 1.9°  x1.0: 5.2°  x1.1: 9.7°
            x1.2: 24.2°  x1.3: 88.5°                        （对照 12.5°）
    L 形    x0.7: 5.2°  x0.8: 4.8°  x0.9: 3.7°  x1.0: 4.0°  x1.1: 4.6°
            x1.2:  5.7°  x1.3: 15.4°                        （对照  7.6°）

**悬崖在 x1.1 到 x1.3 之间，而且是单边的：偏小完全安全甚至更好，偏大才崩。**
偏小的模板整个嵌在点云里，对应关系仍然唯一；偏大的闭合盒子让局部点云沿着面滑。
1.89 倍正落在悬崖外面。改成逐帧中位之后，拟合长方体的角度 RMSE 从 81.07° 掉到
12.42°，资产网格从 80.92° 掉到 32.54°。

所以那条否定针对的是**盒子模板和错网格模板**，不是模板这条路。

受控实验还给出模板的两条要求：**尺寸和比例要准**（正方体那一档退回对照水平），
**不能转动对称**（球和圆柱比不用模板还糟，没有偏航参考）。但要注意受控实验里
物体本身就是长方体，盒子模板在构造上就是对的；换成 L 形、T 形之后，T 形上包围盒
盒子反而从 5.9° 涨到 21.7°——它的立鳍在正中间，包围盒里大部分是空的。

### 这一轮关掉和没关掉的

**关掉的**：体素降采样（26.6°→27.9°，更差）；运动模型直接拟合（静止物体从运动模型
得到的残差下降 0.532，比真会动的 0.336~0.365 还高，倒挂）；按包围盒搭盒子当模板
（判别比 1.00）。

**没关掉但要重测的**：相机角步长判别子。实测 static -0.034、periodic_rotate +0.319，
方向是反的，看起来是个干净的否定——但同时报的 `cam_step_spread`（每步相机转角的
四分位距 / 中位数）static 只有 **0.036**，也就是角步长近似常数，相关系数的分母趋零，
算出来是噪声。合成探针里为此特意给相机加了非匀速，真实数据的相机档没有这个性质。
**这条路没被关闭，只是量错了。** 换成「假转动幅度 ↔ 相机总角位移」重测：
`rot_per_cam_deg` 中位 0.630、四分位 0.329/1.024（散开 3.1 倍），而 `cam_total_deg`
四分位 12.52/88.92（散开 7.1 倍，预测变量够散，不退化）。结论是相机角位移**解释了
一部分但不是确定性的**，不能直接扣除。

**猜错并验掉的**：以为盒子模板把真转动压没是因为长方体自带 90° 转动对称、把 114°
混叠成 24°，顺带怀疑合成实验「真转 90° 保住 89.7°」是撞上对称周期的巧合。加一档
50°（不是任何对称元的整数倍）重验，解出 48.1°（长方体）/ 44.9°（L）/ 52.9°（T），
照样跟得上。**对称混叠这个解释不成立**，合成那一行也不是巧合。真实数据上塌陷的
原因仍然未知。

### 两条杠杆都探到头了

**视角数**：1 / 3 / 7 / 15 / 全部帧 = 0.149 / 0.243 / 0.297 / 0.284 / 0.311。
7 帧之后不动。机制清楚——一帧只是一个局部视角，同一包围盒比例下完全不同的物体
在单个视角上可能都配得差不多，多视角把它们分开；但七个视角之后信息就饱和了。

**初始偏航**：猜瓶颈是「ICP 只从质心对齐起手一次、停在局部极小」，绕竖轴加
4 / 8 个初值每帧取最小残差。**反而更差**：0.297 → 0.216 → 0.216。

原因是选择统计上的偏差：取最小 = 给每个候选更大的搜索预算，而**错的候选从中
得益更多**——正确的那个本来就配得上，错的那个多了几次撞运气的机会。
可以推广：**在「挑哪个」这类任务里，「多试几个初值取最好」会系统性偏袒错的
候选**；候选之间的分必须可比。视角数那条之所以有效，是因为它给所有候选同样多的
**证据**，不是更大的**搜索预算**。

`periodic_rotate` 在所有视角数、所有偏航数下都是 **0.000**，两条杠杆都救不了。
转动物体上挑资产本身就要先解出位姿，而那正是点二没解决的那件事——这条链在
转动物体上是断的，所以落地顺序由数据定死了：**先在静止物体上兑现点一**
（那里平台 0.625），再用挑对的网格去啃转动的。

### 下一步

缺的那一环是「怎么把 365 名变成 33 名」。残差挑资产在 33 个里成立（8.0 倍），
而图像嵌入把正确答案放在中位 365 名——中间差一个数量级。两条路：把图像候选的前
几百个全过一遍残差（代价线性），或者先用一个更便宜的判据（边长比例、kit）把几百
收到几十再过残差。前者可以直接量，先量它。


## 真实几何把表面级判据吃掉了（2026-10-24）

上一节所有的数都是真值掩码 + 真值深度。换成**真实 VGGT 几何**（真值分割保住物体
身份，只换深度；assetgt60，60 条片段）重跑：

### 残差挑资产基本死了

| 候选集 | 全真值 | **真实几何** | 随机猜 |
|---|---|---|---|
| 33 | 0.297 | **0.041** | 0.030 |
| 129 | 0.203 | ~0.014 | 0.0078 |
| 385 | 0.162 | ~0.000 | 0.0026 |

33 个候选时只有静止物体 0.250、prismatic 0.125 非零，其余全桶 0.000。

**这完整解释了端到端的 0/78。** 不是管线接错了，是判据本身在真实深度上失效。
之前为这个 0 想过两种解释——候选集里没有正确答案，或者挑不中——答案是第二种，
而且不是「挑不准」，是「完全没有信号」。

### 位姿那一侧也大幅收窄

静止物体，配对口径（n=8）：

| 臂 | range | RMSE |
|---|---|---|
| 基线 | 30.72° | 19.65° |
| 正确网格 | 31.76° | **38.63°** |
| 逐帧包围盒盒子 | 65.52° | 48.19° |
| 真值位姿模板（预言机） | 15.07° | 8.34° |

正确网格在绝对假转动上**不再领先**，RMSE 还差一倍。判别比（静止对 spin）从基线
1.62 抬到 2.41，相对增益还在（+49%），但远不是全真值上的 2.60 → 4.31。

**预言机模板仍然有效**（15.07 / 8.34，而基线在真实几何下是 35.19 / 19.30）。
所以「完整性」这个论点本身不依赖干净深度；垮掉的是**从候选网格里挑对**和
**用近似网格配准**这两件事。

### 候选集没问题，失效全在「挑」这一步

另一处口径也查了。之前报的「正确资产中位第 365 名」用的是**集合判据**（片段里任何
一个真值资产排得靠前就算），放水的上界。按**这个物体自己的**真值资产重量：

| 真值类别 | 自己那个资产的名次中位 | 前 256 名覆盖率 |
|---|---|---|
| **static** | **36** | **0.750**（前 384 名 0.875） |
| spin | 104.5 | 0.625 |
| periodic_rotate | 153 | 0.714 |
| prismatic | 328.5 | 0.500 |
| trajectory | 512 | 0.444 |
| dynamic | 1148 | 0.200 |
| revolute | 1325 | 0.286 |
| periodic_translate | 2445.5 | 0.167 |

加权合计前 256 名覆盖率 **0.419**。账因此能对上：真实几何下
0.419 × 0.041 ≈ 0.017，期望 1.3/78，实测 0，在泊松噪声内。干净几何下会是
0.419 × 0.203 ≈ 0.085，约 6.6/78。

**提候选这一侧没问题，静止物体上还相当好**（名次中位 36、前 384 名覆盖 0.875）。
瓶颈很干净地落在一处：残差判据在真实深度下没信号。

### 读法

残差判据和模板配准共同依赖的是**表面细节**，而深度噪声吃掉的正是这个。
所以两个创新点的共同约束是**几何质量**，不是检索、不是模板来源。

方法论上这是 [[synthetic-gains-need-a-real-data-leg]] 升一级：**真值感知也是一种
「合成」**，它的结论同样要配一条真实感知的腿。今天连着两次用口径宽松的数去算
口径严格的预期（模板容差对 scale_ok、集合判据的 365 对端到端覆盖率），都是预测
落空之后才回头发现的——预测—对照这个习惯把口径错误变成了可见的。

### 分解：相机不是主因，换判据也不行

**相机。** 33 个候选的挑对比例：全真值（真值相机 + 真值深度）0.297 ／
只换相机（真值相机 + 神经深度）**0.054** ／ gtseg（神经相机 + 神经深度）0.041 ／
随机 0.030。把相机位姿换成真值几乎没救回来。之前那个推断——「点云是好的只是被
摆错位置，坏的是配准条件数，而条件数最直接的杀手是相机位姿」——不成立。

**判据。** 试了 D2 形状分布（随机点对的距离直方图，全局统计量，逐点噪声会被几千
个点对平均掉），协议和残差完全对齐，唯一变量是判据：

| 判据 | 挑对（33 候选） | 随机 |
|---|---|---|
| D2 形状分布 | 0.027 | 0.030 |
| ICP 残差 | 0.027 | 0.030 |

完全打平，都在随机水平。只有静止物体保住一点信号（129 候选时两者都是 0.125，
16.1 倍）。

**这个结果否掉的比预想的多。** 原假设是「残差对逐点噪声敏感，换成全局统计量就
好了」。把抽象层次从逐点表面提到点对分布**一点没有改善**——说明 VGGT 的深度在
这些物体上**根本没携带足够的形状信息**，不是「有信息但被噪声盖住了」。

### 下一步变了

不是继续抬检索命中率。上面两条分解把路也关掉了两条——不是修相机（只换相机
0.041→0.054），也不是换鲁棒判据（D2 和残差打平，都在随机）。剩下的两条：

一、**换几何后端**。VGGT 的深度在这批物体上没携带足够的形状信息。

二、~~换数据分布~~ **验过了，也不是。** 按掩码像素面积分桶（真实深度，33 个候选，
随机 0.030）：中 200~600px 挑对 0.033（n=30），大 >600px 挑对 0.045（n=44，面积
中位 1264px）。大物体一点没好。

### 根因：压扁 + 缩小，不是噪声

**噪声不是原因。** 真值深度 + 各向同性高斯噪声（按包围盒对角线的比例，n=74）：

| 噪声 | 0 | 0.5% | 1% | 2% | 5% | **真实 VGGT** | 随机 |
|---|---|---|---|---|---|---|---|
| 挑对 | 0.297 | 0.230 | 0.216 | 0.176 | **0.108** | **0.041** | 0.030 |

连 5% 噪声都只掉到 0.108，仍是真实 VGGT 的 2.6 倍。衰减对噪声平缓，真实情况掉得
狠得多——误差的**性质**不同，不是量的问题。「等更准的深度模型」这个默认假设
很可能不管用。

**结构性畸变，两种叠加**（static 桶 n=67）：

| 量 | 中位 | 四分位 |
|---|---|---|
| 最短轴的比例（该是 1.0） | **0.526** | 0.311 / 0.779 |
| 中间轴的比例 | 0.924 | 0.577 / 1.062 |
| 最长边的绝对倍数 | **0.420** | 0.226 / 0.740 |

一、**沿最短轴被压扁到正确比例的一半**，中间轴基本对，各桶一致（0.50~0.76）。
这是沿视线方向分辨率不足的典型表现。
二、**尺度整体偏小两倍半**，最长边只到真值的 42%。

第一条直接解释了所有表面级判据失效：**点云的形状本身就不是那个资产的形状**，
不是抖动过的版本。残差、D2、模板配准都在比形状，而形状是错的——跟噪声大小无关，
所以 5% 噪声复现不出来也就顺理成章。

### 第十步：轮廓判据，方向对但量级不够

根因是深度误差**沿相机视线**，而轮廓是二维的、沿视线的误差投影为零——这是唯一
从原理上绕开根因的判据。用资产缩略图的轮廓（Hu 矩 + 紧实度 + 长宽比）对观测掩码：

| 判据（真实几何，33 候选，n=74） | 挑对 | 随机 | 倍数 |
|---|---|---|---|
| **轮廓（二维）** | **0.081** | 0.030 | **2.7×** |
| ICP 残差（三维表面） | 0.041 | 0.030 | 1.4× |
| 干净几何下的残差 | 0.297 | 0.030 | 9.8× |

**方向对、量级不够。** 轮廓比深度判据强一倍，印证了机制；但绝对水平离可用差得远，
候选加到 129 就掉回 1.8 倍。原因：缩略图只有 4 个固定视角、有些还紧贴裁切
（实测一张前景占 96%），Hu 矩比任意姿态的单视角轮廓本来就粗。要做实得**按观测
位姿渲染候选的轮廓**。

#### 这一步差点变成最大的假阳性

第一版给出 **74/74 全对**。缩略图的 alpha 全是 255（有 alpha 通道不等于用了它），
取出来的「轮廓」是整个 160×160 方块，每个资产的 Hu 矩一模一样、距离全相等；
而 `if d < best_d` 严格小于保留先遍历到的那个，候选列表是 `[gt_ref] + picked`
——真值永远排首位，每次平局都「赢」。

三个自检信号一致地抓出来了：候选数 = 33（不是被跳过）、**margin 恒为 0.000**、
错配零假设正确失败，改成排序破平局后挑对率掉到 0.000~0.125。

**任何「挑最小」的实现都要报第一名对第二名的差。** 差为 0 说明判据没有分辨力，
名次完全由遍历顺序决定——而正确答案往往因为写法方便被放在首位。

### 第十一步：按观测位姿渲染轮廓，不碰深度的判据最好

把视角对上（用这一帧的相机朝向渲染候选，正交投影，按二维包围盒归一化比 IoU）：

| 判据（真实几何，33 候选，n=74） | 挑对 | 倍数 |
|---|---|---|
| ICP 残差（三维表面，**用深度**） | 0.041 | 1.4× |
| 缩略图轮廓（4 固定视角 + Hu 矩） | 0.081 | 2.7× |
| **按观测位姿渲染 + IoU** | **0.108** | **3.6×** |
| （对照）干净几何下的残差 | 0.297 | 9.8× |

自检过：n_scored 全是 33，margin 0.008~0.021（非零，不是平局假象）。

**机制坐实：不碰深度的判据比用深度的强 2.6 倍。** 正交投影是关键——透视需要知道
物体离相机多远，而那个距离正是深度给的、深度是坏的。这是第一次在真实几何上让
创新点一的判据明显离开随机。

缺口也清楚：margin 只有 1~2% 的 IoU 差，信号很薄，离干净几何的 0.297 还差一半
以上。

**两条可做的：** 一、创新点一的稠密监督信号**该走二维**（掩码/轮廓），不是三维
深度点云，沿这条继续加精度（更多偏航/俯仰、真正的光栅化而不是点撒）。
二、**创新点二有一条没试过的路**——今天所有位姿实验都建在深度点云的 ICP 上，
而如果二维轮廓携带更多可用形状信息，位姿也该换成**轮廓对齐**。

### 第十二步：同一套信号解不了位姿

顺着「不碰深度」把位姿也试了（点二今天所有实验都建在深度点云的 ICP 上，这条一次
没碰过）：每帧扫 36 个偏航，取和观测掩码 IoU 最高的那个，用真值资产网格。
**饱和式失败：**

| | 解出的转角范围 | 真值 | 对照（深度 ICP） |
|---|---|---|---|
| static | **170.0°**（四分位 150/180） | 0° | 30.72° |
| spin | **180.0°** | 95.7° | 49.80° |
| periodic_rotate | **180.0°** | 65.6° | 67.33° |

偏航在整个圆周上乱跳。原因和判据那一侧的缺口**是同一个**：margin 只有 1~2% 的
IoU 差，IoU 对偏航的曲面几乎是平的，argmax 就是噪声。

**挑候选够用（候选之间差别大），解位姿不够用**——后者要的是精细的角度分辨。

**一处限制：** 这一版为了不碰深度把轮廓按二维包围盒归一化了，而这一步恰好扔掉了
位置和尺度信息，真正的渲染会保留。所以关闭的是**这个廉价版本**，不是「轮廓解
位姿」这个想法本身。

### 第十三步：分辨率不是限制因素，前面的结论站得住

怀疑「信号薄」是光栅化造成的假象：格子 48×48 = 2304，而网格点 3000 个，平均每格
1.3 个点等于把包围框填实。提到 96×96、采样点 3000 → 20000：

| | 48 格 / 3000 点 | 96 格 / 20000 点 |
|---|---|---|
| 挑候选 margin | 0.008~0.021 | 0.008~0.027 |
| 静止物体解出的位姿 | 170° | 165° |
| spin 解出的位姿 | 180° | 180° |
| 挑对率（static / spin） | 0.250 / 0.125 | 0.250 / 0.125 |

**四倍格子面积、近七倍采样点，两个指标都没动。** 轮廓信号本身薄是真的。

这一轮验证的是前面的结论而不是推翻它——当天前六次回头验有四次推翻了我的说法，
这次没有。

### 第十四步：这套方法的真实形态

此前所有「挑资产」的实验，候选集都是人造的「真值 + 形状最近邻」，而管线真正面对
的是**图像检索的前 N 名**。补上这个口径（n=74，真实几何，候选里没有真值时记
**0 而不是缺失**）：

| 候选集 | 覆盖率 | 最终挑对 | 条件准确率 | 相对随机 |
|---|---|---|---|---|
| 图像前 64 名 | 0.284 | **0.054** | 0.19 | 3.5× |
| 图像前 256 名 | 0.419 | 0.014 | 0.033 | 3.6× |

**一、端到端能从 0 抬起来。** 0.054 × 78 ≈ 4 个物体，而管线现在 0/78——管线用的
是深度残差判据（0.041），这套用图像提候选 + 轮廓挑。

**二、候选集要小。** 前 64 名比前 256 名好近四倍，尽管后者覆盖率高一半。
**挑的能力随候选数衰减得比覆盖率涨得快。** 和干净几何上的结论（衰减平缓，
候选多 12 倍只掉四成五）正好相反——真实几何下判据弱，衰减就陡。
所以接进管线时 image_admit 和 residual_top 要设在 64 附近，别往大了开。

**三、瓶颈是两半，大致均分。** 覆盖率 0.284、条件准确率 0.19，两边各要改善约三倍
才能到四成。不是单点突破能解决的。

### 第十五步：端到端 0/75 的归因——过检 × 弱判据，不是掩码

把轮廓挑接进管线后端到端仍是 0/75。机制跑通了（**93/122 条决策改由图像候选胜出**，
之前 0），所以不是开关没点着。

**掩码质量不是原因。** 扫容差线（真值掩码膨胀/腐蚀 1/2/4 格，96 格轮廓上 4 格约
等于物体宽度的 4%）：挑对率 ~0.08 / ~0.12 / ~0.09 / ~0.09，**全在噪声内**。
原来的假设（「轮廓判据整个建立在掩码上，边界一糊就失效」）错了。

**真正的解释是过检。** 管线的物体来自检测器，object_precision 中位 0.333、
均值 0.292——75 个产出物体里只有约 **22 个**真对应某个真值物体，其余根本没有
「正确资产」可言，进了分母就永远不可能命中：

    22 × 条件准确率 0.108 ≈ 2.4 个，实测 0，泊松下 P(0|λ=2.4) ≈ 9%，在噪声内。

**端到端那个 0 是两件事叠出来的：过检吃掉七成分母 × 判据本身只有 0.108。**
两边各要改善三倍左右才看得见数。

**报端到端命中率时必须同时报 object_precision**——否则判据的改进会被过检淹没，
看起来像「方法没用」。

### 第十六步：减少过检不增加命中数，约束在判据

上一步写的「过检和判据各要改善三倍」**是错的**。收紧轨迹长度过滤
（min_track_frame_frac 0.30 → 0.50）：

| | 过滤 0.30 | 过滤 0.50 |
|---|---|---|
| 产出资产节点 | 75 | **53**（−29%） |
| 命中率 | 0/75 | **0/53** |
| 同套件率 | 0.133 | 0.113 |
| 形状距离中位 | 0.0755 | 0.0698 |

**过滤同时删掉真阳性和假阳性**，即使它只删假的，真物体还是约 22 个，
期望命中 22 × 0.108 ≈ 2.4 一点没变——收紧只提高命中**率**的分母侧，
不增加命中**数**。**真正的约束是判据的条件准确率。**

**还有一条**：两臂都是 0 而期望各 2.4，单次 P(0|λ=2.4) ≈ 9%、两次都 0 约 1%
——说明**管线里的条件准确率比探针的 0.108 更低**。探针用真值掩码加真值物体身份；
管线里即使配上真值物体的那些，掩码框的也可能是**别的东西**（不只是边界糊，是框错），
而掩码退化扫描只模拟了边界误差。

### 第十七步：把缝合上——管线第一次非零命中

代码路径固定成管线那条，只把物体身份和掩码换成真值：

| 管线配置 | 产出 | 命中率 | 同套件率 | 决策数 |
|---|---|---|---|---|
| 神经分割 + 轮廓挑 | 75 | **0/75** | 0.133 | 122 |
| 神经分割 + 收紧过滤 0.50 | 53 | 0/53 | 0.113 | 87 |
| **真值分割 + 轮廓挑** | 68 | **1/68** | **0.221** | 70 |

**管线第一次出现非零命中**，同套件率也是今天所有臂里最高的。三段账对得起来：

- 探针（真值掩码 + 真值身份 + 图像前 64 名）0.054 → 68 个该出 3.7 个
- 管线 + 真值分割 实测 1/68，泊松下 P(≤1|λ=3.7) ≈ 10%，偏低但在范围内
- 管线 + 神经分割 0/75

**分割质量是管线侧的主导因素**，判据本身在理想分割下给 ~0.05。

今天四次「探针说该有、实测没有」都出在这道缝上，而每次都是靠推理找解释
（感知档位、口径、掩码质量、过检），对过也错过。**直接量这道缝一次就钉死了。**

所以创新点一的下一步是**分割质量**，不是再换判据——判据已经从 0.041（深度残差）
抬到 0.108（轮廓），而分割从真值换成神经就把 1/68 打回 0/75。注意这和「减少过检」
不是一回事：收紧轨迹过滤砍掉 29% 产出但命中仍是 0，因为它同时删真假阳性。

### 第十八步：名次融合——当天 0% → 7.7%

纯 IoU 挑**把图像分扔了**：候选是图像检索排出来的（正确资产在静止物体上中位排第
36 名），而只按 IoU 重排等于丢掉那个排序。两个信号弱但**独立**（图像看外观、
轮廓看二维形状），用 RRF（倒数名次融合，k=60，无参数；IoU 和余弦相似度量纲不可比，
按名次融合比按分数融合稳）：

| | 纯 IoU 挑 | **名次融合** |
|---|---|---|
| 命中率 | 1/68 = 0.015 | **5/65 = 0.077**（5.2 倍） |
| 同套件率 | 0.221 | **0.262** |
| 形状距离中位 | 0.0806 | **0.0675**（−16%） |

两边都是 61 个真值资产节点、0 失败。**三个指标同向改善**——探针脚本里那条判据
正好用上：只有命中率涨可能是碰巧，三个一起动才是真的。7.7% 还超过探针的组合预测
0.054，因为那个测量只用了轮廓一个信号。

**创新点一当天从 0% 走到管线里的 7.7%**，路径是：判据从深度残差换成轮廓
（0.041 → 0.108）→ 候选集从 256 收到 64 → 图像名次和轮廓 IoU 做 RRF。
**仍然需要真值分割**（神经分割下同一配置 0/75），下一步是分割质量。

### 第十九步：名次融合救不了神经分割——约束是分割

把当天调对的全套配置（轮廓判据 + 候选 64 + 名次融合）放到**神经分割**上：
仍是 **0/69**，同套件 0.145（纯 IoU 时 0.133），形状距离 0.0638（当天最好）。

**完整分解：**

| 变量 | 改动 | 命中率 |
|---|---|---|
| 判据 | 深度残差 → 轮廓 | 0.041 → 0.108（探针） |
| 候选集 | 256 → 64 | 0.014 → 0.054（探针） |
| 融合 | 纯 IoU → 名次融合 | 1/68 → 5/65（管线，真值分割） |
| **分割** | **真值 → 神经** | **5/65 → 0/69** |

**分割质量是创新点一落地的唯一剩余约束。** 判据、候选集、融合方式三样当天都调对
了，合起来在真值分割上把命中率从 0 做到 7.7%，换成神经分割全部归零。

论文里这个数要带限定——「真值分割下 7.7%」。要去掉限定，下一步是分割：更好的
分割后端，或者**用轮廓判据反过来约束分割**（掩码和候选网格的轮廓应当一致，
这条没试过）。

### 第二十步：约束是「框的是什么」，不是掩码形状质量

神经分割下同一配置 0/69。查了两步，**两个假设都被否**：

**一、轮廓一致性当假阳性过滤器——方向反了。** 轮廓 IoU 中位：真值分割 0.043、
神经分割 **0.102**，神经那组更高。（这个信号恰好不在过检那张八条排除清单里，
所以值得试；试了，不成立。）

**二、「神经掩码更圆滑所以分不开候选」——也错了。** 形状统计几乎一样：

| | 占外接框比例 | 边界复杂度 |
|---|---|---|
| 真值掩码 | 0.790 | 24.9 |
| 神经掩码 | **0.750** | **25.3** |

神经掩码不但不更圆滑，还略微更不规则。

**所以差别不在掩码的形状质量，在掩码框的是什么东西。** 一个盖住「半张桌子加一块
地板」的掩码形状统计完全正常、甚至能和某个资产的轮廓对得不错（这解释了神经那组
IoU 反而更高），但它不对应任何真值物体，永远不可能命中。

**创新点一的剩余约束 = 检测找的是不是真物体**（过检问题，见本文前面的八条排除
清单），不是判据、不是掩码细节保真。

### 第二十一步：候选集 64 是工作点，128 三个指标全退

探针上 64 优于 256 是在**纯轮廓挑**时测的。加了名次融合之后融合会把正确答案往前
拉，按说能缓解稀释、最优点右移——**验了，不成立**：

| | 候选 64 | 候选 128 |
|---|---|---|
| 命中率 | **5/65 = 0.077** | 3/64 = 0.047 |
| 同套件率 | **0.262** | 0.234 |
| 形状距离中位 | **0.0675** | 0.0709 |

三个指标全退，**64 就是工作点**，和探针的结论一致。这条参数线封口。

### 第二十二步：神经分割那道缺口，三条解释全否

| 试的 | 结果 |
|---|---|
| 轮廓一致性当假阳性过滤器 | 方向反了（神经 IoU 0.102 > 真值 0.043） |
| 「神经掩码更圆滑」 | 形状统计几乎一样（占外接框 0.750 对 0.790、边界复杂度 25.3 对 24.9） |
| 掩码收成最大连通块 | **无效**（0/67，同套件 0.134 对 0.145，形状距离持平） |

**剩下的只有一条：检测器找的物体本身就和真值不是同一批。** 那是过检问题，
前面已有八条排除清单，这里再加两条（轮廓一致性、最大连通块），共十条。

**判断：不带新想法地继续往这张清单上加第十一条不是好的资源使用。**
创新点一的方法侧已经做完（真值分割下 7.7%，四个变量全部定位并调到最优），
再推要么等检测侧有新思路，要么换数据分布——真值分割本身就是「检测完美」的上界，
7.7% 就是那个上界。

### 第二十三步：瓶颈是命名器——真值类名当短语，命中 14.9%

把**真值类名直接当检测短语**喂给 GroundingDINO（分割、几何全是神经的）：

| 配置 | 命中率 | 同套件率 | 形状距离 |
|---|---|---|---|
| 神经分割 + VLM 短语 | 0/69 = 0.000 | 0.145 | 0.0638 |
| 真值分割 + VLM 短语 | 5/65 = 0.077 | 0.262 | 0.0675 |
| **神经分割 + 真值类名当短语** | **7/47 = 0.149** | **0.553** | **0.0343** |

**给检测器正确的短语，比给它完美的掩码更有价值**——14.9% 对 7.7%，同套件率翻倍
还多，形状距离降一半。三个指标同向，是当天所有臂里最好的。产出数从 69 掉到 47，
精确率跟着涨（具体短语不招那么多虚检）。

**所以瓶颈是命名器，不是检测器、不是分割、不是几何。** 上一步判断「剩余约束是
检测器找的物体和真值不是同一批」只说对一半：检测器确实找错了，但原因是**没告诉
它找什么**——VLM 命名器给泛词，真值类名是具体的。

这把瓶颈从「换检测栈／换几何后端」（周级、不确定）挪到「改命名器」（明确的工程
目标）。注意真值类名里大量是 `tile 099`、`bevel lq brick slope 3x2` 这种资产文件名
——连这种非自然语言的短语都能做到 14.9%，说明**具体性本身就值钱**，不需要很自然。

**下一步：** 让命名器产出具体而非泛化的词。两条可做——用图像检索的 top-k 资产名
反过来当检测短语（闭环），或者让 VLM 看裁剪图说出更细的类名。

### 第二十四步：闭环解不开——名字必须来自检索之外

把**检索出来的候选资产名**当检测短语（两遍感知，全程不碰真值）：

| 短语来源 | 命中率 | 同套件率 |
|---|---|---|
| VLM 命名器（泛词） | 0/69 | 0.145 |
| 闭环检索名（每物体前 3 个、上限 40） | 2/112 | 0.161 |
| 闭环检索名（每物体前 1 个、上限 10） | **0/76** | 0.132 |
| **真值类名（预言机）** | **7/47 = 0.149** | **0.553** |

**两版都不成立。** 40 个短语时过检暴涨（产出 69→112），压到 10 个又回到 0。

**根因是循环本身**：闭环只能提出它已经检索到的名字，而第一遍检索是 0/69——候选
大多是错的资产，名字描述不了真实物体。**检索差 → 名字差 → 检测差 → 检索差**，
环自己解不开。

**所以那 14.9% 要更精确地表述：具体的名字有用，但名字必须来自检索之外。**
下一步是**命名器本身**——让 VLM 看裁剪图说出具体类名（现在给泛词）。这是提示词／
模型的问题，和几何、检索都无关，是整条链上最便宜改的一环。别再试闭环变体。

### 第二十五步：口径更正 + 通用词表也不行

**此前所有「神经分割」臂用的不是 VLM 命名器，是写死的 `DEFAULT_PHRASES`**：
`toy train, train track, cardboard box, conveyor belt, ball, toy car, door, table,
cup, bottle, chair, robot`——来自项目早期的玩具火车／桌面视频，而 assetgt 里是
摩天楼、瓷砖、路口、岩石、宇航员。原因：`rebuild_with_assets.py` 调
`run_perception` 时不传 client，命名器那条分支直接跳过。**命名器从来没跑过。**

**通用词表也不行。** 从资产库推出高频词（tile wall brick road track building tree
window roof tower，库是推理时就有的）：**0/77**，同套件 0.169。

| 短语来源 | 命中率 | 同套件率 |
|---|---|---|
| 写死的默认词（玩具火车域） | 0/69 | 0.145 |
| 资产库高频词（领域对口） | 0/77 | 0.169 |
| 闭环检索名 | 0/76 ~ 2/112 | 0.132–0.161 |
| **真值类名（逐物体具体）** | **7/47 = 0.149** | **0.553** |

**管用的不是「领域对口的词表」，是「逐物体的具体名字」。** 通用类别词给不了检测器
足够信息，而且检出一个 "tile" 之后库里有 496 个 tile，检索照样分不开。

**能产出逐物体具体名字的只有 VLM 命名器，而它没接上。** 下一步是基础设施：
给 rebuild_with_assets 接一个 VLM client。接上之后还要改提示词——现在
`prompts/namer.md` 明文写着「no colours, no adjectives about size」，
**正好禁掉了今天测出来值钱的那一维**（真值类名里管用的是 sedan sports、
tree pine、candy cane red 这种带子类带颜色的）。

### 第二十六步：不经过检测的具体短语也不行——patch 采到的是场景不是物体

绕开闭环（名字不来自检测结果）：关键帧切 4×4 网格，每块 patch 直接做图像检索取
资产名，投票取前 8。全程不碰真值。

| 短语来源 | 命中率 | 同套件率 |
|---|---|---|
| 写死默认词（玩具火车域） | 0/69 | 0.145 |
| 资产库高频词 | 0/77 | 0.169 |
| 闭环检索名 | 0/76 ~ 2/112 | 0.132–0.161 |
| 网格检索名 v1（含 tileNNN） | 0/50 | **0.240** |
| 网格检索名 v2（滤掉纯编号） | 0/57 | 0.175 |
| **真值类名（预言机）** | **7/47 = 0.149** | **0.553** |

v1 的同套件 0.240 是所有神经分割臂里最好的，但精确命中仍 0。**v2 滤掉纯编号名
反而更差**——`tileNNN` 虽然无意义却让检测器少找到东西（产出 50→57），滤掉之后
过检回来了。

**根本问题在生成的短语里看得很清楚**：全是 wall / cliff / ground / floor / tile
这类**场景表面**，而真值资产节点是 bedbunk / frying pan lid / sedan sports /
astronautb 这种**摆进场景的物件**。网格 patch 大量落在地形上，分不开物体和背景。

**所以逐物体的具体名字只能来自一个看得懂「哪里是物体」的东西——VLM 命名器。**
它在 rebuild_with_assets 里没接上，接上需要推理端点。这是基础设施步骤，不是实验。

### 第二十七步：命名器接好了，卡在密钥

把 VLM 命名器接进 `rebuild_with_assets.py`（之前调 `run_perception` 不传 client，
命名器那条分支直接跳过，落到写死的 DEFAULT_PHRASES）。接上之后仍不生效，查了四处：

**根因一：worktree 没有 `.secrets/`。** 作业跑在 git worktree 里，而 `.secrets/`
按设计不入 git，`api_key_file` 又是相对仓库根目录的。所以**任何需要密钥的组件，
在「从 worktree 跑作业」这套方式下从来就跑不起来**。软链解决。

**根因二：密钥失效。** 软链之后 `make_client` 成功、请求到达 OpenRouter，
返回 `401 User not found` ——网络通，是账号密钥本身被拒。

**三层遮蔽让这个明确的失败看起来像「什么都没发生」：** make_client 失败时设计上
退回默认词不抛错；那行提示被作业脚本的 `| tail -15` 吃掉（管线前面有 76 秒抽帧
输出）；失败发生在 `name_phrases` 之前所以 `errors.jsonl` 也是空的。

**状态：线接好了（含 `--no-namer` 对照开关、worktree 软链），等一把有效密钥。**

### 第二十八步：命名这条路不成立——那 14.9% 是循环的

把 VLM 命名器接上（本地 vLLM Qwen3.5-27B-FP8，12 条片段 `phrases_source=vlm_namer`
全部确认真跑）：**命中 0/50、同套件 0.180、形状距离 0.0728**。比写死的默认词
（0/69、0.145）好一点点，离真值类名那档（7/47、0.553）差很远。

**命名器本身没问题**：冒烟那条真值是 `bedbunk`，它给出 **`bunk bed`**，叫对了。

**问题在真值类名是什么东西：**

    tile 099 · bevel lq brick slope 3x2 · square lq brick slope 3x1
    building skyscraper b · terrain roadcross · path stonecircle

**这些是资产文件名，没有任何视觉模型能从画面里说出来。** 真值类名那 14.9% 之所以
高，不是「具体的名字帮助了检测」，而是**那些名字就是检索的键**——文字检索拿资产
自己的名字去查自己，同套件率 0.553 正是这么来的。

**所以第二十三步那个「瓶颈是命名器」的头号结论，是循环论证的产物。** 和本文前面
「文字检索上限是循环的」同一个根：**这个数据集的类名来自资产文件名**。

任何以真值类名为输入的实验，在这个数据集上都是循环的，不能当上界也不能当目标。
要测命名的价值，得先让数据集的类名**不等于**资产名（生成器一行改动）。

### 第二十九步：数据集的靶子坏了 + 命名这条路封口

**先是一个数据集 bug。** `assetgt`（10-23 21:51 生成，驼峰修复之前）有 **14.8%
的真值类名是坏的**（`tableCoffeeGlass` → `tablecoffeeglass`、`bedBunk` →
`bedbunk`），而类名正是检索查询的目标——**当天所有端到端的数都是在这个损坏的
靶子上打的**。`assetgt60`（当晚 23:26 重新生成）类名 100% 干净。

发现路径：命名器说 `bunk bed`、真值类名是 `bedbunk`，语义对上但分词零交集 →
怀疑资产库有大量连写复合词 → 查下来只有 1.2% 且多是真单词 → 那就是个例外 →
查它的资产 id 是 `furniture/bedBunk`，驼峰没拆 → 查全数据集 14.8% 都这样。
**起点是没有把「命名器 0/50」直接当结论收工，而是去看了它到底说了什么词。**

**在干净数据集上重跑**（24 条、122 个真值资产节点、两臂只差短语来源、
`phrases_source=vlm_namer` ×24 全部确认真跑）：

| | 产出 | 命中率 | 同套件率 | 形状距离 |
|---|---|---|---|---|
| 写死默认词 | 137 | **0/137** | 0.153 | 0.0728 |
| VLM 命名器 | 114 | **0/114** | 0.140 | 0.0753 |

**命名器没有带来改善**，同套件率还略低。**命名这条路关闭。**

### 旧的下一步（已被上面取代一半）

补救方向是**表面重建**，不是降噪，也不是换一个「更准」的单目深度（它们同样会沿
视线压扁）。而且**现成的廉价修正都试过了**——沿最短主轴拉回无效（第 9 步）。
真要补偿得**逐帧沿视线**做，或者换思路用**多视轮廓约束**（visual hull）：
轮廓是二维的，不受沿视线的深度误差影响。两条都是实打实的工作量，不是一行变换。

### 诊断链闭合

每一环都排除了一个具体假设：

1. 端到端 0/78 → 不是管线接错，是判据失效
2. 候选覆盖率 0.419（静止 0.875）→ 提候选没问题
3. 真值几何 0.297 对真实几何 0.041 → 失效在几何这一维
4. 只换相机 0.054 → **不是相机位姿**
5. D2 与残差打平（都 0.027）→ **不是判据对噪声敏感**
6. 大物体 0.045 对中物体 0.033 → **不是物体太小**
7. 5% 噪声仍有 0.108 → **不是深度噪声**
8. 最短轴比例 0.526、尺度 0.420 → **是沿视线压扁 + 整体缩小**
9. 沿最短主轴拉回 1.9 倍 → **无效**（各档都在随机附近，静止反而 0.250→0.125）。
   这个修正从设计上就错了：深度误差沿**相机视线**，而视线方向**逐帧都在变**；
   物体自身的最短主轴是固定方向，拿它去撤销逐帧变化的畸变撤不掉。量到的 0.526
   是那个畸变在物体坐标系里的**投影平均**，不是畸变本身——**症状，不是杠杆**。

**VGGT 重建出来的物体形状本身就不对，不是正确形状的噪声版本。**

所以创新点一的机制**已被证明成立**（干净几何上 8~62 倍于随机，密度匹配零假设下
仍站得住），但在当前几何栈上**没有可行路径**。要推进只有换几何后端这一条。
这个结论对创新点二同样适用——正确网格当模板依赖同一个表面信息。

其中第 4、5 条是我自己提出又被数据否掉的假设，第 6 条是一直没动过的变量
（今天所有实验都在 stage_scale 0.15 上跑）。记在这里是因为排除清单的价值和
正面结果相当：它挡住了后来者重走这三条。

## 候选召回才是大头：一直在优化天花板更低的那个因子

端到端挑对率 0.054 早就能分解干净，但我没按分解去安排工作：

    挑对率 0.054 = 候选覆盖率 0.284 × 候选集内条件准确率 0.19

连着做的轮廓重排、RRF 名次融合、裁剪图命名，**全在修右边那个 0.19**。
而右边再完美也只能到 0.284——重排挑不出一个从没进过候选集的资产。左边那个
0.284 只试过「把同一个信号开大」（64→128→256：覆盖率涨到 0.419、条件准确率
塌到 0.033，净亏），**从没试过换一个信号去生成候选**。

乘积式指标里两个因子的可改进上限完全不对称，但手感上一样，而且右因子改起来
更顺手（代码在手边、跑得快）。于是反复优化上限更低的那个。

### 为什么几何该在候选生成这一步就进场

到今天为止，几何只被允许**重排一个由外观生成的 64 人池子**，从没被允许在
全库 4812 个里**生成候选**。而创新点一的主张正是「用连续稠密表征监督离散的
符号程序选择」——那它不该只在最后一步进场。所以给探针加了两条新腿：
边长比例（三个数）和轮廓 Hu 矩（九个数），都在全库里排序。

### 边长比例这条腿封顶了，而且不是几何的锅

先踩了一次循环：拿**真值** `geom.extent` 的边长比例去全库排序，前 64 覆盖
0.350，远高于重建 extent 的 0.000。差点写下「形状的信号是真的，差距全在几何，
修几何有用」。查了一下，这个数据集对资产是**等比缩放**——真值 extent 的比例
和 `asset_shapes.json` 里的原生比例 L1 差中位 **0.0003**（148 个节点）。
等于拿资产自己的比例去找它自己。和类名那个坑同构：**真值派生出来的同一个量，
同时当了查询和键。**

循环的数不是废数，**读法要反过来**：它量的是描述子的分辨力上限，不是几何的
上限。这么一读结论反而更硬。在 4812 个资产里随机抽 300 个，问「边长比例和它
几乎一样的有几个」：

| L1 容差 | 撞车个数（中位） | 四分位 | 最多 |
|---|---|---|---|
| 0.02 | 13 | 4 / 70 | 229 |
| 0.05 | **52** | 30 / 173 | 349 |
| 0.10 | 158 | 111 / 362 | 586 |

候选预算只有 64。**哪怕边长比例量得完全精确**，中位情况下也有 52 个资产并列，
前 64 名里挑不出唯一解——0.350 就是这么来的。反过来看精度要求：要让撞车数
降到 64 以下，L1 误差得好于 0.05；而 VGGT 压扁对应的 L1 误差在 0.3~0.5 量级，
**差十倍**。两条独立论证同向：这条腿封死，不是「几何还不够好」。

### 一个正在成形的判断：全局低维描述子当不了候选生成器

轮廓那条腿本来是最有希望的——已坐实根因是 VGGT 沿相机视线压扁物体，而沿视线
的误差投影到像平面恰好为零，所以轮廓是唯一绕开根因的判据（实测比深度残差好
2.6 倍），而且它不循环：资产侧是渲染缩略图，观测侧是视频掩码，两次不同的渲染，
正是要解决的任务本身。

但全库排序下来，轮廓的表现和边长比例是一类的。三个数的边长比例、九个数的
Hu 矩，**都是全局低维描述子**，在 4812 个资产里都撞车撞到挑不出来；唯一能
生成候选的是高维学出来的图像嵌入，而它封在 0.284。

如果全量样本证实这一点，那这篇文章的位置就清楚了：**判据本身（用稠密几何给
离散选择打分）是成立的**——干净几何上 4.1~62 倍于随机、真值分割下管线 9.0%；
**卡住的是检索前端**，而它是个表示学习问题，不是这篇文章的主张。老实写成
「给定候选集包含正确答案时，连续稠密几何能有效监督离散符号选择」，并把候选
召回量化成显式的前提条件，比声称端到端可用要硬。

## 嵌入没问题，是数据集给物体重新上色

覆盖率 0.284 到底是「嵌入不行」还是「视频域不行」，这两种原因的修法完全相反，
所以必须先分开。判据用留一视角法：拿资产 A 的一个视角当查询，**把 A 的那个
视角从库里挖掉**，问 A 排第几（A 另外三个视角还在）。纯渲染对纯渲染、同材质、
只差视角——它量的就是嵌入本身的跨视角实例检索能力。

    中位名次 1 / 4812   第一名 0.668   前 64 名覆盖 0.990   （600 个资产）

**DINOv2 做这件事完全没问题**，4812 个资产对它不算多。所以 0.284 的锅在视频域。

而域里最大的一项基本可以确定是颜色：这个数据集**给每个物体重新上色**，
`material` 字段给的是任意的 base_color / roughness / metalness / emissive——
`food/chinese` 被渲染成 `#4fce98` 的青色，`retro-urban/detail-bricks-type-b`
带自发光。库里的缩略图却是资产的**原生外观**。于是查询（一个青色物体）和库里
那一项（原生贴图的同一个网格）在一个对外观敏感的嵌入里看起来像两个东西。

这条解释和已有的数都对得上，而且它解释了一件之前没解释的事：为什么同套件率
（0.14~0.19）总是远高于命中率（~0）——同一套件的资产往往形状相近但贴图不同，
颜色一乱，套件内就挑不准了。

验法便宜：两侧都转灰度再嵌入，把颜色这一项去掉。涨 → 颜色是主因，不用换模型；
不涨 → 锅在光照/粗糙度/自发光，再往下查。

## 判据可用 ≠ 判据自带置信度

模板路线在**正确网格**下把静止/转动的判别比从 2.60 提到 4.31，在检索到的
（多半是错的）网格下比基线还差，而命中率只有 9%。整体开模板必然净亏——
这就是它挂起的原因。但「整体开」不是唯一选项：**如果能分辨哪些物体检索对了**，
就只在那批上开，不需要检索变好。

真值分割那一臂的前 38 条、111 个决策，整体命中 0.081：

| 信号 | 四个桶的命中率（低→高） | 最高 25% / 整体 |
|---|---|---|
| 文字分 | 0.000 / 0.000 / 0.071 / **0.250** | **3.08 倍** |
| 图像分 | 0.074 / 0.107 / 0.036 / 0.107 | 1.32 倍（不单调） |
| 轮廓 IoU | 0.037 / 0.107 / 0.107 / 0.071 | 0.88 倍（平的） |

两件事要紧。一、文字分单调且 3 倍，是目前找到的第一个可用置信信号；而且这一臂
用的是写死的默认短语（`--no-namer`），不是真值类名，所以不触犯那个循环。

二、**轮廓 IoU 和对错完全不相关**，而它正是我投入最多的那个判据。它能在候选集
里挑出一个，但它的分数不表示挑得对不对。对创新点一这是个要紧的负面结果：
**判据可用不等于判据自带置信度**，而下游（模板、关节）要的恰恰是后者。

## 分母里有一半的决策在真值里没有正确答案

三条关于覆盖率的假设接连被量化证伪（嵌入 0.990、颜色只占 37.6%、分辨率 16px
仍 0.958），间接判据用完了。于是**把真实查询图存出来直接看**——12 张里 7 张的
左边是 `block_*` / `rail_*`，素面盒子。

查真值（assetgt60，60 条）：

| 真值节点类型 | 有资产 ref |
|---|---|
| `obj_*` | **290 / 290 = 100%** |
| `rail_*` | 0 / 249 |
| `block_*` | 0 / 60 |
| `ground_*` | 0 / 60 |

**真值明说结构件是基本体，「该挑哪个」这个问题对它们不存在。** 按节点类型拆开
273 个决策：

| 类型 | 命中率 | 占比 |
|---|---|---|
| `obj_*` | **28/122 = 0.230** | 44.7% |
| `rail_*` | 0/103 = 0.000 | 37.7% |
| `block_*` | 0/48 = 0.000 | 17.6% |

**整体那个 0.103 里 55.3% 是不该做的决策。真实检索准确率是 0.230。**
覆盖率 0.284 和 `0.284 x 0.19 = 0.054` 那个分解，也都是在同一个被污染的人群上
算出来的。

### 根因：`moves` 在静止结构件上 88.2% 假阳

`direct.py` 判结构件是 `is_static_class(cls) and not has_motion`，而
`has_motion` 里那个 `moves` 标志在 `rail_*` 上 **60/68 = 88.2%** 说它会动
（真值里 rail 全是 static），多半是 ICP 在相机运动下无中生有解出转动。
于是结构件规则被假阳整个关掉，rail 进 `objects`、标成 `kind=asset`、送进检索。

当初换用 `moves` 的理由写在注释里：「真值感知上**查全** 0.925」——
**只量了查全，而这里需要的恰恰是精确率**。

### 两条不依赖类名的替代都不成立

- **位移**：`rail` 的「位移/对角线」中位 1.81，真物体 1.93，分不开。
- **几何**：以为轨道该细长，实测反过来——`obj` 细长比 2.20 比 `rail` 的 1.90
  还高；各档阈值下两者比例几乎一样。而且 `rail` 重建出来的最长边中位只有
  **0.04 米**，**轨道根本没被重建成轨道，是一堆碎片**。

所以只能靠类名。而类名在真值分割这一档是真值给的，神经分割那档
`class_accuracy` 是 0.000——**这个修复在端到端上不一定复现，是个要写进限制的事**。

### 修法改了两版

第一版按词匹配，验证时抓到 `obj_2`（类名 `terrain road cross`，货真价实的路面
资产）被推进了 static。量下来**资产库的类名本身就带这些词**：`wall` 23/290、
`road` 16、`track` 11、`floor` 6、`block` 4、`rail` 3。

真值里结构件的类名是**单个词**，所以换成**完全相等**：抓到结构件 369/369 =
100%，误伤资产 1/290 = 0.3%。这个道理 `is_ground_class` 的注释里早就写了
（「直接改归属，强度太高宁可漏不可错」），我加同强度的新判据时没照着做。

端到端验证（5 条片段）：`rail`/`block`/`ground` 全部进 static+primitive，
资产决策 **0% 落在结构件上**（修前 55.3%），物体保留 13/14 = 92.9%
（丢的那个类名就叫 `floor`，是既有的 `is_ground_class` 捞走的，与本次改动无关）。

## 清干净之后的分解：判据不是瓶颈，候选召回才是

两个独立测出来的数，口径一致（都在良定义人群 `obj_*`，都用「本片段真值集合」
这个集合判据），可以相除：

    端到端挑对率  0.209   （158 个决策，管线真跑）
    候选覆盖率    0.243   （185 个物体，图像分在全库 4812 个里前 64）
    ------------------------------------------------
    条件准确率    ≤ 0.86

对照今早那个被污染的分解：覆盖率 0.284 x 条件准确率 0.19 = 0.054。

**0.86 是上界不是下界**：覆盖率只算了图像分，而管线的候选池还有文字那一路，
真实覆盖率更高、相除之后条件准确率更低。但即使管线覆盖率到 0.35，
条件准确率也还有 0.60——远高于 0.19。

**这条直接回答创新点一提的问题。** 用稠密几何（轮廓 IoU / 配准残差）从候选里
挑那一个，**挑得相当准**；整条链卡在**正确答案根本没进候选池**。
所以要投的是候选生成，不是判据。

### 三种表示的候选生成，量完了

| 索引 | 良定义人群前 64 覆盖 | 中位名次 |
|---|---|---|
| 彩色 | 0.243 | 317 |
| 灰度 | **0.249** | 346 |
| 纯轮廓 | 0.119 | 419 |

**灰度 ≈ 彩色（+0.006，n=185 上就是噪声）→ 颜色这条关闭。** 染色/原生的
大小匹配对照印证了同一件事：彩色索引上原生 0.067 对染色 0.047（1.4 倍，真实
但很小），灰度索引上收敛到 0.060/0.057——灰度确实抹平了那个差，但抹平之后
总量没涨。**纯轮廓明显更差**，说明 DINOv2 在这个库上靠的不只是外形。

### 裁剪尺寸这条要和降采样阶梯一起读

按裁剪短边分桶（彩色索引，良定义人群）单调上升：

    10-20px 0.059   20-28px 0.108   28-48px 0.186   48-159px 0.186

但降采样阶梯把缩略图**两侧一起**降到 16px，留一视角自匹配仍有 **0.958**
（原生 0.990）。两条合起来：**干净的分辨率损失无害，真实的小裁剪图确实更差**，
所以尺寸是**别的东西的代理**——遮挡、掩码边缘、斜视角、运动模糊、距离远——
不是像素数本身。只看其中一条都会得出错的结论。

## 二次配准实测：闸对检索有效，但传导不到位姿

模板路线一直卡在一个算术上：正确网格把静止/转动判别比从 2.60 提到 4.31，
但检索命中只有 9%，整体开 = 9% 受益、91% 受害，必然净亏。而文字分能闸出一个
子集（score>=1.0，最高 25%），里面**检索命中 0.625**（整体 0.209），
且与裁剪尺寸无关（r=+0.038）。看起来正是缺的那块。

配准跑在感知里、在 `choose_assets` **之前**，所以这条路必须是**第二遍**：
拿第一遍产出的 `decisions` 回头用检索到的资产网格重新配准一次。写出来跑了
（12 条片段，无失败，**逐物体配对**同时记模板解和基线解）：

| 真值静止的物体 | 模板 RMSE | 基线 RMSE | 模板/基线 | n |
|---|---|---|---|---|
| `open`（未闸） | 45.43 | **20.55** | 2.2x 差 | 10 |
| `gated`（闸出） | 21.73 | **9.16** | 2.4x 差 | 3 |

**闸确实让模板的绝对值变好（45.43 → 21.73），但基线同步变好（20.55 → 9.16），
比值几乎没动。** 静止那一侧 n=13、方向一致：**模板比基线差一倍以上。**

转动那一侧样本太小（spin n=1、periodic_rotate n=1、revolute n=2），不读，
方向也还混着。

### 预测错了，错因值得记

算盈亏平衡时我用的基线是 **25.28°**，那来自**另一轮**（ab015fix、全真值感知），
而这一轮同一批物体上的基线是 **9.16~20.55°**。**把不同配置的数混进同一个算式**，
平衡点就没有意义。算期望收益之前，基线必须和实验臂来自**同一批样本、同一条
代码路径**。

这一轮的结论本身可信，恰恰是因为 tmpl 和 base 是**同一个物体、同一次运行**
同时记下来的——配对设计救回了一个本来会被小样本毁掉的实验。

**结论：文字分闸 + 资产网格模板这条关闭。** 闸对**检索**是真有效的，
但那个优势**没有传导到位姿**这一级。创新点二的瓶颈不在「模板拿的网格对不对」。

## 多帧检索：方向对，但额外的帧是冗余的

瓶颈定位到候选覆盖率之后，第一个直接攻天花板的机制是多帧检索——覆盖率这一路
从头到尾只用了一张裁剪图（`best_observation` 取框面积最大的那一帧）。
理由本来很硬：同一个项目里**残差判据**的多视角收益是
1/3/7 帧 = 0.149 / 0.243 / 0.311（2.1 倍），7 帧才饱和。

实测（良定义人群 `obj_*`，n=185，配对）：

| 帧数 | 前 64 覆盖 | 中位名次 |
|---|---|---|
| 1 | 0.243 | 317 |
| 3 | 0.254 | 316 |
| 7 | **0.265** | **315** |

方向单调，但只有 +0.022（相对 +9%），n=185 下在一个标准误以内，**不算确证**。

**最说明问题的是中位名次几乎不动（317 → 315）。** 如果多帧真在加信息，正确
资产的名次应当明显前移。它没动——说明**对帧取 max 基本就是在挑那一个最好的
视角，而 `best_observation` 已经近似挑中了它**，多出来的帧是冗余的。

**为什么和残差那边差这么多**：残差跨帧聚合的是**几何**，不同视角真的在补三维
形状的不同部分；图像嵌入取 max 只是在选一个最好的二维视图，信息没有叠加。
**「多视角有用」这个经验不能从几何判据直接搬到外观判据上。**

方法论上记一条：判断一个聚合是不是在加信息，**看连续量（中位名次）动没动**，
别只看阈值指标（前 64 覆盖）——阈值指标会把「边界上挪了几个」放大成像是进展。

## 收回：判据**才是**瓶颈——上一节把两个不同的候选池混了

上一节用 **0.209 ÷ 0.243 = 0.86** 得出「判据不是瓶颈、候选召回才是」。**那是错的。**

0.243 是**图像分单独**在全库里的覆盖率，而端到端 0.209 是**管线**产生的，
管线的候选池是 `retrieve()` 的 **text+image 融合**。拿 A 池的覆盖率去除 B 池的
准确率，没有意义。

按管线那条真实路径量了一遍（同一个 `retrieve`、同一组 image_weight / image_admit）：

| | 值 |
|---|---|
| 管线真实候选池覆盖率（良定义节点） | **185 / 185 = 1.000** |
| 端到端挑对率 | 0.209 |
| **条件准确率** | **0.209** |

**正确的答案一直在候选池里，而判据只挑对 21%。所以判据就是瓶颈。**

池内随机基线：池子 64 个、每条片段真值资产集合均值 4.83，集合判据下随机
≈ 0.076，所以 **0.209 是池内 2.8 倍于随机**——真实，但远谈不上解决。

### 那个 1.000 是循环的，必须标注

这一档的 `class_guess` 来自真值程序，而真值类名就是资产文件名：实测 **86.5%**
的 `class_guess` 的词直接覆盖资产文件名（`chinese` → `food/chinese`、
`tile 099` → `tower-defense-classic/tile_099`、`frying pan lid` →
`food/frying-pan-lid`）。文字检索拿着答案去找答案，当然 100% 覆盖。
这是同一个循环坑的第三次。

### 对创新点一的影响

**主张要改。** 不是「稠密几何判据挑得很准、卡在召回」，而是：

**在一个 64 个候选、且保证含正确答案的池子里，稠密几何判据（轮廓 IoU /
配准残差）挑对 0.209，池内随机是 0.076——有信号（2.8 倍），但不够。**
而「保证含正确答案」这个前提在这个数据集上是靠循环的类名拿到的，真实系统没有。

所以创新点一要回答的问题仍然敞着，而且现在定位更准：**不是候选生成，
是判据本身的分辨力**。

## 偏置诊断的口径是坏的，修完之后误差在像平面内

反压扁（沿视线把点云拉回来）实测**单调变差**：静止物体的假转动
32.03°（对照）→ 43.84°（×1.5）→ 46.38°（×1.9）。去查为什么，发现**量具坏了**。

管线的 `metric_scale` 实测是
`{"factor": 1.0, "source": "none", "reason": "only 0 objects had a size prior"}`
——**根本没做米制归一**。证据坐标是 VGGT 的任意尺度，真值是米，直接相减得到的
「定位误差」绝大部分是尺度和坐标系差。

症状本来很明显，我差点没看出来：中心误差中位 **1.303 m**，而真值物体最长边
中位只有 **0.212 m**（误差是物体的 6 倍、接近整个场景宽度 3 m）。

**更危险的是它会造出一个看起来很有物理意味的结构**：一个**固定的世界系偏移**，
在视线方向随相机旋转时分解，天然就是「垂直分量远大于沿视线分量」。
对齐前 `perp 1.303` vs `along 0.193`（6.7 倍）正是这个指纹。差一点就据此宣称
「轮廓判据免疫深度误差」那条立论不成立。

### 用 Umeyama 相似变换对齐之后

| | 对齐前 | 对齐后 |
|---|---|---|
| `raw` | 1.927 m | **0.297 m** |
| `along`（沿视线） | 0.193 | **−0.019** |
| `perp`（垂直视线） | 1.303 | **0.206** |

**对齐把误差砍掉 6.5 倍**，证实绝大部分确实是口径问题。残差的结构是：
**沿视线误差基本为零，垂直视线误差 ≈ 0.206 m ≈ 一个物体长度。**

注意这和「VGGT 把**形状**沿视线压扁到 0.526」不矛盾——那是**形状**畸变，
这是**中心**定位误差，两件事。所以轮廓判据的立论没有被推翻。

### 顺带量到：VGGT 的尺度在片段之间差 4 倍

`fit_scale` 中位 **1.131**，四分位 **0.655 / 2.646**。不是一个常数偏差，
是**逐片段漂的**。任何依赖绝对尺寸的判据都要先过这一关。

### 对创新点二的含义

逐帧点云的不一致**主要是像平面内的中心散布**，不是沿视线的压扁。所以
「动观测」的正确方向不是反压扁（实测单调变差，现在知道是**修错了分量**），
而是**稳住像平面内的定位**。

## 转动闸：位姿上成立，但传不到第三档关节

**位姿这一侧是验证过的。** 零假设闸（`rotation_earns_its_keep`，零假设「只有
平移」，备择「平移+转动」）一直实现着、接好了，但默认关着、注记「端到端 A/B
没做之前不开」——**那个 A/B 从没做过**，而我今天所有假转动的数都是在它关着时量的。

做完了（全 60 条、逐物体**配对**、按**真转角连续分桶**）：

| 人群 | n | err_rmse 闸关→0.10 | 更好/更差 | 符号检验 p |
|---|---|---|---|---|
| **真值不转** | 99 | 24.19 → **17.28** | 79 / 18 | **<0.0001** |
| 真值在转 | 34 | 32.56 → 34.67 | 21 / 13 | 0.23 |
| 真转角最大一半（≥41°）| 17 | 42.08 → 37.58 | 8 / 9 | 1.00 |

静止侧大幅改善且高度显著、转动侧无显著损害——正是预登记的判据。
关键佐证：大转动组解出幅度 42.7°→40.1° 而真值中位 **94.9°**，基线本来就严重
解不足，闸几乎没再动它，所以**不是在压真转动**。**已默认开 0.10，1031 测试全过。**

### 但它传不到关节这一档

关节推断的铰链门槛是 **8°**，而假转动是 13.5~26.6°——所以
「`infer_articulation` 可用率只有 0.333、默认关掉」也是闸关着时的结论。
配对重测（129 个物体，真值关节 15 个）：

| | 报出关节 | 其中真的是 | 精确率 |
|---|---|---|---|
| 闸关 | 3 | 1 | **0.333** |
| 闸开 0.10 | 21 | 3 | **0.143** |

**开闸多报 7 倍、精确率掉一半。** 扫阈值也救不回来（闸开最好一档 0.250，
仍低于闸关的 0.333）。推测的机制：闸拒绝转动时返回纯平移解，而**被放行的那些
转动更"干净"**，更容易过 `axis_consistency ≥ 0.85`——报得更多但不更对。

**所以：降低转动 RMSE 不等于改善关节推断。** 原本默认这两件事连在一起
（假转动降下来、假关节自然少），实测是反的。位姿要的是**角度准**，
关节要的是**轴稳定且该不报时不报**，在同一个量上甚至冲突。

**方法论**：中间量改善了不要假定下游跟着改善——中间量 p<0.0001 地变好，
下游却腰斩。要测下游那个量本身。

## 跨帧聚合规则这一族关闭

撤回 min 之后留下一个线索：min 相对 median 的优劣**随可用观测帧数单调翻转**
（帧数 10~19 时赢 14 输 10，帧数 32 时赢 7 输 14；裁剪尺寸分桶没有趋势）。
机制清楚——`min` 取**最差的那一帧**，可用观测越多、抽到的 7 帧视角跨度越大，
**连正确资产也总会有一帧匹配得差**（自遮挡、被别的物体挡），于是把正确答案
一起罚掉。这正好解释了 min 的「赢多输少但方向一半一半」。

据此推出正确形式应当是**低分位而非严格最小值**。全 60 条跑完，**配对检验**：

| 检验（配对，154 个物体）| 结果 |
|---|---|
| 命中率 McNemar | 11 对 12，**p = 1.000** |
| 形状距离 符号检验 | 42 对 47，**p = 0.672**，配对差值中位 −0.0006 |

**q25 完全无效**——不是更差，是和 median 无从区分。

**所以结论收紧成一族**：min 过严、q25 无差别、median 是现状，
**跨帧聚合规则这一族关闭**。再扫 q10/q40 没有意义，整条轴的两端和中间都试过了。

**这个否定比「min 不行」有用**：它排掉的是一整族参数化而不是一个点。
下次再想「多视角应该有用」，要换的是**判据本身**（每帧产出什么），
不是**怎么合并多帧**。

### 创新点一目前的完整排除清单

瓶颈已钉死在**判据的池内分辨力**：管线真实候选池在良定义节点上覆盖 **1.000**
（但循环——86.5% 的 `class_guess` 直接覆盖资产文件名），判据挑对 **0.209**，
池内随机 **0.076**，只有 **2.8 倍**。试过并否掉的：

| 方向 | 结果 |
|---|---|
| 三路名次融合 | 输（命中腰斩）——名次融合丢掉量纲 |
| 灰度索引 / 纯轮廓索引 | 0.249 / 0.119 对彩色 0.243——颜色解释关闭 |
| 多帧图像检索 | +9% 在噪声内，中位名次不动（冗余） |
| 轮廓帧数 3→7 | 三指标两退 |
| 跨帧聚合 min / q25 | 配对检验全不显著，**整族关闭** |

## 轮廓判据的参数面关闭

把轮廓判据的三个内部参数逐个扫完，**全部无效**（都用配对检验，全 60 条片段，
每次约 155 个配对物体）：

| 参数 | 改动 | 配对检验 |
|---|---|---|
| 帧数 | 3 → 7 | 三指标两退（命中 0.213→0.177） |
| 跨帧聚合 | median → min | McNemar p=0.155、符号 p=0.685 → **撤回** |
| 跨帧聚合 | median → q25 | p=1.000 / 0.672 → 无差别 |
| 偏航搜索 | 8 → 2 | p=0.540 / 0.832 → 无差别 |

**结论：轮廓判据的内部参数化整体没有余量。要推动创新点一，得换掉轮廓这个
判据本身，调参数不会有结果。**

### 偏航那一条的额外信息

本来预期**减少**偏航会有利：`hi = max(...)` 是每帧 best-of-8，而
「取最好初值会偏袒错候选」在残差判据上实测过（挑对率 0.297 → 0.216）。
结果 8 和 2 **没有区别**——说明**轮廓匹配对朝向根本不敏感**，这些轮廓太粗，
分辨不出朝向差异。

这和「判据整体只有 2.8 倍于池内随机」是一致的：**判据弱到连自己的超参数都
影响不了它**。一个判据的多个独立超参数都扫不出效果时，那是在说判据本身的
分辨力不够，参数只是在无信息的空间里挪动。

轮廓这一族（二维投影 + IoU）到此为止。下一个候选判据必须在**信息量**上有本质
不同，而不是同一个量的不同算法。

## 非循环条件下的端到端：~2%，换什么都一样

`force_query` 把检索查询词强制成通用词，候选池只由图像分决定，循环断掉
（真值类名=资产文件名，86.5% 重合）。在这个条件下：

| | 命中 |
|---|---|
| 轮廓判据（60 条，175 决策）| 0.023 |
| 残差判据（60 条，175 决策）| 0.017 |
| 残差 + 候选池放大到 256（30 条，87 决策）| 0.011 |

**端到端 ~2%，换判据、放大候选池都不改变。** 最后一行两臂命中数是 2 对 1，
量级太小，**那个实验功效不足**，不作方向性结论用。

### 两条探针结论都没能搬到管线

1. 「残差 0.041 输给轮廓 0.108」→ 据此 `residual_top: 0`。管线真实条件下
   残差反而**显著更好**（0.344 对 0.195，p=0.0056）。
2. 「残差衰减平缓，候选多 12 倍只掉四成五」→ 管线里 64→256 没有兑现。

**同一个教训两次：探针条件下的结论必须单独在管线上复验。** 第一条让我把本该是
主角的判据关了一整天。

### 所以创新点一真正的题目是候选生成

端到端要动，得把覆盖率从 0.243 提上去，而这需要一个**不依赖类名**的候选生成器。
今天测过的都不够：图像嵌入 0.243、灰度 0.249、纯轮廓 0.119、
边长比例（封顶 0.350，且那个上限本身是循环的）。

## 覆盖率曲线：卡住的不是候选生成，是大池子里的判别力

端到端在非循环条件下 ~2%，我据此说「候选生成是真正卡住的那一环」。
**量完曲线之后这个说法要改。** 纯图像检索、只算良定义节点：

| 候选数 | 16 | 32 | 64 | 128 | 256 | 512 | **1024** | 2048 |
|---|---|---|---|---|---|---|---|---|
| 覆盖率 | 0.141 | 0.178 | 0.243 | 0.324 | 0.470 | 0.643 | **0.854** | 0.962 |

**曲线一路陡升，没有早平**——图像嵌入把正确资产排得并不差（中位名次 317），
前 1024 名能捞到 85%。

**所以真正的问题是：候选池一放大，判据就挑不出来。** 实测非循环条件下残差判据
64→256，命中从 0.023 掉到 0.011。

    端到端 = 覆盖率(k) × 池内准确率(k)

**池内准确率掉得比覆盖率涨得快**，乘积在 k≈64 附近见顶，而顶只有 ~2%。

### 题目要重新表述

需要的是**一个在 k≈512~1024 时池内准确率不崩的判据**（那里覆盖率 0.64~0.85），
而不是「更好的检索器」。今天测过的判据（ICP 残差、轮廓 IoU、Hu 矩、边长比例）
都是在 k=64 的尺度上设计和调过的。

这也解释了为什么今天所有候选生成方向（灰度、纯轮廓、多帧、边长比例）收益都很小
——**候选生成本来就不是最紧的那一环**。

方法论：**乘积式指标上说「A 是瓶颈」要先量两条曲线**，因为瓶颈会随工作点移动。

## 把差距算成倍数：非循环条件下判据等于随机

覆盖率曲线 + 实测端到端 ⇒ 判据相对**池内随机**的提升倍数
（真值资产集合均值 4.83，池内随机 ≈ 4.83/k）：

| k | 覆盖率 | 随机端到端 | 实测（残差，非循环）| 提升 |
|---|---|---|---|---|
| 64 | 0.243 | 0.0184 | 0.017 | **0.9x** |
| 256 | 0.470 | 0.0089 | 0.011 | **1.2x** |

**非循环条件下，判据基本等于随机。**

### 这解释了循环与非循环的落差

循环条件下残差判据 0.344、显著优于轮廓（p=0.0056）；非循环下等于随机。
差别不在判据，在**候选池的构成**：

- **循环**：池子由**文字**驱动（类名=资产文件名），干扰项是「名字像的」；
- **非循环**：池子由**图像**驱动，干扰项是**图像近邻**——同套件、同类形状、
  外观相近，**难得多**。

**所以判据真正的弱点是分不开图像近邻**，而那正是检索把正确答案放进池子时
顺带带进来的那一批。

### 要多强才够

端到端 30% 的话：k=512 需 **49 倍**于池内随机，k=1024 需 **74 倍**。
**现状 ~1 倍。** 这是一个半数量级的差距，不是调参能补的。

**方法论**：把「还差多少」算成**相对随机的倍数**，比说「还不够好」有用——
它直接区分「调参问题」和「换方法问题」。

## 2x2 填完：几何完美也救不了图像近邻，修几何这条关闭

| | VGGT 几何 | 真值几何 |
|---|---|---|
| 循环池（干扰项=名字像的）| 0.344 | 0.413 |
| **非循环池（干扰项=图像近邻）** | **0.018** | **0.018** |

**完全相同。即使几何完美，面对图像近邻干扰项，判据依然等于随机**
（池内随机是 0.0184）。

之前只在循环池上验过「几何质量无所谓」，而那里的干扰项是名字像的，
区分它们不需要精细形状。补上缺的那一格之后，**「修几何」整条方向关闭**。

### 创新点一的最终表述

> 用连续稠密几何构造的判据（ICP 残差），在干扰项与目标**外观差异明显**时
> 能有效做离散选择——0.344，显著优于二维轮廓判据（p=0.0056），**不需要梯度**；
> 但在干扰项是**视觉近邻**时等于随机（0.018 对池内随机 0.0184），
> **且这与几何质量无关**（真值几何同样 0.018）。

这不是输入质量问题，是**判据本身分不开视觉近邻**。而视觉近邻正是任何图像检索
把正确答案放进候选池时必然一起带进来的东西——**判据面对的永远是最难的那组**。

### 方法论

「输入质量是瓶颈」这个假设**要在最难的那组样本上验**。我在容易的那组上验完
「几何无所谓」，差点据此放弃修几何；真正该问的是难样本上几何要不要紧，
答案同样是不要紧，但理由完全不同——而理由决定下一步。

## 多帧融合也无效：两个「输入不足」的解释都排除了

真值几何无效之后留下一个反常现象：**完美的深度为什么分不开三维形状不同的
资产？** 查 `residual_pick` 发现它 `for q in queries: icp(q, model)`——
**逐帧比，从不融合**。每帧只有那个视角可见的那片表面，而「图像近邻」按定义
就是**可见表面长得像**的资产。这同时解释了真值几何为什么救不了：
**完美的部分表面仍然是部分的。**

加 `residual_fuse`（多帧 vstack 成一个模型再比），非循环池、114 个配对物体：

| | 命中 | 同套件 |
|---|---|---|
| 逐帧 | 2/114 = 0.018 | 0.202 |
| 融合 | 2/114 = 0.018 | 0.272 |
| McNemar | 1 对 1，p=0.480 | 11 对 3，p=0.061 |

**命中完全没动**，而开关是活的——**40.4% 的决策改了选择**，改完一样不对。

### 两个「输入不足」的解释都排除了

| 解释 | 检验 | 结果 |
|---|---|---|
| 几何质量不够 | 真值几何 | 0.018 对 0.018 |
| 信息不完整（部分表面）| 多帧融合 | 0.018 对 0.018 |

**判据拿到完美几何、拿到完整表面，面对视觉近邻依然等于随机。问题不在输入，
在判据形式本身。**

一个细节：同套件 0.202 → 0.272（p=0.061）——融合让它更接近正确套件，
但仍挑不中正确那一个，和残差判据在循环池里「粗类别对、精确身份错」同型。

### 创新点一的完整表述

> 用连续稠密几何构造的 ICP 残差判据，在干扰项**外观差异明显**时能有效做离散
> 选择（0.344，显著优于二维轮廓，p=0.0056，**不需要梯度**）；面对**视觉近邻**
> 时等于随机（0.018 对池内随机 0.0184），**且这与几何质量和信息完整性都无关**。

要继续，必须换判据的**形式**，而不是改善它的输入。

## 路线一关闭：尾部判据也失败，判据形式这一维试尽

用残差分布的 p90 代替均值（理由：`icp()` 先扔掉离群点再取均值，对局部差异双重
失明，而视觉近邻恰恰只在局部不同）。非循环池、40 条片段：

| | 命中 |
|---|---|
| 均值（现状）| 2/110 = 0.018 |
| **p90 尾部** | **0/114 = 0.000** |

**更差。** 至此路线一的两个维度都试尽：

| 维度 | 试过的 | 结果 |
|---|---|---|
| **输入** | 几何质量（真值几何）、信息完整性（多帧融合）、候选池大小 | 全部 0.018，无变化 |
| **形式** | 均值残差、p90 尾部、轮廓 IoU、Hu 矩、边长比例、图像嵌入 | 全部等于或低于随机 |

**结论：在视觉近邻上做离散资产选择，当前所有判据都不可用，而且不是输入的问题。**
这条写进论文的限制条件，不作为贡献（见 never-ship-negative-results-as-the-paper）。

## #84 查清楚了：尺度误差在片段之间，评测口径不是杠杆

> **前提**：下面所有数来自 `rrf3/arm_off`，而这批跑的是**真值分割**后端
> （`masks_index.json` 的 `backend: gt`）。也就是说分割是完美的、没有幻觉物体，
> 尺度误差纯粹来自深度和标定。真实感知下只会更差，**这些数是上限不是现状**。

2026-10-25。#84 原本的说法是「`assumed_camera_height_m = 1.2` 这个常数和数据集对不上」。
量下来这个说法对，但推论错了。

### 评测口径换不换，几乎不动指标

刚性口径（尺度被 1.2 米假设定死）对相似口径（尺度无关），60 条片段：

| 口径 | 轨迹误差中位 | 运动类型准确率 | 配对物体 | 物体召回 | 物体精确率 | 形状对 |
|---|---|---|---|---|---|---|
| rigid | 0.560 m | 0.135 | 251 | 1.000 | 0.857 | 0.422 |
| similarity | 0.512 m | 0.132 | 250 | 1.000 | 0.845 | 0.416 |

轨迹误差只差 8.6%，其余基本不动。**「把评测改成尺度无关」解决不了任何问题。**

### 中位数掩盖了分布

逐物体 scale_ratio 中位 **1.13**，看着没事。实际分布：

| | 值 |
|---|---|
| 范围 | 0.16 – 23.22 倍 |
| 十分位 | 0.26 / 4.81 |
| 落在 0.8–1.25 | **13.5%** |
| 错超过 2 倍 | **60.5%** |

### 真结论：79.4% 的方差在片段之间

把 log(scale_ratio) 做方差分解（51 条片段、242 个物体，每条至少 2 个物体）：

| | 占比 | 散度 |
|---|---|---|
| 片段**间** | **79.4%** | 2.76 倍 |
| 片段**内** | 20.6% | 1.57 倍 |

假想「每条片段各自除掉自己的尺度中位数」（= 完美的逐片段标定）：
落在 0.8–1.25 的比例 **12.8% → 43.8%**，3.4 倍。**这是 #84 的天花板。**

所以 #84 要修的是**逐片段的尺度标定**，不是评测口径，也不是那个常数本身的取值。

### 验过一条因果链，不成立

猜测：尺度错是「物体只占 9 像素」的下游（见下一节），因为十像素宽的物体没有
可靠的深度延展。实测 **Spearman(掩码像素, |log 尺度比|) = -0.110**，
而预测变量跨 1.88 个数量级（58–4452 像素），不是散度不足造成的假阴性。
按像素分四档，|log 尺度比| 中位 0.96 / 0.90 / 0.61 / 0.66 ——
最大那档（中位 1296 像素，比最小档多 14 倍）仍有约 1.9 倍的尺度误差。

**两个问题是独立的，不要讲成一条链。**

### 为什么 `metric_scale` 永远返回 1.0

`gwm/synthesis/scale.py` 的回退机制本身是对的，但在这个数据集上拿不到任何输入：

- 管线产出的类名里只有 **4.4%** 查得到尺寸先验。最常见的是 `rail`(68)、`block`(28)
  ——Kenney 的资产文件名，不是语义类别。**查表这条路天生是死的。**
- `LARGE_OBJECT_M = 0.5`（物体最长边中位 0.21 m）是次要问题：在它生效之前
  `samples` 就已经是空的了。
- 设计里的第一优先来源是**命名器返回的 `size_m`**，但这些配置根本没跑命名器
  （perception 目录下没有 `namer.json`）。**这是唯一还站得住的逐片段锚。**

### 一个顺带发现的死信号

`evidence.py:364` 写 `ob["conf"] = min(1.0, ob["n"] / 400)`。逐物体点云在
`stride=2` 下约 29 个点，于是这个几何置信度在神经感知下约 **0.07**。它被写进证据的
`confidence.geometry`，而**下游没有任何地方读它**。管线自己已经知道「这个物体
我几乎没有点」，但这句话没人听。

想拿它当过检的过滤器（第三档的关节大多装在幻觉物体上），但**现有数据测不了**：
手头所有 60 条跑的都是真值分割后端，那条路径把 `geometry` 硬写成 1.0，
而且根本不产生幻觉物体。要验这个想法得先有一批神经感知的运行。

## 第三档：样本量确实被扔掉了四分之三，但找回来救不了它

2026-10-25。两个独立的干预，同一个结论。

### 起点：物体在模型眼里只有 9 个像素

| 环节 | 实测 |
|---|---|
| 数据集渲染 | 640×360（`render_dataset.py` 默认值）|
| 感知端输入 | 518×294 —— **VGGT 的硬上限，已经顶满** |
| 物体掩码 | 中位 **115 像素**，包围盒短边 **9 px**（3003 个实例）|
| `unproject(stride=4)` | 逐帧中位剩 7.2 个点 |
| `len(pts) >= 30` 的闸 | 只有 17.4% 的掩码实例过得去 |
| 结果 | 真值 73 个关节物体，探针只看到 **15 个** |

**提高渲染分辨率没用**：VGGT 封顶 518，已经顶满。唯一的杠杆是取景。
（生产管线用 `stride=2`，比探针宽松，所以 79.5% 这个损失是**探针侧**的。）

### 两条独立的路都能把样本找回来

**路一：改步长**（零成本，一个环境变量）

| | 处理物体 | 关节进管线 |
|---|---|---|
| stride=4 | 133 | 15/73 = 20.5% |
| stride=2 | 378 | **44/73 = 60.3%** |

**路二：改取景**（`scripts/recamera.py`，只换 camera 字段，场景一字不动）

| 臂 | 掩码实例 | 像素中位 | 短边中位 | stride4 过闸 | 关节物体 |
|---|---|---|---|---|---|
| baseline | 3003 | 115 | 9 | 17.4% | 2 |
| walkthrough | 2040 | **466** | 23 | 49.4% | **11** |
| object_centric | 1650 | **680** | 24 | 55.8% | 7 |

针孔模型的预测兑现了（像素中位 ×5.9）。实例数下降是预料中的代价：近景换像素、赔覆盖。

### 但判据的准确率一点没动

| | 关节上触发 | 非关节上触发 | 区分比 | Fisher p |
|---|---|---|---|---|
| stride=4（n=15）| 1/15 = 6.7% | 3/118 = 2.5% | 2.62 倍 | 0.384 |
| stride=2（n=44）| 2/44 = 4.5% | 14/334 = 4.2% | **1.08 倍** | 0.576 |

**n=15 时看起来的 2.62 倍区分力是噪声**，样本一大就塌到 1.08 倍。
取景那边同向：walkthrough 关节物体 2→11，报出 2 个**对 0 个**。

所以「关节信号太弱」**不是样本量的产物**。样本量确实是个真问题（而且修好了），
但它不是判据失效的原因。

### 绕开阈值：信号到底存不存在

判据只在 4% 的物体上触发，而真值里 25% 是关节——极度保守。用 AUC 绕开阈值（n=378）：

| 字段 | AUC | 关节 n | 备注 |
|---|---|---|---|
| `axis_consistency` | 0.857 | **2** | 绝大多数物体上是 NaN，**不能用** |
| `gated_total_rotation_deg` | 0.680 | 44 | |
| `total_rotation_deg` | 0.676 | 44 | 排列检验 p=0.0001 |
| `travel_m` | 0.616 | 44 | |
| `hinge_fit_ratio` | **0.521** | 44 | 小样本上测到 0.606，样本一大就塌 |

**但 AUC 0.676 是个会骗人的数。** precision@k：

| 取转角最大的前 k | 关节率 | 对基线 0.116 的倍数 |
|---|---|---|
| top-10 | 0.100 | **0.86（低于基线）** |
| top-44 | 0.136 | 1.17 |
| top-88 | 0.182 | 1.56 |

增益全在排序的**底部**。按真值类别看转角中位数，原因一目了然：

| 类别 | n | 转角中位 | |
|---|---|---|---|
| static | 226 | **19.8°** | |
| trajectory | 25 | 41.2° | |
| spin | 15 | 42.9° | |
| revolute | 20 | 52.8° | **关节** |
| periodic_rotate | 15 | 58.6° | |
| prismatic | 24 | 61.9° | **关节** |
| **periodic_translate** | 29 | **75.1°** | **纯平移，却转角最高** |

`total_rotation_deg` 分开的是**动没动**（第二档，已经解决的），不是**是不是关节**。
在会动的物体里面它零信息。而且**纯平移的 `periodic_translate` 解出 75.1° 的假转动，
比真正转动的 `revolute`（52.8°）还高**——这是 `icp-invents-rotation` 那个老问题，
它在毒化整个第三档。

作为**排除器**倒是有效：转角最小的 100 个里只有 2 个关节（2.0% 对基线 11.6%）。

### 瓶颈的新位置

不是样本量、不是像素预算、不是闸的标定、不是转动闸。是 **ICP 在纯平移物体上
解出的假转动，量级压过真正的关节转动**。在这一处修好之前，任何搭在「解出转角」
上的关节判据都不可能工作。

## 坏的是测量，不是判据

把**真值的运动幅度**和**管线测到的运动幅度**并排放。真值来自 60 条片段的
`gt_states.jsonl`（逐帧四元数和位置，整段最大幅度）；测量来自 `stride2/rows.json`
的 378 个物体（过了点数闸的那批，所以是子总体）。

| 真值类别 | 真值转角 | **测到的转角** | 真值位移 | **测到的位移** |
|---|---|---|---|---|
| static | 0° | **19.8°** | 0 m | **0.270 m** |
| prismatic | 0° | **61.9°** | 2.111 m | 0.586 m |
| periodic_translate | 0° | **75.1°** | 2.487 m | 0.828 m |
| revolute | **94.3°** | 52.8° | 0.139 m | 0.482 m |
| periodic_rotate | 80.4° | 58.6° | 0 m | 0.663 m |
| trajectory | 0° | 41.2° | 0.552 m | 0.649 m |
| **spin** | **179.5°** | **42.9°** | 0 m | 0.520 m |

**真值转角跨 0–180°，测出来全挤在 20–75°。** 不动的东西上无中生有
（static 19.8°），大动的东西上严重压缩（spin 179.5°→42.9°）。

> **位移那一列有混淆，不能当压缩率读。** 管线的世界尺度本身就错（上一节：逐片段
> 尺度比散在 0.16–23 倍），「2.111 m 测成 0.586 m」里混了尺度误差。
> **转角那一列不受影响**（无尺度量），**static 上 0.270 m 的虚假位移也不受影响**
> （真值是 0，乘任何尺度还是 0）。所以「无中生有」两个通道都成立，
> 「压缩」只有转角那边是干净的。

**瓶颈不在「判据选得不对」这一层，在测量这一层。** 这解释了三件以前解释不了的事：

1. 为什么判据空间试尽——判据读到的输入本身不携带信息。
2. 为什么样本量翻三倍（15/73 → 44/73）一点用没有。
3. 为什么 `periodic_translate` 解出的假转动（75.1°）比真转 94° 的 `revolute`（52.8°）还高。

### 一个结构性事实：一多半的关节真值转角是 0

73 个真值关节里 **41 个（56%）是 prismatic**，真值转角恰好是 0、位移 2.111 m。
看真值程序的 `motion` 字段：

| 类别 | 字段 | 本质 |
|---|---|---|
| prismatic (41) | `axis, from, schedule` | 沿固定轴**单次斜坡** |
| periodic_translate (36) | `axis, amp, period, phase` | 沿固定轴**正弦振荡** |
| revolute (32) | `axis, pivot, range_deg, schedule` | 绕固定支点单次转动 |
| trajectory (41) | `keyframes, interp` | 任意平滑路径 |

**真值程序里没有父件/依附字段**，prismatic 和 periodic_translate 的区别纯粹是
时间剖面。**而第三档的判据在拟合铰链。** 一多半的目标物体真值转角是 0，
铰链拟合对它们在结构上就不适用。

另外 **`straightness` 是个死字段**：378 个物体里只有 16 个有值，
之前读到的 AUC 是在 n=1~2 上算的。

## 假转动：重构配准这一族全部关闭，驱动量是视点变化

2026-10-25 晚。沿着「ICP 在不转的物体上解出转动」这条，把能改配准结构的办法试尽了。
全部在 `object_centric` 那批数据上（掩码像素中位 680，短边 24 px），
12 条片段、真值分割、真值一起记。

### 三个变量，两个无效一个有效

| 变量 | 做法 | 结果 |
|---|---|---|
| 配准**步数** | geometry_frames 12 / 24 / 48 / 96 | **无效**：假转动 15.15 / 15.06 / 15.85 / 16.68°，解出转角 21.67 / 24.63 / 24.48 / 29.01° |
| **断开累积** | `refine_poses`（全部帧建模型、每帧独立配）| **无效**：24 个静止物体配对，更好 11 更差 13，差中位 **+0.00°**，p=0.84 |
| **固定参照** | 第一帧点云当模板 | **显著更差**：对 gate010 更好 4 更差 20，差中位 +5.02°，**p=0.0015** |
| **相机视点变化** | 三种相机轨迹 | **强效**：Spearman +0.387，n=42，排列检验 **p=0.010** |

前两个互相印证：**根本没有累积，所以断开累积什么都不改变。**
步数翻 **8.6 倍**（11 → 95 步）毫无变化：假转动 ∝ N^0.046，解出转角 ∝ N^0.135。
这一点独立地排除了同一个假说。（n 从 45 涨到 87 是因为帧多了之后更多物体过得了
「≥6 帧」那道闸——这个总体变化只会让大样本那端**更差**，所以它不解释「平」，
反而让「平」更站得住。）

### `refine_poses` 为什么无效——这部分可迁移

它拿**第一遍解出的位姿**去建那个「完整模型」。位姿已经漂了，模型就是一团糊，
每帧再配它只能复现同样的漂移。**它继承了自己要修的那个误差。**

> **通用形态：想用一个「固定参照」去断开累积，那个参照不能由被修的估计量生成。**

而第一帧的点云确实满足这个条件（不依赖位姿估计），但它**本身也是个视角相关的
局部表面**，所以视角一拉开重叠就少，反而更差——这和 `visible_subset` 的注释
预言的一致。

### 收敛到的定量关系

三个相机臂的整段扫掠角（51° / 155° / 334°）对假转动（15.97 / 23.11 / 43.25°）：

> **假转动 ≈ c · (整段视点变化)^0.53** —— 一个平方根律。

**而且它对「怎么采样、怎么串联这同一段观测」完全不敏感。**
所以假转动不是算法的实现缺陷，是「从不同视角看到的**局部表面**之间做配准」
这件事本身的内禀模糊性。

### 唯一有效的东西仍然是转动闸

| 臂 | 静止假转动 | 解出转角 | 真转动那侧 |
|---|---|---|---|
| gate_off | 22.93° | 36.01° | spin 真值 67.89 → 解出 48.85，err 15.95 |
| **gate010** | **16.54°**（降 28%）| **29.25°** | spin → 41.16，err **15.95**（没塌）|

这次是在 680 像素、真值一起记、n=24 的条件下复现的，比第一次那个 A/B 更硬。

### 对第三档的含义

位姿这一级的判据空间和**实现空间**现在都试尽了。还能动的只剩两处：

1. **让每帧的观测不再是视角相关的局部表面**（完整性论点）。唯一验证有效的完整
   模板是真值网格，而它要靠资产检索——那条路已经关闭（见前文路线一）。
2. **换掉「先解位姿、再判关节」这个问题形式**，不再需要解位姿。
