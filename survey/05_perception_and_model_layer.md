# 调研 05：感知栈与模型层选型（2026-09-18）

> 由调研 agent 生成；标注 **未核实** = 未能从一手来源确认。硬件假设：L40S 48 GB ×1–4/节点（无 NVLink），vLLM，另有前沿 API。Claude 模型 ID 与定价已对照官方参考核对（`claude-opus-5` $5/$25，`claude-fable-5-1` $10/$50，`claude-sonnet-5` $2/$10，单位为每百万 token）。

## 0. 结论速览

| 层 | 基线主选 | 备选 | 理由 |
|---|---|---|---|
| A1 位姿+内参+深度+静/动掩码 | **ViPE**（NVIDIA，Apache-2.0，v1.2.0 2026/06） | **VGGT-Omega**（Meta，CVPR 2026，纯前馈）；MegaSaM（Apache-2.0） | ViPE 专为动态随手视频设计，一次输出内参/位姿/近度量深度/动态掩码，3–5 FPS，内存有界；VGGT-Omega 秒级前馈、500 帧 43 GB，但 license 与掩码输出待确认 |
| A2 实例分割+跨帧 ID+类别 | **SAM 3.1**（2026-03-27，SAM License 允许商用） | Grounded-SAM-2（全 Apache-2.0） | 文本概念提示→所有实例掩码+ID+视频跟踪，一模型替代检测+分割+跟踪 |
| A3 轨迹 / 3D 框 / 运动分类 | SpatialTrackerV2（3D 轨迹+动静标签）或 CoTracker3；掩码×深度反投影→OBB；螺旋运动拟合分类 | TAPIR（Apache） | 无现成"运动基元分类器"，用 Chasles 螺旋拟合最简 |
| B 程序合成 VLM（基线） | **API：Claude Opus 5**（`claude-opus-5`，1M ctx，128K out）；难例用 Fable 5.1 | 自托管 **Qwen3-VL-32B-Instruct**（FP8，1–2×L40S）或 Qwen3.5-27B | 代码质量+长上下文；Gemini 3.1 Pro 是唯一原生吃视频的 API |
| B' 二阶段程序员（文本） | Qwen3-Coder-30B-A3B（Apache-2.0，2×L40S BF16 / 1×FP8） | DeepSeek-V4-Flash（MIT，284B，未核实能否塞进 4×L40S） | 证据 JSON→three.js 代码 |
| B'' 后续微调对象 | **Qwen3-VL-8B-Instruct** 或 **Qwen3.5-9B**（Apache-2.0，原生视频，1×L40S 推理） | InternVL3.5-8B（MIT） | Scenix 已用 Qwen3.5-4B/9B SFT 出场景程序，是最近似先例 |
| C 训练栈 | SFT：LLaMA-Factory / ms-swift；RL：**EasyR1（verl）** 或 **TRL GRPOTrainer**（异步可调用奖励） | OpenRLHF（HTTP 远程奖励，多轮 VLM RL） | 4×L40S 足以跑 7–9B VLM LoRA-GRPO |

## 1. 动态单目视频的相机位姿 + 深度

| 模型 | 日期 | 动态物体 | 输出 | 速度 / 显存 | License |
|---|---|---|---|---|---|
| **ViPE** https://github.com/nv-tlabs/vipe | 2025-08；v1.2.0 2026/06 | ✅ GroundingDINO→SAM→XMem 传播掩码，掩码取反约束 BA | 内参、位姿、近度量稠密深度、动态掩码；深度先验可选 DA3/MoGe-2 等 | 3–5 FPS @640×480；300 帧约 1–2 min | Apache-2.0（UniK3D 组件 CC BY-NC-SA） |
| **VGGT-Omega** https://github.com/facebookresearch/vggt-omega | CVPR 2026；权重 2026-05 | ✅ 声称静态与动态；未见动态掩码输出 | 位姿+内参、深度+置信度 | 500 帧 43.2 GB → 300 帧@512 L40S 可跑 | **未核实** |
| VGGT (v1) | CVPR 2025 | ❌ | 位姿/内参/深度/点图 | 200 帧 40.6 GB | 非商用 + 申请制商用 |
| VGGT-Long / VGGT4D | ICRA 2025 / 2025-11 | ❌ / ✅ 免训练挖动态掩码 | 分块+回环 / 动态掩码 | – | 跟随 VGGT |
| **MoRe** https://github.com/HellexF/MoRe | CVPR 2026 | ✅ attention forcing 分离动/静 | 位姿、深度、点云、运动掩码；基于 VGGT | 未给出 | **未标注 license** |
| MegaSaM | CVPR 2025 | ✅ | 内参、位姿、视频深度 | ≈0.7 FPS | Apache-2.0 |
| MonST3R / CUT3R / STream3R / Pi3(X) | 2025–2026 | ✅/弱/✅/❌ | 点图、位姿、深度 | 各异 | 均非商用权重 |
| Depth Anything 3 | 2025-11 | ❌ 静态；DA3-Streaming <12 GB | 深度/射线→位姿+内参 | 流式 <12 GB | Small/Base/Metric/Mono Apache-2.0；Large/Giant CC BY-NC |
| MapAnything | 3DV 2026 | ❌ | 度量点图/深度/位姿；可注入已知内参 | – | 有 Apache-2.0 版 |
| SpatialTrackerV2 | ICCV 2025 | ✅ 逐轨迹动静标签 | 视频深度、位姿、3D 轨迹 | 比优化法快 50× | CC BY-NC 4.0 |
| Uni4D | CVPR 2025 | ✅ | 组合 UniDepth+CoTracker3+Grounded-SAM-2 能量最小化 | 分钟级 | MIT |

**推荐**：主选 ViPE（工程最稳、Apache-2.0、显式动态掩码、支持 DA3 深度先验；缺点 BA 非纯前馈）。备选 VGGT-Omega（纯前馈秒级；需叠加 SAM 3 掩码；license 待查）。失败退回 MegaSaM。非商用权重的 MonST3R/CUT3R/STream3R/Pi3 不进基线主路径。

## 2. 物体级 4D 证据

- **分割/跟踪**：SAM 3 / 3.1 https://github.com/facebookresearch/sam3 （2025-11 / 2026-03；848M；文本短语/点/框/示例提示→所有实例掩码+ID+视频跟踪；SAM License 允许商用，HF gated）。备选 Grounded-SAM-2（SAM 2.1 + Grounding DINO 1.0，全 Apache-2.0）。
- **开放词汇检测**：Grounding DINO 1.5/1.6 与 DINO-X 为 API-only；OWLv2（Apache）、Florence-2（MIT）、Qwen3-VL 自带 grounding。基线建议：VLM 对首帧列候选名词短语 → 喂 SAM 3 做概念分割，免去独立检测器。
- **点跟踪**：CoTracker3（CC BY-NC）；TAPIR/BootsTAPIR（Apache-2.0）；Track-On2；SpatialTrackerV2 直接给 3D 轨迹+动静标签。
- **3D 框提升**：SAM 3 掩码 × ViPE 深度 → 反投影到世界系 → 去离群 → 地面法向对齐后 PCA 得 OBB → 沿轨迹 SE(3) 平滑。
- **运动分类 / 关节估计**：Articulate-Anything（ICLR 2025，文本/图/视频→URDF）；VideoArtGS / ArtGS / FreeArtGS；REACTO；**Articulation in Prime**（2026-05，单段随手视频→基元部件+revolute/prismatic 类型+轴，代码未核实）；**Track, Articulate, Act**（2026-09-16，单目视频→关节类型/轴/状态轨迹，核心是稠密 3D 轨迹刚性聚类，与 DSL 最契合，代码未核实）。

**推荐最小证据栈**：ViPE → SAM 3.1 → SpatialTrackerV2 或 CoTracker3 → 每实例逐帧 OBB → 对 OBB 的 SE(3) 序列做 Chasles 螺旋拟合：|旋转|≈0 → prismatic；轴固定且沿轴平移≈0 → revolute；相位自相关 → periodic；其余 → free-rigid；接触 = 掩码底面与静态深度的距离阈值。

## 3. 4D 重建基线（残差层 / 对比）

| 方法 | 输出 | three.js 可导出 | License |
|---|---|---|---|
| MoSca (CVPR 2025) | 4D 运动脚手架 + 高斯 | 需自写 PLY dump | MIT |
| Shape of Motion | 3DGS + SE(3) 运动基 | 可按帧导出（小改） | MIT |
| 4DGaussians (hustvl) | HexPlane 形变高斯 | ✅ 逐时刻 PLY 脚本 | Apache-2.0 |
| GEN3C (NVIDIA) | 相机可控视频 | 视频，非资产 | 代码 Apache-2.0 |
| 2026 video→4D（4DNeX、RiGS、C4G、Mesh4D、Stream4D） | – | – | 均未核实 |

three.js 渲染：**Spark**（MIT，v2.2.0）首选；残差层建议静态背景一份 splat + 每个动态实例一份 splat，由 DSL 程序驱动变换。

## 4. 程序合成 VLM

**(a) API**：Claude Opus 5 `claude-opus-5`（1M / 128K，$5/$25）；Claude Fable 5.1 `claude-fable-5-1`（$10/$50）；Claude Sonnet 5 `claude-sonnet-5`（$2/$10）；GPT-5.5 / 5.6（细节未核实）；Gemini 3.1 Pro（2026-02，原生视频输入）。

**(b) 开源 VLM**（vLLM 支持视频 = ✅）

| 模型 | 日期 | 尺寸 | License | 视频 | L40S 配置 |
|---|---|---|---|---|---|
| Qwen2.5-VL | 2025-01 | 7B/32B/72B | Apache-2.0（72B Qwen license） | ✅ | 7B→1×；32B→2×；72B→4× |
| **Qwen3-VL** https://github.com/QwenLM/Qwen3-VL | 2025-09~10 | 2B/4B/8B/32B；30B-A3B/235B-A22B MoE | Apache-2.0 | ✅ 原生（2 fps，时间戳对齐） | 8B→1×；32B→2×（FP8 1× 紧）；30B-A3B FP8→1× |
| **Qwen3.5**（原生多模态） | 2026-02 | 0.8/2/4/9/27B；35B-A3B/122B-A10B | Apache-2.0 | ✅（需 vLLM main） | 9B→1×；27B→2× BF16 或 1× FP8 |
| InternVL3.5 | 2025-08 | 1–38B；MoE | MIT（代码） | ✅ | 8B/14B→1×；38B→2× |
| GLM-4.6V / Flash | 2025-12 | 106B-A12B / 9B | MIT | ✅ 128K 长视频 | 9B→1× |
| Gemma 3 27B / Gemma 4 | 2025-03 / 2026 | 27B / 2–31B | Gemma 条款 | Gemma 3 仅图像 | 27B→2× |
| Kimi-VL / MiMo-VL-7B | 2025 | 16B-A3B / 8B | MIT | 部分 | 1× |
| Molmo 2 | 2025-12 | 8B/4B | Apache-2.0 | ✅ 视频指点+跟踪 | 1×（适合当证据器） |

**(c) 代码专用二阶段**：Qwen3-Coder-30B-A3B（Apache-2.0，256K）；DeepSeek-V3.2 / V4-Flash（MIT，文本-only）。

**推荐**：基线用 Claude Opus 5（8–16 关键帧 + 证据 JSON + DSL 语法 → 程序；难例切 Fable 5.1；想原生吃视频则 Gemini 3.1 Pro）。自托管基线 Qwen3-VL-32B-Instruct-FP8（1–2×L40S）。后续 SFT/RL 对象 Qwen3-VL-8B-Instruct 或 Qwen3.5-9B。可选二阶段：VLM 出场景规格 → Qwen3-Coder-30B-A3B 出 three.js。

## 5. 训练栈（SFT + RL，非可微可执行奖励）

| 框架 | 多模态 | 算法 | 自定义奖励 | 备注 |
|---|---|---|---|---|
| LLaMA-Factory | Qwen3-VL 等 | SFT/DPO；LoRA | — | SFT 首选 |
| ms-swift | 300+ MLLM | GRPO/DAPO/GSPO/… | 奖励插件 | 2026-09 活跃 |
| **TRL GRPOTrainer** | ✅ VLM | GRPO | 可调用/异步协程奖励函数 | 最易接入 three.js 渲染服务 |
| verl | ✅ | PPO/GRPO，LoRA RL | `custom_reward_function` | vLLM/SGLang rollout |
| **EasyR1** https://github.com/hiyouga/EasyR1 | Qwen2/2.5/3-VL | GRPO/DAPO/… | 自定义奖励 python | 7B：全参 4×40 GB 或 LoRA 2×32 GB |
| OpenRLHF | ✅ 含 Qwen3.5；多轮 VLM RL | PPO/GRPO/RLOO | HTTP 远程奖励 | Apache-2.0 |

执行/渲染奖励的 RL 先例：RLRF（NeurIPS 2025，SVG 渲染比对）、cadrille（ICLR 2026 Oral，GRPO + 执行 CAD 后 IoU）、CAD-RL、SimWorlds（2607.01766，4D Blender 程序 + 确定性验证器）、SceneCode（2605.19587，含关节元数据的可执行室内世界程序）。

**4×L40S 可行性（7–9B VLM + LoRA + GRPO）**：可行。2 卡 vLLM rollout（FP8 策略）+ 2 卡 FSDP/LoRA 训练；奖励服务用 Node + Playwright 渲染 three.js（CPU 或 1 卡）；每 prompt 8–16 帧 ×≈300 视觉 token ≈5–10k token，组大小 4–8。无 NVLink 下尽量 1 卡放整个策略。

## 6. 后续 SFT 数据集

| 数据集 | 类型 | 规模 / 标注 | 可用性 |
|---|---|---|---|
| **XScene**（Scenix，2026-08） | 室内静态场景程序 | ≈110k 场景 / 560k 图；程序 = 房间蓝图 + 物体（类别/中心/尺寸/朝向/支撑） | 发布未核实 |
| **Kubric MOVi-A..F** | 合成动态多物体视频 | 每集 ≈9.75k；24 帧/2 s；GT：实例分割、深度、光流、逐帧相机、物体位姿/速度/3D 框；E/F 相机线性运动 | `tfds.load("movi_e", data_dir="gs://kubric-public/tfds")` |
| **OmniWorld-Game** | 游戏引擎录制 | 96k 段 1280×720@24fps；深度、位姿、光流、前景掩码 | HF `InternRobotics/OmniWorld`；license 未核实 |
| ViPE 标注集 | 真实/生成视频伪 GT | DynPose-100K++ 99.5k 段；Wild-SDG-1M | HF nvidia/* |
| GF-Minecraft（GameFactory） | 真实游戏 + 引擎动作 | 70 h；键鼠/相机 delta | HF |
| PhysGame | 游戏物理故障 QA | 880 段 | 公开 |
| ProcTHOR-10K / Infinigen / 3D-FRONT | 静态室内生成器/库 | – | 部分非商用 |
| CARLA（MIT） | 可自录位姿/深度/分割/3D 框 | – | 公开 |

**数据建议**：动态短片 GT 首选 Kubric MOVi-E/F（相机移动 + 1–3 个抛掷动态物体，匹配设定）与 OmniWorld-Game；MOVi 的物体位姿/速度序列可直接编译成 DSL 程序做 SFT；室内静态布局借鉴 XScene 的程序 schema。

## 未核实清单
VGGT-Omega 与 MoRe 的 license；VGGT-Omega/VGGT4D 是否输出动态掩码；Pi3X/DA3/SAM3 在 L40S 上的实测显存；GPT-5.6 日期与 OpenAI 视频输入；Gemma 4 视频支持；Qwen3-VL-235B / DeepSeek-V4-Flash INT4 是否塞进 4×48 GB；XScene/OmniWorld/GF-Minecraft 的许可与下载。
