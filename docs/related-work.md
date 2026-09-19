# 相关工作

> 按类别整理的相关工作与本课题的关系，供写作与对比使用。条目来自 RP 原有清单与 2026-09-18 的调研（`../survey/01`、`04`、`05`）；标注"未核实"的是调研时未能从一手来源确认的信息。

## 0. 定位

现有工作四支各占一角，没有系统走通"单目视频 → 带时序与动力学的场景程序 → Web 引擎"：

| 支线 | 代表 | 有什么 | 缺什么 |
|---|---|---|---|
| 视频 → 几何资产 → 引擎 | Video2Game、HoloScene、Vid2Sim、Marble / HY-World 2.0 的视频输入 | 可执行、可玩 | 产物是 mesh / 3DGS / 碰撞体等数据，逻辑是固定模板；静态场景，不建模视频中的运动 |
| 视频 → 符号程序 | Guzdial 2017、FAE、Mechanic Maker、PoE-World | 真正输出解释动力学与规则的程序 | 限于 2D 像素游戏，无 3D、无 Web 运行时 |
| 文本 / 图 → 仿真代码 | ChronoAgentic、SimuScene、VDAWorld、Mage 的中间表示路线 | 可执行仿真程序；Mage 证明中间表示优于直接出代码 | 输入不是视频 |
| 视频 → 结构化语言 | SceneScript | 3D、语言程序式 | 仅静态布局，无动力学 |

"4D（动态）"与"程序化表示"两环同时缺失，是本课题的位置。

## 1. 以程序为场景表征（inverse graphics）

| 工作 | 要点 | 关系 |
|---|---|---|
| IG-LLM（2024） | 单图到图形代码，LLM 解码 CLIP 特征 | 单图、静态 |
| SceneCraft（ICML 2024）、3D-GPT、Holodeck（CVPR 2024）、LayoutGPT | 文本到 Blender 或引擎场景程序 | 文本输入 |
| SceneScript（ECCV 2024） | 第一人称视频经点云到布局级结构化语言 | 视频输入但仅静态布局 |
| SEIG / Thinking in Blender（2026.06） | 单图到 Blender 程序，分阶段解码，checklist 式验证 | 分阶段解码与验证方式直接借鉴 |
| Scenix（2026.08） | 稀疏视角到可执行场景程序，生成资产实例化，闭环空间精修，XScene 约 11 万样本 | 静态部分对齐的对象；本课题重点在动态与时序 |
| The Scene Language、ShapeAssembly、Learning to Infer 3D Shape Programs | 程序化表征 | 表征思想 |
| SpatialGrammar（2026.04） | DSL + 编译期约束反馈，小模型接近大模型 | 校验先于渲染 |
| SceneCode（2026.05）、SimWorlds（2026.07） | 含关节元数据的可执行室内世界程序；4D Blender 程序 + 确定性验证器 | 最接近"4D 程序"的近期工作，输入为文本 |

## 2. 代码 / 状态式世界模型与程序合成

| 工作 | 要点 | 关系 |
|---|---|---|
| WorldCoder（NeurIPS 2024） | 交互轨迹到 Python 世界模型 | 输入非视频 |
| Code World Models for General Game Playing（2025.10） | 自然语言规则 + 轨迹到 Python 状态机 + MCTS，十款棋牌 | "游戏 = 代码世界模型" |
| Code World Model: Coding Agent as World Brain（2026.08） | 代码式世界模型 | RP 1.1 的路线代表 |
| From Pixels to States（2026.07） | 以引擎状态为中心的世界模型分析 | 动机 |
| PoE-World（2025.05） | 少量示范到程序专家乘积世界模型 | 从观测合成可执行程序 |
| Game Engine Learning from Video（IJCAI 2017）、FAE（Amii 2025） | 2D 游戏视频到前向仿真规则 / DSL 程序 | "视频 → 程序"最直接的先例，但 2D |
| Mechanic Maker（AIIDE 2024） | 用户示范到机制程序 | 从示例学规则 |
| ChronoAgentic（2026.05） | 文本到 PyChrono 仿真代码；PhyWorldBench 82.5% 对视频生成 52.5% | 论证"生成仿真程序优于生成视频" |
| SimuScene（2026.02）、VDAWorld（2025.12） | 文本 / 图到可执行物理抽象 | 文本或图输入 |
| GAVEL（NeurIPS 2024） | Ludii 描述语言 + 演化 + LLM 生成棋类规则 | "游戏 = 程序"最纯粹的例子 |
| MarioGPT（NeurIPS 2023）、Word2World（2024） | 文本到关卡 / 2D 世界 | 文本到可玩规格 |

## 3. 视频到可执行 / 可仿真

| 工作 | 输入 → 输出 | 运行时 | 开源 | 差距 |
|---|---|---|---|---|
| **Video2Game**（CVPR 2024） | 视频 → NeRF → 神经纹理 mesh + 凸分解碰撞体 → 浏览器游戏 | three.js + cannon.js | MIT | 最接近；输出是资产，逻辑手写；静态 |
| **HoloScene**（NeurIPS 2025） | 视频 → 交互式场景图（mesh + 3DGS + 物理参数 + 层级 / 接触） | UE 第三人称 demo | 是 | 场景图已是半结构化程序；非 Web、无行为规则、静态 |
| Vid2Sim（2025.01 / 2025.06） | 单目视频 → 数字孪生 / 物理参数 | — | 部分 | 面向机器人 |
| PhysGen（ECCV 2024） | 单图 → 物理理解 → 刚体仿真 → 视频 | 自研 | 是 | "图 → 显式仿真器 → 渲染"三段式 |
| Game2World Engine（2026.08） | 野外游戏视频（303 款）→ 去 HUD 训练数据 | — | 将开源 | 视频仅作训练数据 |

## 4. 视频式世界模型

| 工作 | 要点 | 开源 |
|---|---|---|
| GameNGen（2024.08） | 动作 + 历史帧 → DOOM 帧 | 否 |
| Genie 2 / Genie 3 / Project Genie（2024.12 / 2025.08 / 2026.01） | 单图或文本 → 可玩世界，逐帧生成，官方明确非 3DGS | 否 |
| Oasis / open-oasis（2024.10） | 键鼠 → Minecraft 帧 | 500M MIT |
| GameGen-X（ICLR 2025）、GameFactory（ICCV 2025） | 文本 / 图 + 控制 → 游戏视频；GF-Minecraft 数据集 | 是 / 数据集 |
| WHAM / Muse（2025.02）、PlayGen（2024.12） | 帧 + 动作 ↔ 帧 | 权重开源 |
| Matrix-Game 2.0 / 3.0（2025 / 2026.03） | 图 + 键鼠 → 分钟级 25~40 fps | 全开源 |
| Hunyuan-GameCraft 1.0 / 2（2025.08 / 2025.11） | 单图 + 键鼠 → 视频；2 支持自然语言指令 | 1.0 开源 |
| Yan（2025.08） | Sim / Gen / Edit 三件，显式解耦机制仿真与渲染 | 未开源 |
| HY-WorldPlay（2025.12）、Odyssey、Runway GWM-1 | 实时视频世界模型 | 权重开源 / 否 / 否 |

共同局限：无持久可编辑状态，不可导出到引擎，不是程序。

## 5. 显式 3D 世界生成

| 工作 | 输入 → 输出 | 运行时 | 开源 | 关系 |
|---|---|---|---|---|
| HunyuanWorld 1.0（2025.07） | 文本 / 图 → 全景 → 分层 mesh | Web viewer；Unity / UE | 是 | 图 → 可探索 mesh 世界 |
| HunyuanWorld-Voyager（2025.09） | 图 + 相机轨迹 → RGBD 视频 + 实时 3D 重建 | — | 是 | 视频 ⇄ 3D 桥 |
| **HY-World 2.0**（2026.04） | 文本 / 单图 / 多视图 / **视频** → 3DGS、mesh、点云、深度、相机 | Blender / Unity / UE / Isaac | 是 | 视频 → 3DGS/mesh 的开源方案 |
| WorldGen（Meta，2025.11） | 文本 → LLM 程序化布局 + 扩散 3D + navmesh | 游戏引擎 | 代码未核实 | 唯一显式产出 navmesh 的文本 → 世界 |
| **Marble + Spark**（World Labs，2025.11 / 2026.04） | 文本 / 图 / **视频** → 3DGS + 碰撞 GLB；Spark 是 MIT 的 three.js 3DGS 渲染器 | three.js | Spark 开源；Marble 商用 | 与本课题运行时完全一致，缺动力学与程序 |

## 6. agentic 游戏开发与基准

| 工作 | 要点 | 关系 |
|---|---|---|
| **OpenGame / GameCoder-27B**（2026.04，Apache-2.0） | 文本 → 完整 Web 游戏；模板 skill + 调试 skill；单 agent 六阶段；有界修复第三轮平台期 | 首个开源 agentic 文本 → three.js / Phaser 框架；模板库 +5.8 |
| **Mage**（2026.05） | 文本 / 结构化中间表示 → Unity C#；直接出代码机制 F1 约 0.12，经中间表示大幅提升 | 支持"先出场景程序再出代码" |
| PlayCoder（2026.04）、Play2Code（2026.05） | 可玩性与执行成功率分离；GUI agent 边玩边改 | "编译通过不等于可玩" |
| AVR-Agent（2025.08） | 文本 + 资产库 → JS 游戏；录像评审 | 迭代优于一次生成；反馈可操作性是瓶颈 |
| AutoUE（ACL 2026）、SPRITE（2026.03）、Sketch2Scene（2024.08） | UE 生成；UI 截图 → 资产；草图 → 可玩 3D 场景 | 图片输入的代表 |
| GameGPT（2023）、ChatDev、MetaGPT | 早期多角色框架 | 参考 |
| **GameDevBench**（ICML 2026，Godot 4，333 任务） | 视觉反馈稳定提分；最佳 68.8% | 评测方法 |
| **GameCraft-Bench**（2026.06，Godot 4，140 任务） | 端到端整游戏 + 键鼠轨迹回放 + 多模态评审；最佳 41.5% | 试玩验收方法 |
| GameXpert-Bench（2026.08）、WorldCoder-Bench（2026）、OpenGame-Bench | 三轨任务；浏览器 3D 世界 StateProbe；headless + VLM 评分 | 状态 schema 固化的依据 |
| Knowledge-Conditioned Single-Pass Unity（2026.07） | 26 个玩法概念单次生成 0 个编译通过 | 反面证据：无迭代不可行 |

## 7. 多智能体有效性

Cemri et al., Why Do Multi-Agent LLM Systems Fail（NeurIPS 2025）；How Generation Architecture Shapes Code Complexity（2026.06）；AgentCoder（2023）；Anthropic 多智能体研究系统博文；SceneConductor（2026.06）、VULCAN（2025.12）。结论与数字见 `feedback.md` 第 7 节。

## 8. 视觉反馈与渲染奖励

SceneCraft、BlenderAlchemy（ECCV 2024）、BlenderGym（CVPR 2025）、SEIG、Scenix、Agentic 3D Scene Generation with Spatially Contextualized VLMs（2025）、SceneWeaver（NeurIPS 2025）、SceneAssistant（2026.03）、Design2Code、UI2Code^N（2025.11）、VF-Coder（2026.04）、ChartMimic、Visual-SDPO（2026.06）、RLRF（NeurIPS 2025）、cadrille（ICLR 2026 Oral）、CAD-RL、3DCodeBench（2026.06）。机制表见 `feedback.md` 第 8 节。

## 9. 感知与重建

| 类别 | 工作 |
|---|---|
| 动态视频位姿与深度 | ViPE（NVIDIA 2025.08）、VGGT（CVPR 2025）、VGGT-Omega（CVPR 2026）、VGGT-Long、VGGT4D、MoRe（CVPR 2026）、MegaSaM（CVPR 2025）、MonST3R（ICLR 2025）、CUT3R、Easi3R、STream3R（ICLR 2026）、Pi3 / Pi3X、Depth Anything 3、Video Depth Anything、MapAnything（3DV 2026）、Uni4D、Geo4D |
| 分割、跟踪、检测 | SAM 2 / 3 / 3.1、Grounded-SAM-2、DEVA、Cutie、Grounding DINO 1.5/1.6、OWLv2、Florence-2、CoTracker3、TAPIR / BootsTAPIR、Track-On2、SpatialTrackerV2 |
| 关节估计 | Articulate-Anything（ICLR 2025）、Real2Code、URDFormer、ArtGS、VideoArtGS、REACTO、Articulation in Prime（2026.05）、Track Articulate Act（2026.09）、Particulate（2025.12）、PAct、ArtLLM、URDF-Anything+、MonoArt（2026 Q1） |
| 4D 重建 | MoSca（CVPR 2025）、Shape of Motion（ICCV 2025）、4DGaussians、Dynamic 3D Gaussians、Free4D、GEN3C、Mesh4D（2026.01） |

## 10. 资产

Objaverse / Objaverse-XL / Objaverse++；Holodeck 的 CLIP + SBERT + 几何检索；Holodeck 2.0 改为 Hunyuan3D 生成；Scenix 的 mask + 描述合成参考图再生成；OpenShape、Uni3D；TRELLIS / TRELLIS.2；Hunyuan3D 2.x / Omni / Part；TripoSG；Step1X-3D；Direct3D-S2；PartCrafter；SAM 3D Objects；AssetGen（2026.05）。细节见 `assets.md`。

## 11. 产品与内核

Rosebud AI 与其开源内核 Roseblox；Bitmagic；Bezi；Astrocade；Roblox Cube 3D / 4D（Cube 3D 开源）；Unity AI；Godot MCP 插件生态；VibeGame；fly.pieter.com 等 vibe coding 游戏。用途见 `runtime.md` 第 7 节。
