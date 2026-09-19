# 调研 01：「单句 / 单图 / 视频 → 可玩游戏或可探索 3D 世界」相关工作（2023–2026.09）

> 由调研 agent 生成，调研日期 2026-09-18。所有条目均至少有一个网页来源；无法核实的字段标 **未核实**。

## 0. 一句话结论

- **A 类（代码/引擎生成）** 已从 2023 年的多智能体 demo（GameGPT、ChatDev）走到 2026 年的 agentic game dev：开源框架（OpenGame）、真实引擎基准（GameDevBench / GameCraft-Bench / GameXpert-Bench，基于 Godot 或 Web）、商用产品（Rosebud、Bitmagic、Unity AI、Roblox Cube 4D）。**输入几乎全是文本**，图片输入仅限 UI 截图/草图（SPRITE、Sketch2Scene），**没有以视频为输入的代码生成工作**。
- **B 类（神经世界模型）** 2025–2026 爆发（Genie 3、Matrix-Game 3.0、HY-WorldPlay、GameCraft-2、Odyssey、GWM-1），但输出是逐帧视频，无持久可编辑状态；另一支显式 3D 世界生成（Marble、HY-World 2.0、HunyuanWorld 1.0、WorldGen）输出 3DGS/mesh，可进引擎，但没有游戏逻辑/动力学程序。
- **最贴近"单目视频 → 可执行 4D 场景程序 → three.js 游戏"的工作**是 Video2Game（CVPR 2024，视频→NeRF→mesh+碰撞体→three.js+cannon.js）和 HoloScene（NeurIPS 2025，视频→带物理参数的交互式场景图→UE 游戏）；两者输出的是数据资产而非程序。"从视频学出可执行程序"仅存在于 2D 像素游戏的程序合成线（Guzdial 2017、FAE 2025/26、PoE-World）。**"视频 → 带动力学/规则的符号化场景程序 → Web 引擎"目前没有已发表系统**，是明确空白。

## 1. A 类：代码 / 引擎生成

### 1.1 商业产品

| 项目 | 输入 → 输出 | 引擎/运行时 | 开源 | 可编辑/持久状态 | 关系 |
|---|---|---|---|---|---|
| **Rosebud AI**（2023–） | 文本 → JavaScript 游戏代码 + 资产 | Phaser（2D）/ three.js（3D） | 否（内核 Roseblox 开源） | 是 | 与目标运行时一致的"文本→代码"参照 |
| **Bitmagic**（Steam EA） | 文本 + 图片 → 3D 游戏 | 自研 | 否 | 是 | 图片→3D 游戏的商业先例 |
| **Bezi**（Unity 内 AI agent） | 文本 → Unity C# 代码、场景编辑、Play Mode 调试 | Unity | 否 | 是 | 引擎内 agent 范式 |
| **Astrocade**（$12M 种子） | 文本 → 可玩游戏（无代码） | 未核实 | 否 | remix | 消费级 prompt→game |
| **Ludo.ai** | 关键词 → 概念/GDD、精灵、3D 模型 | 无 | 否 | — | 仅资产/设计阶段 |
| **Roblox Cube 3D / 4D**（4D 公测 2026.02） | 文本 → mesh；4D：体验内文本 → 可驾驭的功能性对象 | Roblox | Cube 3D 开源 github.com/Roblox/cube | 是 | "文本→功能对象（几何+行为）"最接近"4D 程序"的商业实现 |
| **Unity AI**（2026） | 文本/图片 → 资产与可玩场景、C# 代码；GDC 2026 演示 prompt→roguelike 约 20 分钟（**未核实官方**） | Unity 6 + MCP | 否 | 是 | 引擎内 agent |
| **Godot MCP 插件生态** | LLM 经 MCP 建场景/节点/GDScript，确定性 playtest | Godot 4 | 是（gdai-mcp-plugin-godot、godot-mcp、Godot-MCP） | 是 | 开源可复用的"agent 操纵引擎"基建 |
| **Vibe coding 游戏**（fly.pieter.com，2025.02） | 对话 → three.js MMO 飞行游戏 | three.js | 部分 | 是 | 证明 LLM 直接写 three.js 游戏可行 |

### 1.2 论文 / 开源框架

| 项目 | 日期 | 输入 → 输出 | 引擎 | 开源 | 关系 |
|---|---|---|---|---|---|
| MarioGPT | NeurIPS 2023 | 文本 → Mario 关卡 | Java | MIT | 文本→关卡程序 |
| GameGPT | 2023.10 | 文本 → 多智能体分阶段生成游戏代码 | – | 未核实 | 早期 agentic 框架 |
| ChatDev / MetaGPT | 2023.06（ChatDev 2.0 于 2026.01） | 一句话 → 完整软件（Gomoku/2048 demo） | Python | 是 | 通用多智能体 |
| Word2World | 2024 | 故事 → 2D tile 世界 | 自带 | 是 | 文本→可玩世界规格 |
| GAVEL | NeurIPS 2024 | Ludii 描述语言 → 演化 + LLM 生成棋类规则 | Ludii | 是 | "游戏=程序"最纯粹的例子 |
| Sketch2Scene | Tencent 2024.08 | 草图 → 2D 图 → 可玩 3D 场景 | Unity/Unreal | 是 | 图片→可玩 3D 场景代表 |
| Mechanic Maker | AIIDE 2024 | 用户示范 → 符号程序合成机制 | 自研 | 未核实 | 从示例学规则程序 |
| Multi-Agent Game Gen via AVR | 2025.08 | 文本 + 资产库 → JS 游戏；录像做 omni-model 评审 | 浏览器 | 未核实 | 引入录像评估闭环 |
| SPRITE | 2026.03 | UI 截图 → YAML IR → 引擎 UI 资产 | – | 未核实 | 图片→引擎资产 |
| AutoUE | ACL 2026 Findings | 文本 → 模型检索 + 场景 + 玩法代码 + 自动 playtest | Unreal | 未核实 | 3D 引擎 agentic 生成 |
| **OpenGame** | 2026.04.21 (arXiv 2604.18394) | 文本 → 完整 Web 游戏；Game Skill（模板+调试）；GameCoder-27B | canvas / Phaser / three.js | Apache-2.0，含权重 github.com/leigest519/OpenGame | **首个开源 agentic 文本→three.js/Phaser 框架**，可作下游代码生成 backbone |
| PlayCoder | 2026.04 | 文本 → 可玩 GUI 代码；Exec@3 38.1%，Play@3 20.3% | 浏览器 | 未核实 | "编译通过≠可玩" |
| **Mage** | 2026.05 (arXiv 2605.07342) | 文本/结构化 IR → Unity C# | Unity | 未核实 | 直接 NL→C# 运行率 43% 但机制 F1≈0.12；**结构化 IR 显著提升机制保真** |
| Play2Code / PlaytestArena | 2026.05 | 文本 → 浏览器游戏；GUI agent 边玩边改 | 浏览器 | 未核实 | 玩测反馈闭环 |
| Knowledge-Conditioned Single-Pass Unity | 2026.07 | 26 个玩法概念 → Unity C#，单次生成 0 个编译通过 | Unity | — | 反面证据：无迭代修复不可行 |

### 1.3 基准

| 基准 | 日期/引擎 | 规模 & 结论 | 链接 |
|---|---|---|---|
| GameDevBench | ICML 2026，Godot 4.4.1 | 333 任务，最佳 68.8%；视觉反馈稳定提分 | github.com/waynchi/gamedevbench |
| GameCraft-Bench | 2026.06，Godot 4 | 140 任务，端到端整游戏 + 回放交互 + 多模态评审；最佳 41.46% | tongxuluo.github.io/gamecraft-bench-website |
| GameXpert-Bench | 2026.08 | 214 任务：GameGen/GameFix/GameOpt | arxiv.org/abs/2608.21833 |
| OpenGame-Bench | 2026.04 | 150 prompt，headless 浏览器 + VLM 评审 | 同 OpenGame |

## 2. B 类：神经世界模型 / 世界生成

### 2.1 神经视频世界模型（输出 = 动作条件的逐帧视频）

| 项目 | 日期 | 输入 → 输出 | 开源 |
|---|---|---|---|
| GameNGen (Google) | 2024.08 | 动作+历史帧 → DOOM 帧 | 否 |
| Oasis / Mirage (Decart) | 2024.10 / 2025 | 键鼠 → Minecraft 帧 | open-oasis 500M MIT |
| GameGen-X | ICLR 2025 | 文本/图 + 控制 → 游戏视频 | 是 |
| Genie 2 | 2024.12 | 单图 → 720p 可玩世界 ≈1 分钟 | 否 |
| Genie 3 / Project Genie | 2025.08 / 2026.01 | 文本(+图) → 24fps 720p 可玩数分钟 | 否；官方明确逐帧生成而非 3DGS |
| GameFactory (Kling) | ICCV 2025 | 文本+键鼠 → 视频；GF-Minecraft 70h | 数据集开源 |
| WHAM / Muse (MSR) | 2025.02 | 帧+手柄 ↔ 帧/动作 | 权重开源 |
| PlayGen (Tencent) | 2024.12 | 文本 → 可玩帧，RTX 2060 实时 | 是 |
| Matrix-Game 2.0 / 3.0 (Skywork) | 2025 / 2026.03 | 图+键鼠 → 25fps 分钟级；3.0：5B 720p@40fps | 全开源 |
| Hunyuan-GameCraft 1.0 / 2 | 2025.08 / 2025.11 | 单图+文本+键鼠 → 视频；2：自然语言指令交互 | 1.0 开源 |
| Yan (Tencent) | 2025.08 | Sim / Gen / Edit 三件 | 未开源 |
| HY-WorldPlay | 2025.12 | 图/文本+键鼠 → 流式视频 | 权重开源 |
| Odyssey / Runway GWM-1 | 2025–2026 | 实时视频 | 否 |

共同局限：无持久、可编辑的场景状态；不可导出到引擎；不是程序。

### 2.2 显式 3D 世界生成（输出 = mesh / 3DGS）

| 项目 | 日期 | 输入 → 输出 | 运行时 | 开源 | 关系 |
|---|---|---|---|---|---|
| HunyuanWorld 1.0 | 2025.07 | 文本/图 → 全景 → 分层 mesh | Web viewer；Unity/UE | 权重开源 | 图→可探索 mesh 世界 |
| HunyuanWorld-Voyager | 2025.09 | 图+相机轨迹 → RGBD 视频 + 实时 3D 重建 | – | 开源 | 视频⇄3D 桥 |
| **HY-World 2.0**（WorldMirror 2.0） | 2026.04 | 文本/单图/多视图/**视频** → 3DGS、mesh、点云、深度、法线、相机 | Blender/Unity/UE/Isaac | 开源 github.com/Tencent-Hunyuan/HY-World-2.0 | **视频→3DGS/mesh 的开源方案** |
| WorldGen (Meta) | 2025.11 | 文本 → LLM 程序化布局 + 扩散 3D + navmesh | 游戏引擎 | 代码未核实 | 唯一显式产出 navmesh 的文本→世界 |
| **Marble + Spark** (World Labs) | Marble 2025.11；Spark 2.0 2026.04 | 文本/图/**视频** → 3DGS + 碰撞 GLB + 视觉 GLB | **Spark：MIT three.js 3DGS 渲染器** | Marble 商用（免费档 4 世界） | **与本课题运行时完全一致**：视频→3DGS+collider→three.js；缺动力学/程序 |
| Visionary | 2025.12 | 3DGS/4DGS/mesh + ONNX 的 WebGPU 平台 | 浏览器 | 项目页 | Web 端 4DGS 运行时候选 |

## 3. C 类：视频 → 可执行 / 可仿真（最贴近课题）

| 项目 | 日期 | 输入 → 输出 | 运行时 | 开源 | 差距 |
|---|---|---|---|---|---|
| **Video2Game** | CVPR 2024 (arXiv 2404.09833) | 视频 → NeRF → 神经纹理 mesh + 凸分解碰撞体 → 浏览器游戏 | **three.js + cannon.js** | MIT github.com/video2game/video2game | **最接近**；输出是资产，逻辑手写模板；静态，无程序 |
| **HoloScene** | NeurIPS 2025 (arXiv 2510.05560) | 视频 → 交互式场景图（mesh + 3DGS + 物理参数 + 层级/接触关系） | UE 第三人称 demo | github.com/xiahongchi/HoloScene | 场景图已是半结构化程序；非 Web、无行为规则、静态 |
| Vid2Sim（城市导航 / 物理） | 2025.01 / 2025.06 | 单目视频 → 数字孪生 / 物理参数 | – | 部分 | 面向机器人 |
| PhysGen | ECCV 2024 | 单图 → 物理理解 → 刚体仿真 → 视频 | 自研 | 是 | "图→显式仿真器→渲染"三段式 |
| SceneScript (Meta) | 2024 | 第一视角视频/点云 → CAD 式结构化语言 | – | 权重（学术） | 唯一"视频→语言程序"的 3D 工作，仅静态布局 |
| Game Engine Learning from Video (Guzdial) | IJCAI 2017 | Mario 视频 → 前向仿真规则 | 自研 | — | 2D 程序合成线起点 |
| **FAE** (UAlberta/Amii) | 2025.08 (arXiv 2508.11836) | Pac-Man / River Raid 视频 → Retro Coder DSL 程序 | 自研 DSL | 无代码 | **"视频→程序"最直接的 2025–26 工作**，但 2D、非可玩 |
| PoE-World (Ellis 组) | 2025.05 | 少量示范 → Python 程序专家乘积世界模型 | Python | 是 | 从观测合成可执行程序 |
| WorldCoder | NeurIPS 2024 | 交互轨迹 → Python 世界模型 | Python | 是 | 输入非视频 |
| Code World Models for GGP | 2025.10 | 自然语言规则 + 轨迹 → Python 状态机 + MCTS | Python | 未核实 | "游戏=代码世界模型" |
| **ChronoAgentic** | 2026.05 (arXiv 2605.14398) | 文本 → PyChrono 仿真代码；PhyWorldBench 82.5% vs T2V 52.5% | Chrono | 未核实 | 明确论证"生成仿真程序优于生成视频" |
| VDAWorld / SimuScene | 2025.12 / 2026.02 | 图+caption → 可执行抽象 / 文本 → 物理动画代码 | 多仿真器 | 未核实 | 图/文→可执行抽象 |
| Game2World Engine | 2026.08 | 野外游戏视频（303 款）→ 去 HUD 训练数据 | — | 将开源 | 视频仅作训练数据 |

## 4. 两个具体问题

**(1) 是否已有"视频 → 可执行游戏 / 场景程序"的工作？** 没有完全匹配的。四支各占一角：视频→几何资产→引擎（Video2Game、HoloScene、Marble/HY-World 2.0）静态、数据；视频→符号程序（Guzdial、FAE、PoE-World）2D、无 Web 运行时；文本/图→仿真代码（ChronoAgentic、SimuScene、Mage IR 路线）输入非视频；视频→结构化语言（SceneScript）仅静态布局。"4D（动态）"与"程序化表示"两环同时缺失。

**(2) 今天就能跑通的端到端 pipeline**

| 路线 | 可用性 | 备注 |
|---|---|---|
| Video2Game（MIT） | 视频→浏览器 three.js 游戏 | 需 COLMAP + NeRF 训练，重；静态 |
| HoloScene | 视频→可仿真场景图 | UE demo，非 Web |
| **Marble（免费档）+ Spark（MIT）** | 视频/图/文本→3DGS+碰撞 GLB→three.js | **最快落地**；无动力学 |
| HY-World 2.0（开源权重） | 视频/图/文本→3DGS/mesh | 需大显存 |
| OpenGame（Apache-2.0） | 文本→Phaser/three.js 游戏 | 可作下游代码生成器 |
| Godot MCP 插件 + Claude/Cursor | 文本→Godot 工程，可自动 playtest | — |
| Matrix-Game 3.0 / HY-WorldPlay 等 | 图→可控视频帧 | 非程序、非持久 |

**对本课题的定位**：可行的组合是 Video2Game/HY-World 2.0 类几何前端（或 Marble API）负责静态几何与碰撞体；FAE/PoE-World 类程序合成负责从视频提取物体、事件与动力学规则；Mage/OpenGame 证明的"结构化 IR → three.js 代码"负责落地。三者尚无人拼接，且 Spark 已把 3DGS 与 three.js mesh 融合渲染打通。

## 5. 主要来源
Video2Game https://arxiv.org/abs/2404.09833 ；HoloScene https://arxiv.org/abs/2510.05560 ；FAE https://arxiv.org/abs/2508.11836 ；PoE-World https://arxiv.org/abs/2505.10819 ；WorldCoder https://arxiv.org/abs/2402.12275 ；ChronoAgentic https://arxiv.org/abs/2605.14398 ；SceneScript https://www.projectaria.com/scenescript/ ；Genie 3 https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/ ；Matrix-Game https://github.com/SkyworkAI/Matrix-Game ；HY-World 2.0 https://github.com/Tencent-Hunyuan/HY-World-2.0 ；WorldGen https://arxiv.org/abs/2511.16825 ；Marble https://www.worldlabs.ai/blog/marble-world-model ；Spark https://github.com/sparkjsdev/spark ；OpenGame https://arxiv.org/abs/2604.18394 https://github.com/leigest519/OpenGame ；GameDevBench https://github.com/waynchi/gamedevbench ；GameCraft-Bench https://arxiv.org/abs/2606.17861 ；Mage https://arxiv.org/abs/2605.07342 ；PlayCoder https://arxiv.org/abs/2604.19742 ；Play2Code https://arxiv.org/abs/2605.28258 ；Sketch2Scene https://arxiv.org/abs/2408.04567 ；GAVEL https://arxiv.org/abs/2407.09388 ；Roblox Cube https://github.com/Roblox/cube ；Rosebud https://rosebud.ai/ai-game-creator ；Bitmagic https://bitmagic.ai/lab/ ；Godot MCP https://github.com/3ddelano/gdai-mcp-plugin-godot ；Game2World https://arxiv.org/abs/2608.24680 ；世界模型综述 https://arxiv.org/abs/2606.01164
