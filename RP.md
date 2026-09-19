# GameWorldModel的研究提案

---

## 0. 概述

输入一段无标定的单目视频，系统推断出一份带时间轴的可执行场景程序（4D Scene Program），包括静态结构、物体集合、每个物体的运动程序和相机轨迹。程序编译后在 three.js 中实时运行，支持逐物体编辑，并可作为可玩游戏世界的场景底座。

---

## 1. 动机

### 1.1 两条世界模型路线

| 路线 | 代表工作 | 特点 | 局限 |
|---|---|---|---|
| 视频生成式 | Genie 3、Matrix-Game、Hunyuan-GameCraft | 视觉保真度高，开放域 | 输出不确定，无持久状态，长时一致性差，难以编辑，推理开销大 |
| 代码/状态式 | WorldCoder、Code World Model（2026.08）、From Pixels to States（2026.07） | 确定性执行，可编辑，实时，状态可持久 | 状态来自文本描述或已有引擎，没有从视频获取状态的方法 |

本工作研究代码/状态路线中缺失的一环：从视频与图片推断程序化的场景状态。

### 1.2 现有 inverse graphics 工作

以程序作为场景表征的工作已有一条线：

- IG-LLM（2024）：单图到图形代码，LLM 解码 CLIP 特征。
- SceneCraft、3D-GPT、Holodeck（2024）：文本到 Blender 或引擎代码。
- SceneScript（ECCV 2024）：第一人称视频经点云到布局级结构化语言（墙、门、窗、包围盒）。
- SEIG / Thinking in Blender（2026.06）：单图到 Blender 程序，分阶段解码。
- Scenix（2026.08.06）：稀疏无标定 RGB 到可执行场景程序，程序包含房间轮廓和物体的身份、位置、尺寸、朝向、支撑关系；用生成资产实例化，闭环空间精修；配套约 11 万室内合成数据 XScene。作者下一步计划用 RL 做跨视角一致性。

上述工作处理的都是静态场景，输入为单图或少量图片，输出没有时间轴。视频中的连续相机轨迹、运动物体、关节结构和时序事件尚未被这类方法利用，而这些信息是构建游戏世界所需要的。

### 1.3 视频引入的三个问题

1. 相机运动与物体运动的解耦：画面中的位移需要分配给相机程序或物体程序。
2. 跨帧程序一致性：物体身份需要在时间上保持，程序参数需要跨帧稳定。
3. 程序与重建的划分：哪些内容表示为程序节点（可动、可交互、有重复结构），哪些留给 mesh 或 Gaussian 残差。在静态场景中这个划分影响不大；在动态场景中它决定了哪些物体可以被编辑和交互。

---

## 2. 问题定义

### 2.1 输入

单目 RGB 视频 $V = \{I_t\}_{t=1}^{T}$，无相机内外参，包含相机运动和若干运动物体。可选输入一句自然语言提示，例如"这是一个第三人称平台跳跃游戏的录像"。

### 2.2 输出

4D 场景程序 $P$，由四部分组成：

```
P = ( C, S, O, R )
  C : 相机轨迹程序        — 关键帧位姿 + 插值方式，或参数化跟随规则
  S : 静态结构           — 地面、墙体、地形块等 primitive 及其材质
  O : 动态/可交互物体集合 — 每个物体 = 几何(primitive | 资产引用) + 位姿 + 层级 + 运动程序 M_i
  R : 残差重建           — 程序未覆盖部分的 mesh / Gaussian（可选）
```

### 2.3 DSL 草案

设计要求：紧凑、可 token 化、可编译到 three.js 及其他引擎。模型输出 DSL，编译器负责生成 three.js 代码。

```
scene "clip_0042" {
  camera {
    intrinsics: auto
    motion: spline(keyframes=[(t=0.0, pos=..., rot=...), ...])
  }

  static {
    ground_0 : plane(size=[40,40], material=grass)
    wall_1   : box(pos=[..], extent=[..], rot=[..], material=stone)
    terrain_2: heightfield(ref="terrain_2.png", scale=[..])
  }

  objects {
    door_7 : asset(class="door", extent=[1.0,2.1,0.1]) @ pose(pos=[..], rot=[..])
      motion: revolute(axis=[0,1,0], pivot=[..], range=[0,90], schedule=[(t=3.2,to=90)])

    platform_3 : box(extent=[2,0.3,2], material=metal) @ pose(pos=[..])
      motion: periodic_translate(axis=[0,1,0], amp=1.5, period=4.0, phase=0.0)

    cart_9 : asset(class="minecart") @ pose(pos=[..])
      motion: trajectory(keyframes=[(t=0,pos=..),(t=2.5,pos=..),...], interp=linear)

    coin_12 : instances(proto=asset(class="coin"), poses=[..,..,..])
      motion: spin(axis=[0,1,0], rate=180)
      event : despawn_on_contact(with=player)
  }

  residual: gaussians(ref="clip_0042_residual.ply")
}
```

运动原语词表（可扩展）：`static`、`trajectory`、`revolute`、`prismatic`、`periodic_translate`、`periodic_rotate`、`spin`、`spawn/despawn`、`trigger`。

### 2.4 程序与残差的划分规则（初版）

表示为程序节点：有运动的物体、语义上可交互的物体（门、开关、可拾取物）、重复出现的实例、规则几何体。
表示为残差：大面积不规则静态几何（植被、岩石细节）、远景。
用两个量衡量划分质量：程序覆盖率（被程序节点解释的像素或几何比例）和残差体积。

---

## 3. 方法

### 3.1 流水线

```
视频 ──► [A] 感知与证据提取 ──► [B] 分阶段程序解码 ──► [C] 执行反馈精修 ──► [D] 残差补全 ──► 4D 程序 + three.js 场景
              │                        ▲                      │
              └── 相机/深度/动静分离/轨迹 ┘◄── 逐帧重渲染比对 ────┘
```

### 3.2 [A] 感知与证据提取

使用现有模型：

- 相机位姿与深度：VGGT 一类前馈几何模型，或 CVPR 2026 的动态重建模型（如 MoRe，通过 attention-forcing 分离动态运动与静态结构）。
- 动静分离与实例轨迹：SAM2 加视频跟踪，得到每个候选物体随时间的 3D 包围盒序列。

输出结构化证据 token：`{obj_id, class_guess, bbox_3d(t), is_dynamic, contact_relations}`。证据 token 与视频帧一起作为 VLM 的输入。

### 3.3 [B] 分阶段程序解码

VLM（Qwen-VL 量级，可微调）自回归输出 DSL。SEIG 报告分阶段解码优于一次性解码所有因素；这里按时空结构划分阶段：

1. 相机程序 C
2. 静态结构 S
3. 物体几何与初始位姿 O_geo
4. 每个物体的运动程序 M_i
5. 事件与关系（接触、触发、支撑）

每个阶段以前一阶段的编译结果（渲染图和证据对齐报告）为条件。

### 3.4 [C] 执行反馈精修

three.js 不可微，精修通过执行、渲染、比对、修正的循环完成。Scenix 的精修在单个静态场景内比较空间布局；这里的比较逐帧进行，包括：

- 逐帧渲染的深度、实例掩码、DINO 特征与感知结果的差异；
- 每个物体的程序轨迹与观测轨迹的误差；
- 物体身份的跨帧一致性（ID 交换、凭空出现或消失）；
- 相机轨迹残差。

反馈有两种用法：

- 推理时：生成结构化错误报告（例如"platform_3 在 t=6~8s 位置偏差 0.8m，疑似周期参数错误"），VLM 修订对应节点，迭代 2~3 轮。
- 训练时：作为 RL 奖励（GRPO 或同类方法）。奖励由渲染比对计算，不需要真值程序，因此可以在无标注的真实视频上训练。

### 3.5 [D] 残差补全

程序渲染结果与观测之间的差异区域用 Gaussian 或 mesh 拟合为残差层，挂在程序的 `residual` 节点下。残差只用于渲染，不参与交互逻辑。

### 3.6 训练策略

1. SFT：在合成数据上以真值 DSL 做监督微调，包含分阶段格式。
2. RL：以执行反馈为奖励，合成视频与真实无标注视频混合训练。
3. 课程安排：静态场景，然后单运动物体，然后多物体加关节和事件。

---

## 4. 数据引擎

### 4.1 合成数据：VidProg-Synth（暂名）

用 three.js 或 Godot 程序化生成。每条样本包含真值程序、逐帧相机、深度、实例掩码和物体轨迹。

- 场景模板：室内房间、户外地形块、平台关卡、赛道、走廊迷宫。
- 资产：开源资产库（Objaverse 类）按类别抽样，加 primitive。
- 运动采样：从 2.3 的原语词表按分布抽样参数，每场景 1~8 个动态物体。
- 相机：手持式抖动轨迹、第三人称跟随、固定俯视、环绕。
- 渲染变化：光照、材质、天气、分辨率、帧率。
- 规模目标：100k 条以上 clip，每条 5~15 秒。作为参照，Scenix 的 XScene 约 11 万样本。

### 4.2 真实评测集：VidProg-Real（暂名）

- 游戏录像（有真值）：用开源引擎游戏（Godot 示例项目、开源 three.js 游戏）在录制时同步导出引擎状态，得到真值物体位姿与运动，200~500 条。
- 游戏录像（无真值）：商业游戏公开录像，用于定性展示和人评。
- 日常视频：包含运动物体的手持视频（DAVIS、TAPVid 类来源），人工标注粗粒度程序（物体类别、运动类型、关节轴），约 100 条。

---

## 5. 评估协议

### 5.1 指标

| 维度 | 指标 |
|---|---|
| 静态布局（与 Scenix 对齐） | 物体检出 F1、位置/尺寸/朝向误差、支撑关系准确率 |
| 相机 | 轨迹 ATE / RPE |
| 动态 | 运动类型分类准确率、轨迹 ATE、关节轴角度误差、周期/幅度相对误差、事件时序误差 |
| 时序一致性 | 物体 ID 交换次数、凭空出现/消失次数、程序参数跨帧漂移 |
| 程序质量 | 程序覆盖率、残差体积、节点数与真值的比值、编译成功率 |
| 重渲染 | 留出帧 PSNR / LPIPS、FVD |
| 可编辑性 | 修改一个物体的运动程序后重渲染的任务成功率与人评 |
| 下游可玩性（应用章节） | 绑定玩法后 VLM agent 通关率、人评 |

### 5.2 基线

- Scenix（按帧抽样作为多视角输入）
- SceneScript 风格的布局语言模型
- SEIG（单帧到程序）
- Video2Game（视频到几何到 three.js，输出为几何）
- 4D Gaussian 方法（MoSca、Shape of Motion 类，输出为不可编辑的 Gaussian；只比较渲染保真度）
- Code World Model（若开源）
- 消融基线：一次性 VLM 直接生成 three.js 代码、去掉证据 token、去掉反馈闭环

### 5.3 消融

证据 token 有无；分阶段解码与一次性解码；反馈迭代轮数；SFT 与 SFT+RL；程序/残差划分规则；DSL 粒度（粗、细）。

---

## 6. 应用：双输入可控游戏生成

这一部分展示程序表征的应用，放在主实验之后。

- 视觉输入（视频或图片）经本文方法得到 4D 场景程序，决定场景与动态。
- 语言输入（玩法描述）经 LLM 生成玩法程序（控制、规则、胜负条件）。
- 绑定层：在场景程序节点上推断玩法所需的 affordance（可行走面、可放置点、路径、出生点、目标点），把玩法程序挂到场景节点上。
- 六种玩法模板：第三人称平台跳跃、赛车、塔防、俯视角生存、简单射击、解谜/逃脱。
- 解耦实验：固定视频更换文字，检验只有玩法变化；固定文字更换视频，检验只有场景变化。
- 交互编辑：用一句话修改运动程序或规则，重新编译。

同一份 4D 场景程序可以承载多种玩法。

---

## 7. 贡献声明（草稿）

1. 提出视频到 4D 可执行场景程序的任务定义，把运动物体、关节、事件和相机轨迹纳入可执行场景程序，并给出程序与重建的可量化划分。
2. 提出带时间轴的执行反馈精修：以逐帧重渲染比对构造时序一致性信号，用于推理时修订，也作为不依赖真值程序的 RL 奖励，使模型能在真实无标注视频上训练。
3. 构建 VidProg-Synth / VidProg-Real 数据与评测协议，覆盖静态布局、动态程序、时序一致性、程序质量与可编辑性。
4. 在可控游戏生成上验证程序表征：同一场景程序承载多种玩法，支持语言级编辑。

---

## 8. 风险与应对

| 风险 | 应对 |
|---|---|
| 被视为 Scenix 的增量工作 | 静态部分复用或对齐 Scenix；实验重点放在动态、时序和程序/残差划分；标题和摘要以动态/4D 为关键词 |
| VLM 长视频上下文不足 | 证据 token 压缩加关键帧采样；运动信息主要通过证据 token 传递 |
| 合成到真实的分布偏移 | RL 阶段混入真实无标注视频；真实评测集中有真值的部分来自开源引擎录像 |
| DSL 表达力与可学习性的矛盾 | 先用小词表跑通，按消融结果扩展；保留通用节点（自由 mesh 加关键帧轨迹）保证编译成功率 |
| 感知前端错误级联 | 证据 token 附带置信度；反馈闭环允许 VLM 否决证据 |
| 9 周内无法完成全部内容 | 见第 9 节与 `docs/baseline.md` 的最小可发表版本与备选方案 |

---

## 9. 实施路线与文档索引

第一阶段的目标是尽快跑通"视频进、游戏出"的闭环，暂不做 benchmark 与模型训练。2026-09-18 完成了五路调研（已有工作、运行时与游玩层、素材、agent 与视觉反馈、感知与模型层）和 Vulcan 集群实测，结论与实施细节按层拆到 `docs/` 目录，本节只保留路线图。

### 9.1 六个问题的结论

1. **已有工作**：没有系统走通"视频 → 带时序与动力学的场景程序 → Web 引擎"。最近的是 Video2Game（视频到 mesh 到 three.js，静态、无程序）、HoloScene（视频到带物理参数的场景图到 UE）、FAE（2D 游戏视频到 DSL）、Marble + Spark（视频到 3DGS 到 three.js，无动力学）。文本到游戏方向上 OpenGame 与 Mage 表明：模板库有效，先出中间表示再出代码优于直接出代码。
2. **Three.js**：作为编译目标。vanilla three.js 加 Rapier 物理、recast 导航、miniplex ECS，免打包 ESM；Spark 渲染 Gaussian 残差。
3. **素材库**：要，但小。参数化 primitive 兜底，一两千个 CC0 低多边形 GLB 加 CLIP 检索，hero 物体可选 TRELLIS.2 生成。
4. **Multi-agent**：不用角色扮演式多智能体。按阶段分解加有渲染落地的验证器；脚手架 Pydantic AI，可在 Claude API 与本地 vLLM 之间切换。
5. **游玩层**：固定内核加模板槽位，DSL 只填绑定段，状态 schema 固化在内核；试玩验收用轨迹回放加 VLM 评审。
6. **视觉反馈**：Playwright 无头 Chromium 渲染 RGB、深度、ID 三个 pass（集群上 CPU 软渲染与 GPU 硬件加速均已实测可用）；数值层生成逐物体 pass/fail 子句，VLM 只看失败区域并输出结构化清单。
7. **模型层**：需要。感知用 ViPE、SAM 3.1、SpatialTrackerV2；程序合成基线走 `claude-opus-5` API，自托管备选 Qwen3-VL-32B；后续微调对象 Qwen3-VL-8B 或 Qwen3.5-9B。

### 9.2 六个阶段

手写程序跑通 → 视频变证据 → 证据变静态程序 → 让物体动起来 → 程序变游戏 → 素材与展示。前两个阶段可并行，前五个阶段构成闭环，第六个阶段全部可选。每阶段约一周，之后三周留给第 4、5 节的数据与评测。任务清单、验收标准、备选方案见 `docs/baseline.md`。

### 9.3 文档索引

| 文档 | 内容 |
|---|---|
| `docs/baseline.md` | 第一阶段计划：目标与验收、测试片段、六个阶段的任务清单、依赖、备选、待决 |
| `docs/architecture.md` | 模块与数据流、产物目录、接口契约（evidence.json、`window.__game`、反馈报告） |
| `docs/dsl.md` | 场景程序 DSL：文本形式与 JSON 形式、最小词表、校验规则、编译映射、与 Scene Spec 的对应 |
| `docs/runtime.md` | 运行时选型、three.js 游戏栈、内核 systems、玩法模板与槽位、无头渲染、试玩验收 |
| `docs/perception.md` | 感知模型对比、证据提取算法（OBB、螺旋拟合、接触）、资源预算 |
| `docs/models.md` | API 与开源 VLM 对比（视觉与视频支持、部署、许可、价格）、调用方式、成本估算、训练栈与数据集 |
| `docs/feedback.md` | 视觉反馈原则、数值层指标、VLM 批评、候选选优、编排与脚手架、多智能体证据 |
| `docs/assets.md` | 三层回退、CC0 素材库、检索配方、生成模型、许可规则 |
| `docs/related-work.md` | 按类别整理的相关工作与本课题定位 |
| `docs/cluster.md` | Vulcan 登录节点与计算节点差异、网络出口、无头渲染实测、依赖打包、作业模板 |
| `survey/` | 五份原始调研报告与集群探测脚本、日志 |

---

## 10. 待决问题

1. DSL 粒度：物体几何以 primitive 组合为主还是以资产引用为主。前者可学习性好，后者视觉效果好。建议双轨，由消融决定。
2. 相机程序用关键帧还是参数化规则（跟随、环绕）。游戏录像中参数化规则更常见。
3. 证据 token 是否加入训练目标作为辅助损失，还是只作为条件输入。
4. 真实游戏录像中的 UI 叠加和镜头切换如何处理。前期过滤，后期再处理。
5. 题目候选：*Video2Program*；*Programs in Motion: Inferring 4D Executable Scene Programs from Monocular Video*；*Seeing Worlds as Code*。

---

## 附：相关工作清单（写作时需引用或对比）

- Scenix (2026.08)：稀疏视角到可执行场景程序，XScene 数据
- SceneScript (ECCV 2024)：视频/点云到结构化布局语言
- IG-LLM (2024)、SEIG / Thinking in Blender (2026.06)：单图到图形程序
- SceneCraft (ICML 2024)、3D-GPT、Holodeck (CVPR 2024)、LayoutGPT：文本到场景程序
- The Scene Language、ShapeAssembly、Learning to Infer 3D Shape Programs：程序化表征
- Video2Game (CVPR 2024)：视频到可交互 three.js 环境
- WorldCoder (NeurIPS 2024)、Code World Models for General Game Playing (2025)、Code World Model: Coding Agent as World Brain (2026.08)：代码式世界模型
- From Pixels to States (2026.07)：以引擎状态为中心的世界模型分析
- Genie 3、Matrix-Game、Hunyuan-GameCraft、GameFactory：视频式世界模型
- VGGT、MonST3R、CUT3R、MoRe (CVPR 2026)、MoSca、Shape of Motion：前馈与动态重建
- OpenGame、GameDevBench、GameCraft-Bench：agent 游戏开发（应用章节对比）

2026-09-18 调研新增的条目（HoloScene、FAE、OpenGame、Mage、Marble/Spark、HY-World 2.0、ViPE、SAM 3、GameDevBench、GameCraft-Bench 等）已按类别整理到 `docs/related-work.md`。
