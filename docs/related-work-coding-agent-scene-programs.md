# 相关工作补充：用编码智能体生成显式场景程序

`related-work.md` 的调研停在 2026-09-18。而 GPT-6 Astra 在 2026-09-03 发布之后，
九月和十月出现了一批「让编码智能体写出显式场景程序」的工作，其中至少一篇已进
NeurIPS 2026。这一篇把那一批整理出来，回答三个问题：我们的基础方法是不是走错了、
别人做到什么程度、哪些做法我们该直接拿过来。

结论先说：**方法方向没有走错，这是当前一条热门且已被顶会认可的路线。但我们在三处
具体做法上落后，而且其中两处正是我们现在卡住的地方。**

## 这一批是什么

| 工作 | 输入 | 产出 | 关键做法 |
|---|---|---|---|
| LEGO-Anything（arXiv 2609.36380，09）| 单图 | Blender 代码 | 智能体闭环：写码 → 执行 → 看渲染 → 改 |
| NeoWorld-Pro（arXiv 2608.24212，**NeurIPS 2026**）| 单图 | Blender Python → URDF + USD | **物理在环**：Isaac Sim 里跑，不合理就修 |
| NeoWorld-3（arXiv 2610.27690，10）| 单图 | 代码定义的物理世界 | — |
| Agentic Real2Sim（arXiv 2607.19190）| 视频（DROID 机器人）| MuJoCo episode | 偏机器人数据 |
| Agentic RSR（arXiv 2610.10479，10）| 工作区视频 + 机器人模型 | MuJoCo | 用已知机器人尺寸定米制尺度 |
| LiteReality-Agent | RGB-D | 可 articulate、simulation-ready 室内场景 | 把重建当软件工程做 |
| BVB（arXiv 2609.15478）| 视频 | Blender 程序 | 是个基准，不是方法 |

更早的前身：SceneCraft（2024，LLM 写 Blender 代码）、VIGA（arXiv 2601.11109）、
Thinking in Blender（arXiv 2606.02580）、Code-as-room（arXiv 2605.18451）。

## 对 `related-work.md` 里「定位」那一节的影响

原来的说法是「现有工作四支各占一角，没有系统走通单目视频 → 带时序与动力学的场景
程序 → Web 引擎」，并把「4D（动态）」和「程序化表示」两环同时缺失作为本课题的位置。

**这个定位基本还成立，但缝隙比九月窄了：**

- NeoWorld-Pro 已经做到「单图 → 程序 → 关节 + 物理属性 → 仿真器里验证」。
  **程序化表示这一环不再是空白**，缺的只是视频输入和时序动力学。
- Agentic Real2Sim 和 Agentic RSR 确实吃视频，但**都绑在机器人数据上**
  （DROID episode、工作区视频 + 已知机器人模型），不是任意视频。
- 没有一篇同时做到「任意单目视频 + 观测到的运动 + 可交互程序」。

所以我们的位置仍在，但**写作时不能再说「没人做程序化场景表征」**，必须说清楚我们
和 NeoWorld-Pro 的差别在输入模态和时序建模上。

## 五条可以直接拿过来的做法

### 一、配对容差应该是相对的，不是绝对的

LEGO-Bench 的几何分 `R` 是 F@5%：把可见表面投到图像平面，按实例掩码分到物体上，
逐物体算精确率和召回，**容差是 `τ(g) = 0.05·z(g)`，随深度缩放**。

我们的 `scripts/eval --max-dist` 是**绝对米制**（当前 0.45 米）。这件事我们自己
踩过——绝对阈值不是尺度不变的，在小场景里等于「什么都能配上」，在大场景里又过紧。
LEGO-Bench 用相对容差是对的，**这一条值得直接改**。

注意一个差别：他们**不做规范对齐**（"compared in camera coordinates without
alignment"），而我们做偏航 + 水平平移对齐。两种口径不能混着比。

### 二、关节不要靠几何推断

这是最要紧的一条。NeoWorld-Pro **根本没有单独的关节检测方法**——由 MLLM 从图像
语义直接生成关节程序（论文举的例子是「从看得见的把手推断抽屉的内腔」），然后
**用物理去修**。

他们的关节指标：

| | NeoWorld-Pro | 最强基线 Articraft |
|---|---|---|
| 漏检率 | **11.83%** | 22.45% |
| 类型错误 | **18.78** | 29.39 |
| 轴误差 | 28.52 | 30.14 |
| 支点误差 | 0.19 | 0.23 |

对照我们：关节那一档在神经感知下 **0/7 全错**，全真值几何 + 全真值分割下也只有
**1/13**。根因已经查清是 #7 的 ICP 伪造转动（静止物体被解出 26.6° 转动，而铰链
门槛是 8°），而那条线上已经排除了二十条修法。

**也就是说：我们在用一条已经证明走不通的几何路线，去做别人用语义先验 + 物理验证
做成的事。** 这是本次调研最值得行动的一条。

### 三、物理当验证器，不只当运行时

NeoWorld-Pro 的物理在环分两层：

- **物体级**：URDF 在 Isaac Sim 里跑自由落体和受力扰动，一个看视频的 MLLM 评审
  关键帧并提修改建议。**改动被限制在惯性参数、关节限位、阻尼、原点和碰撞几何**，
  不动运动学拓扑和可见几何。
- **场景级**：用 CEM 优化每个物体的四维扰动（平面平移、偏航、对数尺度），奖励
  = 语义 MLLM 分 + 穿模惩罚 + 漂移惩罚。

消融（表 4）：完整方法成功率 **92.85%**，去掉语义分 48.81%，**开环 44.05%**。
**闭环把成功率翻了一倍多。**

我们已经有 Rapier 在运行时跑物理，但只用来「跑」，没用来「验」和「修」。
这条的成本不高、收益有先例。

### 四、但不要盲目上智能体自评

LEGO-Anything 量了三个失败模式，其中两个对我们是警告：

- **迭代中回退**：GPT-5.6-sol 有 **29.6% 的编辑让分数变低**；最终提交比它自己
  最好的中间结果低 3.2 分。
- **自评不可靠**：让模型二选一判断哪个渲染更好，和指标方向的一致率——
  **几何上自评 45.8%（低于 50% 的随机水平）**，外观上 62.2%。跨模型评审
  45.4% / 63.9%。

**几何自评低于随机**，这是个硬结论。我们现在是单趟确定性翻译，天然没有这两个问题。
所以如果要引入智能体闭环，**必须带版本控制和外部判据**，不能让模型自己说好不好。
LEGO-Plugin 的三个模块正是针对这个：增强初始化（VGGT 恢复相机）、接地式精化
（SAM 3 + Depth Anything V2 做显式残差）、**版本控制（接受 / 修复 / 回滚）**。

### 五、外观要量，但不是和 3DGS 比

**这一批没有一篇拿 3DGS 或 NeRF 作对比。** NeoWorld-3 的开场白直接写
"Simulation-ready world reconstruction demands physical interactivity beyond
visual realism"。

但他们**确实量外观**：LEGO-Bench 的 `A` 是「用标准引擎重渲染，和参考图逐像素比，
最大通道误差 ≤ 30/255 的像素占比」。总分 `S = 均值( ½(R+A)·V )`，失败算 0。

一个有意思的对照：Gen3DSR 的室内几何分 **65.4%**（高于 GPT-6-astra 的 52.4%），
但外观分 **0.0%**，总分只有 30.1%。**几何和外观是可以分开赢的**，而我们目前的
处境接近 Gen3DSR——有一点几何，外观近乎没有（形状种类准确率 0.300，所有物体
渲成盒子）。

## LEGO-Bench 主结果（供写作引用）

单位 %，三次运行的均值（标准差）。

| 模型 | 室内 V | 室内 R | 室内 A | 室内 S | 室外 S |
|---|---|---|---|---|---|
| GPT-6-astra + Codex | 100.0 (0.0) | 52.4 (1.1) | 54.4 (0.7) | **53.4 (0.8)** | **39.6 (0.8)** |
| GPT-6-sol + Codex | 99.4 (1.1) | 22.2 (2.4) | 42.5 (0.3) | 32.3 (1.0) | 24.2 (0.4) |
| GPT-6-luna + Codex | 99.7 (0.5) | 15.8 (1.2) | 30.5 (0.3) | 23.2 (0.7) | 17.3 (0.5) |
| VIGA | 100.0 (0.0) | 10.8 (1.5) | 20.3 (0.7) | 15.6 (1.1) | 12.6 (0.2) |
| Gen3DSR | 92.0 (1.1) | **65.4 (0.7)** | 0.0 (0.0) | 30.1 (0.4) | 14.0 (0.1) |
| 3D-RE-GEN | 100.0 (0.0) | 1.2 (0.1) | 11.3 (0.2) | 6.3 (0.1) | — |

两个值得注意的趋势：

- **推理预算的影响很大**。在 42 个案例的 Office 子集上，GPT-6-astra 的总分随
  reasoning effort 从 Low 到 XHigh **由 32.3% 升到 61.8%**；GPT-5.6 系列没有
  一致增益。
- **场景复杂度的影响反而小**。总分从 Easy 的 24.6% 到 Hard 的 20.5%，
  而有效率始终 99.5%。换句话说，难的不是场景复杂，是保真度本身。

## 从生成的场景里读视觉任务（LEGO-World）

把生成出来的场景当成一个世界模型，从里面读检测、分割、深度：

| 任务 | 指标 | 从场景读出 | 专用模型 |
|---|---|---|---|
| 检测（COCO val2017）| Box AP ↑ | 30.14 | DINO 59.88 |
| 分割（LVIS v1 val）| Mask AP ↑ | 14.75 | SAM 3 53.96 |
| 深度（ETH3D）| AbsRel ↓ | 0.1554 | Depth Anything 3 0.0783 |

大约是专用模型的一半。作者的结论是「有希望但还不够精确」。

## 我们还站得住的差异化

1. **视频输入，而且用观测到的运动**。这一批主流是单图；做视频的都绑在机器人数据上。
   我们的 `motion` 字段和第一档（观测运动回放）在这批工作里没有对应物。
2. **样式与行为分写，而且在量**。没有检索到别人测解耦度。我们的行为漂移 0.207 米 /
   样式响应 0.247（见 #52）可能是新的。
3. **带真值程序的基准**。LEGO-Bench 是 simulator-grounded 的 208 张图；我们的数据
   由生成器产出、**有已知真值程序**，能算精确的逐物体配对指标，并且支持成对样本
   （同行为换样式）。这是我们手上独有的东西。

## 口径与未核实

- 除 LEGO-Anything 和 NeoWorld-Pro 外，其余条目**只读了摘要或二手摘录**，
  没有通读正文。标注的 arXiv 编号来自搜索结果，引用前应逐一核对。
- NeoWorld-Pro 正文里几处数字在网页渲染中有明显排版错误（"9090 个类别"、
  "fewer than 1010 objects"、MLLM 评分的 "11–77" 量程），本文按最合理的读法
  记为 90 类、少于 10 个物体，**引用前必须查 PDF 原文确认**。
- LEGO-Anything 的网页正文长 166k 字符，本次只读了前 100k，后半部分（含附录与
  部分消融）未读。
- 本文所有「我们」这一侧的数字来自本仓库 2026-10-19 的测量，口径见 #5 的索引评论
  （部分早期数字是在带真值 bug 的批次上量的）。

## 来源

- LEGO-Anything: Coding Agents for 3D Scene Reconstruction — https://arxiv.org/abs/2609.36380v1
- NeoWorld-Pro: Programming Interactive Scenes from Monocular Images for Embodied Simulation（NeurIPS 2026）— https://arxiv.org/html/2608.24212v1
- NeoWorld-3: From Pixels to Code-Defined Physical Worlds — https://arxiv.org/html/2610.27690
- Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents — https://arxiv.org/html/2607.19190v3
- Agentic RSR: Real-to-Sim-to-Real through Scene Reconstruction and Execution-Grounded Robot Policies — https://arxiv.org/html/2610.10479v1
- BVB: Benchmarking Agentic Video Understanding via Programmatic Reconstruction in Blender — https://arxiv.org/pdf/2609.15478
- SceneCraft: An LLM Agent for Synthesizing 3D Scene as Blender Code — https://arxiv.org/pdf/2403.01248
- GPT-6 Astra（发布时间线）— https://en.wikipedia.org/wiki/GPT-6_Astra
- agent × real-to-sim-to-real 论文清单 — https://github.com/YanjieZe/agent-real2sim2real
