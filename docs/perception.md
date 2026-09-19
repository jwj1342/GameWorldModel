# 感知与证据提取

> 从单目视频得到 `evidence.json` 的模型选择与算法。模型对比数据来自 2026-09-18 的调研（`../survey/05`），硬件按 L40S 48 GB 估算。

## 1. 目标

把视频变成 VLM 能直接使用的结构化证据：相机内参与逐帧位姿、静态结构候选、每个物体的逐帧有向包围盒、运动类型猜测、接触关系、置信度，以及供 VLM 看的关键帧索引。证据字段定义见 `architecture.md` 5.1。

## 2. 相机位姿、深度与动态掩码

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

## 3. 实例分割、跟踪与检测

| 任务 | 选用 | 备选 | 说明 |
|---|---|---|---|
| 实例分割 + 跨帧 ID | **SAM 3.1**（2026-03；848M；文本短语 / 点 / 框 / 示例提示；SAM License 允许商用，HF 需申请） | Grounded-SAM-2（SAM 2.1 + Grounding DINO 1.0，全 Apache-2.0） | 一个模型替代检测 + 分割 + 跟踪；单次前向可跟踪 16 个物体 |
| 类别候选 | VLM 对首帧列出名词短语 → 作为 SAM 3.1 提示 | OWLv2（Apache）、Florence-2（MIT）、Qwen3-VL grounding | Grounding DINO 1.5/1.6 与 DINO-X 仅 API |
| 点轨迹 | **SpatialTrackerV2**（3D 轨迹 + 动静标签；CC BY-NC） | CoTracker3（CC BY-NC）；TAPIR / BootsTAPIR（Apache，商用换这个） | 每个实例掩码内取点 |
| 记忆式 VOS | 不需要 | DEVA、Cutie（MIT） | 已被 SAM 3 覆盖 |

## 4. 证据提取算法

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

## 5. 资源预算（300 帧、512 px、1×L40S）

| 步骤 | 估算 |
|---|---|
| ViPE | 1~2 分钟，显存有界 |
| SAM 3.1（约 10 个物体） | 1~2 分钟 |
| SpatialTrackerV2 | 1~3 分钟 |
| 证据提取（CPU） | < 1 分钟 |
| 合计 | 约 5~8 分钟/条；批量处理时一次作业跑多条 |

## 6. 环境与许可

- 虚拟环境用 `python/3.11.5` 模块建在 `/project`，权重预下载到 `/project`，`HF_HOME` 指向 `$SCRATCH`；作业内经代理可访问 HuggingFace 与 GitHub（见 `cluster.md`）。
- 论文用途下 SpatialTrackerV2 与 CoTracker3 的非商用许可可接受；若需商用换 TAPIR。ViPE、SAM 3.1、Depth Anything 3 Small/Base 均可商用。
- Kubric MOVi 片段带真值，用来校验 ViPE 尺度、OBB 精度与运动分类；见 `models.md` 第 6 节的数据集表。

## 7. 与 4D 重建的关系

MoSca、Shape of Motion、4DGaussians 等动态 Gaussian 方法输出不可编辑的高斯，不是本系统的证据来源，而是残差层与对比基线：静态背景一份 splat，每个动态实例一份 splat，由程序驱动变换，用 Spark 在 three.js 中渲染。4DGaussians 有逐时刻 PLY 导出脚本，Shape of Motion 可小改导出。
