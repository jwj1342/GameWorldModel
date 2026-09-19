# 系统架构

> 视频进、游戏出的整条流水线，各模块的输入输出、产物目录与接口契约。方法层面的论述见 `../RP.md` 第 3 节；第一阶段的实施顺序见 `baseline.md`。

## 1. 一句话

输入一段 5~15 秒的单目视频，感知模块把它变成结构化证据，VLM 据证据写出一份带时间轴的场景程序，编译器把程序变成 three.js 场景，无头浏览器把场景渲染回视频视角与原片比对并产出结构化错误报告，VLM 据报告修订程序，通过后绑定玩法模板，输出可以在浏览器里玩的游戏。

## 2. 数据流

```
视频 ─► [感知] ─► evidence.json ─► [程序生成 VLM] ─► program.json ─► [编译] ─► game/ ─► [无头运行] ─► [比对与批评]
                                       ▲                                                                  │
                                       └────────── 结构化错误报告（逐物体 pass/fail 子句 + checklist）◄────┘  最多三轮
                                                                                                          │ 通过
                                              [素材层] ◄──────────────── [玩法绑定] ◄─────────────────────┘
                                                                                                          │
                                                                                          game/（可玩）+ playtest/
```

三条原则贯穿全图：

- 生成侧只产出数据（程序 JSON），不产出代码；运行时内核是固定代码。
- 校验先于渲染：JSON Schema 与语义校验失败的程序直接把错误回喂给 VLM，不进入渲染比对。
- 反馈必须结构化：数值指标先生成逐物体子句，VLM 只对失败子句做定位与建议，不做自由评论。

## 3. 模块一览

| 模块 | 输入 | 输出 | 实现 | 运行位置 | 细节文档 |
|---|---|---|---|---|---|
| 感知 | 视频 | 位姿、内参、深度、动态掩码、实例掩码与跨帧 ID、点轨迹 | Python：ViPE、SAM 3.1、SpatialTrackerV2 | GPU 作业 | `perception.md` |
| 证据提取 | 感知输出 | `evidence.json`：逐物体 OBB 序列、运动类型猜测、接触关系、置信度、静态平面候选 | Python：numpy、螺旋拟合 | 同一作业的 CPU 段 | `perception.md` |
| 程序生成 | 关键帧、`evidence.json`、DSL schema、上一轮错误报告 | `program.json` 或 JSON Patch | Python：Pydantic AI；`claude-opus-5` 或 vLLM 上的 Qwen3-VL | 作业内经代理调 API，或本地 vLLM | `models.md`、`feedback.md` |
| 编译 | `program.json` | `game/`：免打包 ESM 模块 + 资源清单 | TypeScript / Node 20：schema 校验、语义校验、素材解析、代码生成 | 登录节点或作业均可 | `dsl.md` |
| 运行时内核 | `game/` | 浏览器中的可玩场景；`window.__game` 控制面 | three.js、Rapier、recast-navigation、miniplex | 浏览器或 headless Chromium | `runtime.md` |
| 比对与批评 | `game/`、视频关键帧、`evidence.json` | 指标 JSON、逐物体子句、VLM checklist | Node（Playwright 渲染）+ Python（DINOv3、DreamSim、IoU、深度误差） | CPU 作业即可 | `feedback.md` |
| 玩法绑定 | `program.json`、玩法提示 | `binding` 段：模板名 + 槽位 | Python 规则 + 少量 VLM 判断；recast 生成可行走面 | 作业 | `runtime.md` |
| 素材层 | 类别、裁剪图、尺寸 | GLB 路径或 primitive 参数 | Python：CLIP/SigLIP + FAISS；可选 TRELLIS.2 | 检索 CPU；生成 GPU | `assets.md` |

## 4. 产物目录

```
out/<clip_id>/
  frames/                # 抽帧（关键帧索引写在 keyframes.json）
  evidence.json          # 感知证据
  program.json           # 当前最优的 4D 场景程序
  rounds/
    r0/ program.json  compile.log  feedback/  critic.json
    r1/ ...
  game/                  # 编译产物：index.html + modules/ + assets/ + vendor/
  feedback/              # 最终一轮的关键帧 rgb / depth / id 渲染与 metrics.json
  playtest/              # 键鼠轨迹、回放帧序列、VLM 评审、state 日志
  report.md              # 汇总：每轮指标变化、最终子句、试玩结论
```

## 5. 接口契约

### 5.1 evidence.json

| 字段 | 含义 | 来源 |
|---|---|---|
| `camera.intrinsics` | 焦距、主点、畸变 | ViPE |
| `camera.poses[t]` | 逐帧世界到相机的位姿 | ViPE |
| `camera.scale_note` | 近度量尺度的可信度说明 | ViPE |
| `static.planes[]` | 地面与墙体候选：法向、偏移、覆盖像素比例 | 静态深度平面拟合 |
| `objects[].id` | 实例 ID，跨帧稳定 | SAM 3.1 |
| `objects[].class_guess` / `confidence` | 类别猜测与置信度 | SAM 3.1 提示词 + VLM |
| `objects[].obb[t]` | 逐帧有向包围盒：中心、四元数、尺寸 | 掩码 × 深度反投影 + PCA |
| `objects[].is_dynamic` | 是否有相对世界系的运动 | 轨迹动静标签 + OBB 序列 |
| `objects[].motion_guess` | 类型（static / trajectory / revolute / prismatic / periodic / spin）、轴、枢轴、周期、幅度、置信度 | 螺旋拟合 |
| `objects[].contacts[]` | 接触对象与时间区间 | OBB 底面到静态深度的距离 |
| `objects[].best_frame` | 面积最大且遮挡最少的帧，供素材检索裁剪 | 掩码统计 |
| `keyframes[]` | 送给 VLM 的关键帧索引及选择理由 | 相机运动与物体事件均匀抽样 |

### 5.2 program.json

见 `dsl.md`。生成侧第一轮输出完整 JSON，第二轮起输出 JSON Patch。

### 5.3 game/ 与 `window.__game`

`game/` 是免打包 ESM：`index.html` 用 importmap 指向本地 `vendor/`（three、Rapier、recast、miniplex），`modules/` 下是编译器生成的 `scene.js`、`motions.js`、`events.js`、`binding.js`，内核代码在 `vendor/kernel/`。运行时暴露：

| 方法 | 作用 |
|---|---|
| `__game.step(dt)` | 推进一个固定步长；不依赖 requestAnimationFrame |
| `__game.render(pass)` | `pass ∈ {rgb, depth, id}`；返回 base64 PNG 或写入离屏 RenderTarget 后由 Playwright 截图 |
| `__game.state()` | 稳定 schema：玩家位姿、每个物体位姿与运动相位、事件计数、胜负标志、帧号、时间 |
| `__game.input(keys)` | 设置当前帧按键集合，供轨迹回放 |
| `__game.reset(seed)` | 复位并设置随机种子 |
| `__game.mode(m)` | `replay`：相机跟随程序里的相机关键帧；`play`：模板相机接管 |

### 5.4 反馈报告

`feedback/metrics.json` 每关键帧、每物体一行：mask IoU、质心像素偏差、深度 SILog、是否在视野内；全局 DINOv3 余弦与 DreamSim；相机重投影误差。规则层由此生成子句列表：

```
{ "object_id": "platform_3", "check": "pose", "status": "FAIL",
  "evidence": { "iou": 0.31, "centroid_err_m": 0.8, "t_range": [6.0, 8.0] },
  "hint": "suspect period" }
```

VLM 批评的输出限定为：

```
{ "object_id": "...", "issue": "missing | extra | pose | scale | timing | class",
  "evidence": "...", "suggested_edit": { JSON Patch 片段 } }
```

### 5.5 playtest/

`trajectory.json`（按帧的按键集合，由 navmesh 路径生成）、`frames/`（2 fps 抽帧）、`states.jsonl`（每帧 `__game.state()`）、`review.json`（VLM 评审：可玩性、视觉一致性、明显缺陷）、`verdict.json`（规则判定：是否到达目标、是否卡死、帧率）。

## 6. 运行位置与作业划分

| 作业类型 | 资源 | 跑什么 |
|---|---|---|
| 感知作业 | 1×L40S，16 CPU，64 GB，约 30 分钟/批 | ViPE、SAM 3.1、SpatialTrackerV2、证据提取；批量处理多条 clip |
| 生成与反馈作业 | CPU 节点 8 CPU，32 GB；或复用感知作业尾段 | 程序生成（API）、编译、Playwright 渲染、指标计算（DINOv3 在 CPU 可跑，GPU 更快）、试玩回放 |
| 自托管模型作业 | 2×L40S | vLLM 起 Qwen3-VL-32B FP8，作为程序生成的备选 |
| 素材构建作业 | 1×L40S | 缩略图渲染、CLIP 嵌入、TRELLIS.2 生成 |

登录节点只做 npm 安装、打包、编译与提交作业。网络与渲染实测见 `cluster.md`。

## 7. 与研究提案的对应

| RP 第 3 节 | 本架构中的模块 |
|---|---|
| 3.2 感知与证据提取 | 感知 + 证据提取 |
| 3.3 分阶段程序解码 | 程序生成（收成三阶段：相机与静态结构 → 物体几何与位姿 → 运动与事件） |
| 3.4 执行反馈精修 | 编译 + 无头运行 + 比对与批评（推理时用法；训练时用法在 baseline 之后） |
| 3.5 残差补全 | 素材层的可选残差分支（Spark），baseline 阶段只预留接口 |
| 第 6 节 双输入可控游戏生成 | 玩法绑定 |
