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

| 模块 | 输入 | 输出 | 实际用的东西 | 跑在哪 |
|---|---|---|---|---|
| 感知 | 视频 | 位姿、内参、深度、实例掩码与跨帧 ID | VGGT；Grounding DINO + SAM 2.1（后端可换，见 `gwm/perception/base.py` 的两个 Protocol） | CPU 够用，有 GPU 更快 |
| 证据提取 | 感知输出 | `evidence.json`：逐物体 OBB 序列、运动类型猜测、接触关系、地面平面 | numpy、螺旋拟合 | CPU |
| 程序生成 | 关键帧、`evidence.json`、DSL schema、上一轮错误报告 | `program.json` | 手写的三阶段循环；OpenRouter 上的 GLM-4.6V（可换，见 `configs/vlm/`） | 调 API |
| 编译 | `program.json` | `game/`：免打包 ESM + `kernel_config.json` | Python：schema 校验、语义校验、素材解析、打包 | CPU |
| 运行时内核 | `game/` | 浏览器里的可玩场景；`window.__game` 控制面 | three.js r186、Rapier | 浏览器或 headless Chromium |
| 比对与批评 | `game/`、视频关键帧、`evidence.json` | 指标 JSON、逐物体子句、模型给的 JSON 补丁 | Playwright 渲染 + numpy 算指标 + DINOv2 | CPU |
| 玩法绑定 | `program.json` | `binding` 段：模板名 + 槽位 | 纯规则，最大静态顶面当可行走面 | CPU |
| 素材层 | 类别、尺寸 | primitive 参数 | 目前只有几何体替身，库检索和生成是后续阶段 | CPU |

每一层为什么这么选，见 [design-notes.md](design-notes.md)。

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

见本文第 7 节。生成侧第一轮输出完整 JSON，第二轮起输出 JSON Patch。

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

登录节点只做 npm 安装、打包、编译与提交作业。网络与渲染实测见 `../RUNNING.md`。

## 7. 场景程序 DSL

> 4D 场景程序是整个系统的中间表示：VLM 写它，编译器读它，反馈报告指向它的节点。本文给出文本形式（论文用）与 JSON 形式（实现用）、最小词表、校验规则、到 three.js 的编译映射，以及与 threejs-scene-factory Scene Spec 的对应。

### 1. 设计要求

- 紧凑、可 token 化、可 diff：VLM 一次输出或以 JSON Patch 修订。
- 可校验：JSON Schema 管结构，语义校验管 id 唯一、支撑关系、参数范围、素材可解析。
- 可编译到 three.js，也不排斥其他引擎：程序只描述几何、位姿、运动、事件与绑定，不含渲染细节。
- 词表从小开始：先用能覆盖测试片段的最小词表跑通，按消融结果扩展。Mage（2026.05）的实证支持这一路线：直接自然语言到代码运行成功率高但机制保真度差，经结构化中间表示后大幅改善。

### 2. 文本形式（论文中的可读形式）

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

### 3. JSON 形式（实现用）

一个文件一个场景，先过 JSON Schema 再编译。

```jsonc
{
  "meta": { "clip": "clip_0042", "fps": 30, "duration": 8.0, "units": "m", "up": "y" },
  "camera": {
    "intrinsics": { "fov_deg": 62, "aspect": 1.777 },
    "keyframes": [ { "t": 0.0, "pos": [0, 1.6, 5], "quat": [0, 0, 0, 1] } ],
    "interp": "catmull_rom"
  },
  "static": [
    { "id": "ground_0", "geom": { "kind": "plane", "size": [40, 40] },
      "pose": { "pos": [0, 0, 0], "quat": [0, 0, 0, 1] }, "material": "grass" },
    { "id": "wall_1", "geom": { "kind": "box", "extent": [10, 3, 0.3] },
      "pose": { "pos": [0, 1.5, -5], "quat": [0, 0, 0, 1] }, "material": "stone" },
    { "id": "terrain_2", "geom": { "kind": "heightfield", "ref": "terrain_2.png", "scale": [40, 6, 40] },
      "pose": { "pos": [0, 0, 0], "quat": [0, 0, 0, 1] }, "material": "grass" }
  ],
  "objects": [
    {
      "id": "door_7", "class": "door", "confidence": 0.82,
      "geom": { "kind": "asset", "query": "wooden door", "extent": [1.0, 2.1, 0.1] },
      "pose": { "pos": [2, 1.05, -4.85], "quat": [0, 0, 0, 1] },
      "motion": { "type": "revolute", "axis": [0, 1, 0], "pivot": [1.5, 0, -4.85],
                  "range_deg": [0, 90], "schedule": [ { "t": 3.2, "to_deg": 90 } ] },
      "support": "ground_0"
    },
    {
      "id": "platform_3", "class": "platform",
      "geom": { "kind": "primitive", "shape": "box", "extent": [2, 0.3, 2], "material": "metal" },
      "pose": { "pos": [-3, 1, 0], "quat": [0, 0, 0, 1] },
      "motion": { "type": "periodic_translate", "axis": [0, 1, 0], "amp": 1.5, "period": 4.0, "phase": 0.0 }
    },
    {
      "id": "coin_12", "class": "coin",
      "geom": { "kind": "asset", "query": "gold coin", "extent": [0.4, 0.4, 0.05] },
      "instances": [ { "pos": [1, 1, 1], "quat": [0, 0, 0, 1] }, { "pos": [2, 1, 1], "quat": [0, 0, 0, 1] } ],
      "motion": { "type": "spin", "axis": [0, 1, 0], "rate_dps": 180 },
      "events": [ { "type": "despawn_on_contact", "with": "player" } ]
    }
  ],
  "binding": {
    "template": "platformer_3p",
    "slots": {
      "player_spawn": [0, 0.5, 3],
      "goal_volume": { "pos": [6, 1, -3], "extent": [1, 2, 1] },
      "walkable": "auto_navmesh",
      "hazards": [],
      "collectibles": ["coin_12"]
    }
  },
  "residual": null
}
```

### 4. 最小词表

#### 4.1 几何 `geom.kind`

| kind | 字段 | 说明 |
|---|---|---|
| `primitive` | `shape ∈ {box, sphere, cylinder, cone, plane}`、尺寸、`material` | 编译成 three.js 内建几何；永远可用 |
| `asset` | `query`（检索文本）、`extent`（目标包围盒） | 素材层检索 GLB，按 extent 缩放；失败回退 primitive |
| `generated` | `ref`（生成产物路径）、`extent` | 由 image-to-3D 生成的 GLB；仅 hero 物体 |
| `heightfield`（仅 static） | `ref`、`scale` | 高度图地形 |

#### 4.2 运动 `motion.type`

| type | 参数 | `pose(t)` 语义 |
|---|---|---|
| `static` | — | 恒等 |
| `trajectory` | `keyframes[{t, pos, quat?}]`、`interp ∈ {linear, catmull_rom}` | 关键帧插值 |
| `revolute` | `axis`、`pivot`、`range_deg`、`schedule[{t, to_deg}]` 或 `rate_dps` | 绕固定轴转动；schedule 之间线性过渡 |
| `prismatic` | `axis`、`range`、`schedule[{t, to}]` | 沿固定轴平移 |
| `periodic_translate` | `axis`、`amp`、`period`、`phase` | 正弦往复平移。**`pose.pos` 是振荡中心**，求值为 `pose.pos + axis*amp*sin(2πt/period + phase)` |
| `periodic_rotate` | `axis`、`amp_deg`、`period`、`phase`、`pivot?` | 正弦往复转动，幅度是单边幅度；不给 `pivot` 就绕自身中心 |
| `spin` | `axis`、`rate_dps` | 匀速自转 |

#### 4.3 事件 `events[].type`

| type | 参数 | 运行时行为 |
|---|---|---|
| `despawn_on_contact` | `with`（`player` 或物体 id） | 接触后隐藏并计数 |
| `trigger_on_enter` | `volume`、`target`、`action ∈ {open, start, stop}` | 进入体积后触发目标物体的 schedule |

#### 4.4 绑定 `binding`

| 模板 | 槽位 |
|---|---|
| `platformer_3p` | `player_spawn`、`goal_volume`、`walkable`（`auto_navmesh` 或显式 id 列表）、`hazards[]`、`collectibles[]` |
| `topdown_survival`（后续） | `player_spawn`、`arena_bounds`、`enemy_spawns[]`、`enemy_paths[]`、`pickups[]` |

其他四个模板（赛车、塔防、简单射击、解谜）的槽位草案见 `design-notes.md` 运行时一节。

#### 4.5 后续扩展

`spawn / despawn` 的时刻表、`prismatic` 的周期形式、层级关节（父子链）、`trigger` 的更多动作、材质字典的扩展、`camera` 的参数化跟随规则（第三人称跟随、环绕）。扩展只加词，不改结构。

### 5. 校验规则

**结构校验**（JSON Schema）：字段类型、枚举、数组长度、必填项。

**语义校验**（编译器）：

- id 全局唯一；`support` 与 `events[].with` 引用的 id 存在。
- 支撑关系无环；被支撑物体的底面与支撑面的距离在阈值内，否则给出警告而非错误。
- 运动参数范围：`period > 0`，`range_deg` 单调，`schedule` 的 `t` 单调递增且不超过 `meta.duration`。
- `extent` 各维为正；`asset` 的 `query` 非空。
- 绑定槽位引用的 id 存在且类别合理（收集物不能是 static）。
- `camera.keyframes` 的 `t` 单调，至少两帧。

校验失败输出结构化错误列表 `{ path, code, message, suggestion }`，直接回喂 VLM；SpatialGrammar（2026.04）的经验是编译期约束反馈能让小模型接近大模型。

### 6. 编译映射

| DSL 节点 | three.js / Rapier 产物 |
|---|---|
| `static[]` primitive | `Mesh` + Rapier 固定刚体（`cuboid` / `ball` / `cylinder` / `heightfield` 碰撞体） |
| `objects[]` primitive / asset / generated | `Group`（内含 Mesh 或 GLB）+ Rapier 运动学刚体（kinematicPositionBased）；`instances` 用 `InstancedMesh` 或多份 Group |
| `motion` | `motions.js` 中一个纯函数 `pose(t) -> {pos, quat}`；运行时 system 每步写入 Group 与运动学刚体 |
| `events` | `events.js` 中的碰撞回调或体积检测 system |
| `camera.keyframes` | 回放模式的相机 system（Catmull-Rom）；游戏模式由模板相机接管 |
| `binding` | `binding.js`：模板名 + 槽位对象；模板 system 读取槽位 |
| `residual` | Spark 的 `SplatMesh`（后续） |
| 每个物体 | 一个 24 位 ID 颜色，供 `render("id")` 使用 |

### 7. 与 threejs-scene-factory Scene Spec 的对应

| Scene Spec 字段 | 本 DSL | 备注 |
|---|---|---|
| `objects[].geometry {shape, size, radius, height}` | `geom.kind = primitive` | 形状集合相近；本 DSL 增加 asset / generated / heightfield |
| `objects[].transform {position, rotation(euler), scale}` | `pose {pos, quat}` + `extent` | 用四元数与包围盒替代欧拉角与 scale |
| `objects[].children` | 暂无层级 | 后续扩展为关节链 |
| `interactions: hinge / rotate / drag / toggle / hover` | `motion: revolute / spin` + `events` | Scene Spec 的交互由用户触发，本 DSL 的运动由时间轴驱动；用户交互交给玩法模板 |
| `camera {position, target, fov}` | `camera {intrinsics, keyframes}` | 单相机变成相机轨迹 |
| `lights[]` | 暂由模板提供 | 光照不是从视频推断的目标 |
| 无 | `binding`、`residual`、`meta` | 新增 |

可直接借用的做法：校验器的写法（id 唯一、数值范围、颜色格式等语义检查）；"校验 → 生成 → 构建全部通过才出包"的门控与 `validation-report.json`；`catalog.json` 式声明式配方，用来写 primitive 兜底库（门 = 框 + 板 + 绕 Y 轴的 pivot，矿车 = 箱体 + 四个圆柱轮）。

### 8. 程序与残差的划分规则（初版）

表示为程序节点：有运动的物体、语义上可交互的物体（门、开关、可拾取物）、重复出现的实例、规则几何体。表示为残差：大面积不规则静态几何（植被、岩石细节）、远景。用程序覆盖率（被程序节点解释的像素或几何比例）和残差体积衡量划分质量。

## 8. 错误处理

所有可恢复错误写 `errors.jsonl`：`{time, stage, code, message, recoverable, action_taken}`。

| 类别 | 例子 | 策略 |
|---|---|---|
| 基础设施 | 作业超时、OOM、vLLM 端点不可达、Chromium 起不来 | 阶段幂等与 `--resume`；端点重试带退避并读最新端点文件；Chromium 失败切 GPU / CPU 另一路径 |
| 感知失败 | 后端崩溃、位姿发散、没检出物体 | 后端链回退（ViPE → VGGT → 静态相机；SAM 3.1 → Grounded-SAM-2）；无物体时进入"只有静态结构"模式仍出可玩场景 |
| 模型输出 | JSON 不合法、schema 不过、语义不过、超长 | 约束解码；修复循环 ≤ 3；仍失败则证据直译；超长则减少关键帧 |
| 编译 | 引用缺失、参数越界 | 结构化错误列表回喂 Writer |
| 运行时 | 内核异常、WASM 装载失败、渲染超时 | harness 捕获 console / pageerror 写日志；作为该轮失败原因 |
| 反馈不收敛 | 分数震荡 | 回退到历史最优；上限轮数后接受最优并在报告标注 |
| 试玩失败 | 到不了目标、卡死 | 绑定回退（补地面、放宽目标）；仍失败记录并继续出包 |
| 预算 | 调用次数、GPU 时间超限 | 配置中的预算；超限即停止循环并出包 |

原则：任何一类错误都不能让管线空手而归；最差也输出"证据直译 + 静态场景"的可玩包，并在 `report.md` 写明退化路径。

## 9. 哪里用模型，哪里不用

| 角色 | 用什么 | 为什么 |
|---|---|---|
| Namer（首帧名词短语） | Qwen3.5-27B，低温度，JSON 约束 | 轻任务，同一端点 |
| Writer（三阶段生成） | Qwen3.5-27B，结构化输出，关闭 thinking | 用户指定；thinking 与 JSON 约束冲突 |
| Critic（子句 → 建议） | 同 Writer | SEIG 用一个模型兼任 generator 与 verifier；避免两套提示词漂移 |
| 试玩评审 | 同 Writer，仅一句话 | 规则判定为主 |
| 感知聚合、编译、绑定、指标 | 纯代码 | 有确定算法就不用模型 |

不做角色扮演式多智能体（依据见 `design-notes.md` 视觉反馈一节）。MVP 用手写 evaluator-optimizer 循环 + pydantic / jsonschema 校验；`VLMClient` 协议保证之后可接 Claude 做对比或换 Pydantic AI。消融开关：`no_evidence`（Writer 只看帧）、`no_feedback`（一轮）、`single_stage`（一次出全程序）。

## 10. 与研究提案的对应

| RP 第 3 节 | 本架构中的模块 |
|---|---|
| 3.2 感知与证据提取 | 感知 + 证据提取 |
| 3.3 分阶段程序解码 | 程序生成（收成三阶段：相机与静态结构 → 物体几何与位姿 → 运动与事件） |
| 3.4 执行反馈精修 | 编译 + 无头运行 + 比对与批评（推理时用法；训练时用法在 baseline 之后） |
| 3.5 残差补全 | 素材层的可选残差分支（Spark），baseline 阶段只预留接口 |
| 第 6 节 双输入可控游戏生成 | 玩法绑定 |
