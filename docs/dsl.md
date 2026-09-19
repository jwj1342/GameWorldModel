# 场景程序 DSL

> 4D 场景程序是整个系统的中间表示：VLM 写它，编译器读它，反馈报告指向它的节点。本文给出文本形式（论文用）与 JSON 形式（实现用）、最小词表、校验规则、到 three.js 的编译映射，以及与 threejs-scene-factory Scene Spec 的对应。

## 1. 设计要求

- 紧凑、可 token 化、可 diff：VLM 一次输出或以 JSON Patch 修订。
- 可校验：JSON Schema 管结构，语义校验管 id 唯一、支撑关系、参数范围、素材可解析。
- 可编译到 three.js，也不排斥其他引擎：程序只描述几何、位姿、运动、事件与绑定，不含渲染细节。
- 词表从小开始：先用能覆盖测试片段的最小词表跑通，按消融结果扩展。Mage（2026.05）的实证支持这一路线：直接自然语言到代码运行成功率高但机制保真度差，经结构化中间表示后大幅改善。

## 2. 文本形式（论文中的可读形式）

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

## 3. JSON 形式（实现用）

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

## 4. 最小词表

### 4.1 几何 `geom.kind`

| kind | 字段 | 说明 |
|---|---|---|
| `primitive` | `shape ∈ {box, sphere, cylinder, cone, plane}`、尺寸、`material` | 编译成 three.js 内建几何；永远可用 |
| `asset` | `query`（检索文本）、`extent`（目标包围盒） | 素材层检索 GLB，按 extent 缩放；失败回退 primitive |
| `generated` | `ref`（生成产物路径）、`extent` | 由 image-to-3D 生成的 GLB；仅 hero 物体 |
| `heightfield`（仅 static） | `ref`、`scale` | 高度图地形 |

### 4.2 运动 `motion.type`

| type | 参数 | `pose(t)` 语义 |
|---|---|---|
| `static` | — | 恒等 |
| `trajectory` | `keyframes[{t, pos, quat?}]`、`interp ∈ {linear, catmull_rom}` | 关键帧插值 |
| `revolute` | `axis`、`pivot`、`range_deg`、`schedule[{t, to_deg}]` 或 `rate_dps` | 绕固定轴转动；schedule 之间线性过渡 |
| `prismatic` | `axis`、`range`、`schedule[{t, to}]` | 沿固定轴平移 |
| `periodic_translate` | `axis`、`amp`、`period`、`phase` | 正弦往复平移 |
| `periodic_rotate` | `axis`、`amp_deg`、`period`、`phase` | 正弦往复转动 |
| `spin` | `axis`、`rate_dps` | 匀速自转 |

### 4.3 事件 `events[].type`

| type | 参数 | 运行时行为 |
|---|---|---|
| `despawn_on_contact` | `with`（`player` 或物体 id） | 接触后隐藏并计数 |
| `trigger_on_enter` | `volume`、`target`、`action ∈ {open, start, stop}` | 进入体积后触发目标物体的 schedule |

### 4.4 绑定 `binding`

| 模板 | 槽位 |
|---|---|
| `platformer_3p` | `player_spawn`、`goal_volume`、`walkable`（`auto_navmesh` 或显式 id 列表）、`hazards[]`、`collectibles[]` |
| `topdown_survival`（后续） | `player_spawn`、`arena_bounds`、`enemy_spawns[]`、`enemy_paths[]`、`pickups[]` |

其他四个模板（赛车、塔防、简单射击、解谜）的槽位草案见 `runtime.md` 第 5 节。

### 4.5 后续扩展

`spawn / despawn` 的时刻表、`prismatic` 的周期形式、层级关节（父子链）、`trigger` 的更多动作、材质字典的扩展、`camera` 的参数化跟随规则（第三人称跟随、环绕）。扩展只加词，不改结构。

## 5. 校验规则

**结构校验**（JSON Schema）：字段类型、枚举、数组长度、必填项。

**语义校验**（编译器）：

- id 全局唯一；`support` 与 `events[].with` 引用的 id 存在。
- 支撑关系无环；被支撑物体的底面与支撑面的距离在阈值内，否则给出警告而非错误。
- 运动参数范围：`period > 0`，`range_deg` 单调，`schedule` 的 `t` 单调递增且不超过 `meta.duration`。
- `extent` 各维为正；`asset` 的 `query` 非空。
- 绑定槽位引用的 id 存在且类别合理（收集物不能是 static）。
- `camera.keyframes` 的 `t` 单调，至少两帧。

校验失败输出结构化错误列表 `{ path, code, message, suggestion }`，直接回喂 VLM；SpatialGrammar（2026.04）的经验是编译期约束反馈能让小模型接近大模型。

## 6. 编译映射

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

## 7. 与 threejs-scene-factory Scene Spec 的对应

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

## 8. 程序与残差的划分规则（初版）

表示为程序节点：有运动的物体、语义上可交互的物体（门、开关、可拾取物）、重复出现的实例、规则几何体。表示为残差：大面积不规则静态几何（植被、岩石细节）、远景。用程序覆盖率（被程序节点解释的像素或几何比例）和残差体积衡量划分质量。
