# 运行时与游玩层

> 编译目标为什么是 three.js，运行时内核如何组织，玩法模板与槽位怎么定义，无头渲染与确定性步进怎么做，试玩验收怎么做。版本号为 2026-09-18 查询结果。

## 1. 运行时选型

| 引擎 | 版本 | 生态（stars / npm 周下载） | 无头渲染 | 物理 | 结论 |
|---|---|---|---|---|---|
| **three.js** | r186 | 115.6k / 14.8M | headless Chromium 直接可跑 | 外接 Rapier / Jolt / cannon | **选用**：LLM 训练数据最丰富；纯库无编辑器；深度与 ID pass 可在 JS 层完全控制 |
| Babylon.js | 9.27 | 26.1k / 314k | `NullEngine` 只跑逻辑不出图，渲染仍需 Chromium | 内建 Havok | 唯一值得考虑的替代，生态差一个量级 |
| PlayCanvas | 2.22 | 16.8k / 62k | 同 three | 内建 ammo | 场景 JSON 与云端编辑器强绑定 |
| Godot Web | 4.7 | 117k / – | Web 导出需 SharedArrayBuffer；`--headless` 剥离渲染 | 内建 | 生成目标是 GDScript + .tscn；无法从外部注入深度/ID pass；适合最终导出而非中间目标 |
| Phaser | 4.2 | 40.3k / 308k | 同 three | 2D | 纯 2D，不匹配 |

旁证：WorldCoder-Bench（2026）以 three.js 为浏览器 3D 世界生成的唯一基底；Rosebud 的 3D 内核 Roseblox 与 Bitmagic 都选 three.js；npm 下载量 three 与 babylon 之比约 47 : 1。

不用 react-three-fiber：它的 JSX 像声明式 DSL，但引入 React 运行时与 hooks 语义，无头调试与帧级确定性更绕，配套库更新也落后 Rapier 主库。

## 2. three.js 游戏栈

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

## 3. 内核设计

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

## 4. 无头渲染

**启动参数**（Playwright 1.63 + Chromium 1243，已在 Vulcan 实测，见 `cluster.md`）：

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

## 5. 玩法模板与槽位

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

## 6. 试玩验收

不做实时游玩 agent：VideoGameBench 的经验是推理延迟主导失败；BALROG 发现给图像后多数模型反而变差。采用 GameCraft-Bench 的做法：

1. 由 navmesh 路径生成一条出生点到目标的按键序列（含跳跃时机），写成 `trajectory.json`。
2. 无头运行，逐帧 `__game.input` 注入，记录每帧 `state()`，2 fps 抽帧存图。
3. 规则判定：是否到达目标、是否卡死（位置长时间不变）、帧率是否达标、收集物计数是否变化。
4. VLM 评审抽帧序列：可玩性、视觉一致性、明显缺陷，输出结构化评分；作辅助而非主判据。

## 7. 参考仓库

| 仓库 | 用途 |
|---|---|
| rosebudai/roseblox（MIT，2026-09 活跃） | 内核模式：three + Rapier + miniplex，systems 优先级区间，buildless |
| isaac-mason/sketches | three + Rapier + recast 的现代栈样例，质量高 |
| swift502/Sketchbook（已归档） | 第三人称角色与载具控制参考 |
| pmndrs/racing-game | 赛车模板参考（R3F + cannon，需现代化） |
| Casmo/tower-defense | 塔防模板参考 |
| dylanebert/VibeGame | 面向 LLM 的声明式标签 + ECS 思路 |
| heagandev/threejs-agent-starter | 面向 agent 的 three.js 起步模板 |

## 8. 为什么不用现成的声明式场景格式

three.js 自带的 JSON Object/Scene format 只描述静态层次且几何内联冗长；glTF 是资产交换格式，可用 vendor extension 挂元数据（Needle Engine 的做法），但不适合承载运动与绑定；A-Frame 的 HTML 实体组件语法对 LLM 友好但绑定 DOM 与 WebXR；VibeGame 最接近但近一年未更新。结论是自研 JSON DSL，资产用 GLB 承载并以 URI 引用。
