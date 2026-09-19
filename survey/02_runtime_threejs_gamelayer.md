# 调研 02：运行时、three.js 游戏栈、无头渲染与游玩层（2026-09-18）

> 由调研 agent 生成，版本号来自 npm registry / GitHub API 当日查询。未能核实的项标注 **未核实**。
> 补充：报告中"Vulcan 上 Vulkan ICD 未核实"一项已由同日 Slurm 探测补上：L40S 节点上 `--use-angle=vulkan` 得到 `ANGLE (NVIDIA, Vulkan 1.4, NVIDIA L40S)`，硬件加速可用；见 RP 第 9 节。

## Q1. 编译目标运行时选型

| 引擎 | 当前版本 | Stars / 最近 push | npm 周下载 | 无头渲染 | 物理 | glTF | 备注 |
|---|---|---|---|---|---|---|---|
| **three.js** | r186 (0.186.0, 2026-09-08) | 115.6k / 2026-09-18 | 14.8M | Headless Chromium 直接可跑 | 无内建，外接 Rapier/Jolt/cannon | GLTFLoader 一流 | 纯库、无编辑器；ESM + importmap 可免打包 |
| Babylon.js | 9.27.1 (2026-09-18) | 26.1k | 314k | `NullEngine` 只跑逻辑不渲染 | 内建 Havok 插件 | 一流 | 有 Inspector、`.babylon` JSON |
| PlayCanvas | engine 2.22.2 | 16.8k | 62k | 同 three.js | 内建 ammo.js | 一流 | 场景 JSON 与云端 Editor 强绑定 |
| Godot (Web) | 4.7.2 (2026-08-18) | 117k | – | Web 导出需 SharedArrayBuffer；`--headless` 剥离渲染 | 内建 | 一流 | 生成目标是 GDScript + .tscn；无法从外部注入深度/ID pass |
| Phaser | 4.2.1 | 40.3k | 308k | 同 three.js | 2D | 不适用 | 纯 2D |

LLM 熟悉度证据：WorldCoder-Bench（arXiv 2606.01869，2,026 个浏览器 3D 世界生成任务）以 three.js 为唯一基底；npm 周下载 three : babylon ≈ 47 : 1；Rosebud（Roseblox）与 Bitmagic 的 3D 端都选 three.js。

**推荐**：vanilla three.js（ESM，importmap 免打包）+ Rapier 物理。Babylon 是唯一值得考虑的替代；Godot 仅适合最终产品导出；Phaser 只在做 2D 分支时有意义。

## Q2. three.js 游戏栈现状

| 类别 | 库 | 版本 (日期) | Stars / push | 结论 |
|---|---|---|---|---|
| 物理 | **`@dimforge/rapier3d-compat`** | 0.20.0 (2026-08-08) | 5.7k / 2026-09-18 | 首选。WASM，`KinematicCharacterController` 内建 autostep / snap-to-ground / 坡度限制；跨平台确定性用 `-deterministic-compat` |
| 物理 | `jolt-physics` | 1.1.0 (2026-07-11) | 570 | three.js 官方 addon；LLM 熟悉度低 |
| 物理 | `cannon-es` | 0.20.0 (2022-08) | 2.1k / 2024-01 | 实质停更，LLM 训练数据多，只做兜底 |
| 物理 | `ammo.js` | – | 4.6k | 手动内存管理，不推荐 |
| 角色控制 | Rapier KCC / `ecctrl` 2.0.2 | – | 796 | ecctrl 仅 R3F；vanilla 用 Rapier KCC 或 `three-mesh-bvh` shapecast |
| 导航 | **`@recast-navigation/three`** | 0.43.1 | 428 / 2026-07 | 运行时生成 navmesh + crowd，推荐 |
| 导航 | `three-pathfinding` | 1.3.0 (2024-05) | 1.4k | 只做 A*，不生成 navmesh |
| 游戏 AI | `yuka` | 0.7.8 | 1.4k / 2026-09 | steering/FSM/goal-driven，引擎无关 |
| ECS | `bitecs` 0.4.0 / `miniplex` 2.0.0 / `koota` 0.6.6 | – | 1.5k / 1.05k / 734 | miniplex 简单、LLM 易读；Roseblox 采用 |
| 后处理 | `postprocessing` 6.39.5 | – | 2.9k | 与 r186 兼容 |
| React 层 | R3F 9.7.0 / drei 10.7.8 / react-three-rapier 2.2.0 | – | 32.4k / 9.9k / 1.4k | 见下 |
| 相机 | OrbitControls（内建）；`camera-controls` | – | – | 第三人称跟随一般自写 spring-arm + lerp |
| 动画 | `AnimationMixer` + `SkeletonUtils.retargetClip` | 内建 | – | Mixamo FBX→glTF 后播放与重定向 |
| 输入 | 自写 keydown/keyup 状态表 | – | – | 无头测试由 Playwright `keyboard.down/up` 驱动 |

R3F vs vanilla：R3F 的 JSX 像场景 DSL，但引入 React 运行时，无头调试与确定性帧步进更绕。Rosebud 的 Roseblox 也选 vanilla three + Rapier + Miniplex。**推荐 vanilla three.js + Rapier + recast-navigation + miniplex/bitecs + yuka。**

## Q3. 场景格式与现成声明式 DSL

| 格式 / 框架 | 状态 | 能否作为 LLM 目标 |
|---|---|---|
| three.js JSON Object/Scene format 4 | 内建 | 只描述静态层次；几何内联冗长；无行为、相机轨迹、物理 |
| **glTF 2.0 / GLB** | 内建 | 资产交换首选；可用 vendor extension 挂自定义数据（Needle 的 `NEEDLE_components` 做法） |
| USD / USDZ | 不成熟 | 不建议 |
| A-Frame 1.8.0 | 17.6k | HTML 实体-组件语法 LLM 友好，但绑定 DOM/WebXR |
| Threlte 8.6.0 / R3F JSX | – | 需 Svelte / React |
| Needle Engine 5.1.13 | 商业 | glTF 扩展承载整个应用，面向 Unity/Blender 导出 |
| VibeGame (dylanebert) | MIT, 89★, 2025-12 | 最接近"为 LLM 设计的声明式 DSL"，但近一年未更新 |

**推荐**：自研 JSON Schema 形式的 4D 场景程序 DSL，编译器输出 vanilla three.js + Rapier 代码；网格资产用 GLB 承载并以 URI 引用。旁证：Mage 基准（arXiv 2605.07342）发现 NL 直接到代码运行成功率最高但机制 F1≈0.12，经中间表示约束后结构保真度显著提升，支持"LLM 生成 DSL 而非直接生成代码"。

## Q4. Gaussian Splat 渲染（残差层）

| 库 | 版本 | Stars / push | 结论 |
|---|---|---|---|
| **Spark** `@sparkjsdev/spark` | 2.2.0 (2026-09-11) | 3.6k / MIT | three.js 原生对象，流式加载、LOD、多 splat 对象。推荐 |
| GaussianSplats3D (mkkellogg) | 0.4.7 (2025-01) | 2.9k | 近一年停滞 |
| gsplat.js (HF) | 1.2.9 | 1.7k | 独立渲染器，非 three.js 集成 |

## Q5. 无头执行与视觉反馈通道

A. 浏览器路线（推荐）：Playwright 1.63.0。CPU 软渲染 `--headless=new --use-angle=swiftshader`（不要传 `--use-gl=swiftshader-webgl`）。GPU：Chromium 文档说默认 GL 探测需要 X display，无 X 时 `--use-angle=vulkan --enable-features=Vulkan` 在部分配置可用。HPC 坑：无 root 不能 `install-deps`；`--no-sandbox`；`--disable-dev-shm-usage`；`--disable-background-timer-throttling --disable-renderer-backgrounding`。

B. Node 路线：headless-gl 8.1.6 主打 WebGL1，three.js 已要求 WebGL2，高风险；WebGPU/Dawn 路线实验性。不建议主路线。

C. 深度 / 实例 ID pass：深度用 `WebGLRenderTarget` + `DepthTexture` 或 `overrideMaterial = MeshDepthMaterial({depthPacking: RGBADepthPacking})`；实例 mask 渲染前临时把每个 mesh 材质换成 `MeshBasicMaterial({color: idColor})`（GPU picking 例），ID 编码到 RGB 24 位；读回用 `readRenderTargetPixels(Async)`。

D. 确定性帧步进：Playwright `page.clock`（`goto` 前 `install()`，`pauseAt()` + `runFor(16.67)`）；更稳的做法是运行时暴露 `window.__game.step(dt)` / `window.__game.render(pass)` 由 `page.evaluate` 驱动；Rapier 固定 `world.step()`；随机数 `seedrandom`。

## Q6. "引擎内核 + 生成内容"的分层设计与模板

- **Rosebud / Roseblox**（MIT，push 2026-09-17）：three 0.163 + Rapier + Miniplex + camera-controls，buildless；固定 Resource → Setup → Runtime systems 模式，systems 按优先级区间执行（input 10-20 / movement 30-40 / physics 40-45 / animation 50-65 / camera 70-75）；生成内容只是注册新 system/entity。
- **Bitmagic**：自研 TS + three.js 引擎，LLM 从预定义资产库构建；闭源。
- **OpenGame / GameCoder-27B**（arXiv 2604.18394）：Template Skill + Debug Skill 分离；OpenGame-Bench 用 headless browser 执行 + VLM 评分。
- **WorldCoder-Bench**：StateProbe 在沙箱浏览器探测运行时状态；最佳系统仅 27.8% 覆盖，主要失败是"状态 schema 漂移与交互链断裂"，提示要把状态 schema 固化在内核里。

| 类型 | 仓库 | Stars / 最近 push | 栈 |
|---|---|---|---|
| 第三人称 + 载具 | swift502/Sketchbook | 1.75k / 已归档 2024-10 | three + cannon |
| 赛车 | pmndrs/racing-game | 2.2k / 2023-02 | R3F + cannon |
| FPS | mohsenheydari/three-fps | 229 / 2022-05 | ammo + three-pathfinding |
| 塔防 | Casmo/tower-defense | 127 / 2024-08 | three |
| 综合样例库 | isaac-mason/sketches | 343 / 2026-05 | three + Rapier + recast，质量高 |
| AI 起步模板 | heagandev/threejs-agent-starter | 3 / 2026-06 | 面向 agent 的 three.js 模板 |

**推荐**：照 Roseblox 模式自建固定内核（three + Rapier + recast + ECS，buildless ESM），六个 gameplay 模板作为内核内的固定 system 集合，只暴露参数/绑定槽位（`player_spawn`, `goal_volume`, `enemy_paths[]`, `track_spline`, `tower_slots[]`），DSL 的 `binding` 段填槽；生成侧永不改内核代码；`window.__game.state` 做成稳定 schema 供验证。

## Q7. Agent 自动试玩

| 项目 | Stars / push | 要点 |
|---|---|---|
| VideoGameBench (arXiv 2505.18134) | 373 | VLM 实时玩浏览器/DOS 游戏；推理延迟主导失败，提供 Lite 暂停模式 |
| lmgame-Bench / GamingAgent (ICLR 2026) | 976 | Gym 风格统一 API + 感知/记忆 scaffold |
| BALROG | 268 | RL 环境集；视觉输入反而使部分模型变差 |
| GameWorld (arXiv 2604.07429) | 218 | 34 游戏 170 任务，状态可验证指标；对比键鼠 vs 语义动作空间 |
| browser-use | 115k | 通用浏览器 agent，可复用键鼠动作层 |

**推荐**：自建轻量 harness：Playwright 驱动，游戏以暂停-步进模式运行，每步返回 RGB + 深度 + ID mask + `__game.state` JSON；VLM 给动作，状态机验证"可达终点 / 未卡死 / 帧率"等可判定指标。

## 总体推荐栈
vanilla three.js r186 + `@dimforge/rapier3d-compat` 0.20 + `@recast-navigation/three` 0.43 + miniplex/bitecs + yuka + Spark 2.2（残差层）；自研 JSON DSL → 编译到 buildless ESM；Headless Chromium (Playwright 1.63, `--headless=new --use-angle=swiftshader --no-sandbox`) 在 Slurm 计算节点渲染，深度/ID 走 RenderTarget + `readRenderTargetPixelsAsync`，帧步进走 `page.clock` 或显式 `__game.step()`。

未核实项：headless-gl WebGL2 完整度；Needle Engine 授权；aframe-physics-system 维护状态；Bitmagic 引擎内部细节；OpenGame 是否已开源。
