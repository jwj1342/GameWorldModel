# 工程规范：项目组织、抽象边界、错误处理与 agent 选择

> 与 `architecture.md`（模块与接口）和 `baseline.md`（阶段计划）配套。本文回答"怎么组织、怎么隔离、出错怎么办、哪些地方用模型"。

## 1. 原则

**科研项目**

1. 一切实验可复现：每次运行写 `run.json`（git commit、配置快照、模型与权重、随机种子、时间、节点），产物按 `out/<clip>/<run_id>/` 分目录，永不覆盖。
2. 配置驱动：所有选型、阈值、轮数、候选数、消融开关都在 `configs/*.yaml`；代码里没有魔法数。
3. 评测钩子从第一天就在：指标脚本与消融开关（`no_evidence`、`no_feedback`、`single_stage`）和主流程一起写，后面做 benchmark 不用重构。
4. 提示词文件化：`prompts/` 下每个角色一个文件，few-shot 单独存放，改动可 diff。
5. 先合成再真实：先用内核自渲染的合成片段（有真值）对齐管线，再上真实视频。
6. 依赖锁定：`requirements.txt`、`requirements-vllm.txt`、`package-lock.json`，`/project` 下的 tar 包与权重目录记录 sha。
7. 数据许可随数据走：`data/clips/raw/SOURCES.md` 记录来源、作者、许可原文、直链、sha1。

**LLM 驱动的图形程序合成**

1. 固定内核、生成数据：模型只产出 `program.json`，运行时是固定代码，状态 schema 固化。
2. 校验先于渲染：schema 与语义校验失败的程序直接回喂，不进渲染。
3. 结构化反馈：数值指标先出逐物体 pass/fail 子句，VLM 只看失败区域，输出受 schema 约束的建议。
4. 有界循环：轮数、候选数、调用次数与 token 都有上限；不优则回退。
5. 每个阶段一个 smoke 脚本，能在 CPU 作业里独立跑。

**集群**：npm 与浏览器在登录节点装好打包；Python venv 在作业里建；权重预下载到 `/project`；中间产物在 `$SCRATCH`；GPU 只给感知、VLM 与批量渲染；任何超过几分钟的计算都进作业。

## 2. 项目结构

```
GameWorldModel/
  RP.md  README.md  RUNNING.md  docs/  survey/
  configs/     default.yaml（阈值与选型） local.yaml / vulcan.yaml（站点路径，按 GWM_SITE 或自动检测选）
               perception/{cpu,gt}.yaml vlm/{openrouter_*,vllm_local,mock}.yaml ablations/*.yaml
  prompts/     common_dsl.md writer_*.md critic.md namer.md playtest_review.md fewshot/
  gwm/         Python 管线
    config.py errors.py run_clip.py
    perception/  frames.py base.py masks.py vggt_backend.py static_backend.py grounded_sam2_backend.py gt_backend.py evidence.py run.py
    synthesis/   vlm.py mock_vlm.py direct.py writer.py critic.py loop.py
    compiler/    schema/program.schema.json validate.py assets.py bundle.py compile.py ids.py
    feedback/    render.py metrics.py
    binding/     platformer.py
    playtest/    autopilot.py
  kernel/      固定运行时（buildless ESM）：main.js scene.js motions.js physics.js passes.js replay_camera.js templates/platformer_3p.js index.html
  harness/     Node + Playwright：common.mjs render.mjs playtest.mjs record.mjs serve.mjs
  examples/handwritten/program.json
  scripts/     setup_env.sh env_setup.sh stage_node_deps.sh download_weights.py pipeline.sh smoke_kernel.sh
               run_tests.sh serve_vlm.sh trim_clips.sh node_harness.sh collect_results.sh make_release.sh release_files/
  tests/       单元测试（CPU 作业跑）
  data/clips/{raw,trimmed}   视频（不入 git；SOURCES.md 入 git）
  out/<clip>/<run_id>/        产物（不入 git）
  .secrets/    API 密钥（不入 git）
  requirements.txt            通用依赖；requirements-vulcan.lock.txt 是集群上的精确版本
  models/ deps/ venv/         软链接到 /project/aip-zhouyang/jwj/GameWorldModel/{weights,deps,venv}
```

## 3. 抽象边界

| 边界 | 契约 | 可替换实现 |
|---|---|---|
| 感知几何 | `GeometryBackend.estimate(frames, indices, cfg) -> Geometry`（内参、cam→world、深度、可选动态掩码、scale 标记） | VGGT / 静态相机 + 单目深度 / 真值（合成片段） |
| 感知分割 | `SegmentationBackend.segment(frames, indices, phrases, cfg) -> Tracks`（逐帧掩码 + 跨帧 ID） | Grounding DINO + SAM 2.1 / 真值（合成片段） |
| 证据 | `evidence.json`（字段见 `architecture.md` 5.1） | 只依赖上面两个协议 |
| 程序合成 | `VLMClient.chat(system, text, images, json_schema, temperature)` | OpenRouter（默认）/ 自托管 vLLM / 假模型 mock |
| 程序 | `program.json` + JSON Schema + 语义校验 | Writer / Critic / 直译（`synthesis/direct.py`） |
| 编译 | `validate → resolve_assets → bundle` 三个纯函数 | `AssetResolver` 策略链（MVP 只有 primitive） |
| 运行时 | 内核解释 `program.json`；对外只有 `window.__game` | 模板（`templates/*.js`）与运动节点注册表可扩展 |
| 反馈 | harness 命令行契约：`render.mjs --game DIR --times ... --out DIR` | Python 不关心浏览器细节 |
| 编排 | 每阶段幂等，产物存在即跳过，`--resume` 复用 | — |

## 4. 各部分怎么做

- **感知**：抽帧 4 fps → 工作帧集（≤ 64 帧，VGGT 输入 518 宽、高为 14 的倍数，避免裁剪导致掩码错位）→ 关键帧按运动弧长抽样 → 名词短语（VLM Namer 或默认表）→ 几何后端链 → 分割后端链 → 掩码 × 深度反投影 → 逐帧 OBB → 平滑 → 螺旋拟合分类 → 接触 → 地面对齐（RANSAC 平面，y 向上，地面 y=0，按假定相机高度定标）→ `evidence.json` + 叠加视频。
- **规划（程序生成）**：三阶段（相机与静态 → 物体 → 运动与事件），每阶段 JSON Schema 子集约束解码；校验错误直接回喂最多 3 次；仍失败则该阶段用证据直译；Writer 可否决证据但要在 `notes` 写理由。
- **执行**：编译 = 校验 + 素材解析 + 打包；内核解释程序；Rapier 提供静态碰撞体、运动学刚体、KCC 与平台携带；`seek(t)` 在回放模式下是 t 的纯函数（不跑 controller 与 events）；三个 pass 直接渲到主画布用 `toDataURL` 编码。
- **反馈**：数值层（逐物体 IoU、质心、轨迹 ATE、DINOv2）→ 子句 → 只对 FAIL 子句做裁剪对与叠加图 → Critic 输出 JSON Patch → 候选（全部补丁 / 最差物体的补丁）→ 编译渲染打分 → 不优则回退；默认两轮，上限三轮。
- **绑定与试玩**：可行走面取最大静态顶面；出生点靠近首帧相机；目标取最远角；类别决定收集物与危险物；无地面则补地面；自动驾驭朝目标行走、受阻跳跃与侧步；规则判定到达 / 卡死 / 帧率；可选 VLM 一句话评审。

## 5. 错误处理

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

## 6. agent 的选择

| 角色 | 用什么 | 为什么 |
|---|---|---|
| Namer（首帧名词短语） | Qwen3.5-27B，低温度，JSON 约束 | 轻任务，同一端点 |
| Writer（三阶段生成） | Qwen3.5-27B，结构化输出，关闭 thinking | 用户指定；thinking 与 JSON 约束冲突 |
| Critic（子句 → 建议） | 同 Writer | SEIG 用一个模型兼任 generator 与 verifier；避免两套提示词漂移 |
| 试玩评审 | 同 Writer，仅一句话 | 规则判定为主 |
| 感知聚合、编译、绑定、指标 | 纯代码 | 有确定算法就不用模型 |

不做角色扮演式多智能体（依据见 `feedback.md` 第 7 节）。MVP 用手写 evaluator-optimizer 循环 + pydantic / jsonschema 校验；`VLMClient` 协议保证之后可接 Claude 做对比或换 Pydantic AI。消融开关：`no_evidence`（Writer 只看帧）、`no_feedback`（一轮）、`single_stage`（一次出全程序）。
