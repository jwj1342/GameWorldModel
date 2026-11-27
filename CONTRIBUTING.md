# 怎么参与

main 分支开了保护，不能直接往上推，改动都通过 Pull Request 进来，需要一个人批准之后才能合并。

## 流程

```bash
git checkout -b 你的分支名        # 起个能看出在做什么的名字，比如 fix-motion-classification
# 改代码
python -m pytest -q tests        # 开 PR 之前先跑一遍，应该全过
git commit -am "一句话说清楚改了什么"
git push -u origin 你的分支名
gh pr create                     # 或者去网页上开
```

PR 里写清楚改了什么、为什么改。动了感知、生成或者评测的，附上改动前后跑同一份输入的对比。

评论都解决掉、有一个批准之后就能合并。一条线做完就合进 main，别让分支长期挂着：main 应该随时反映项目现在的样子。

## 提交之前注意

不要提交 API key、视频、运行产物、会议转写。哪些目录不进 git 见下面「新东西放哪」。

依赖有变动的话，`requirements.txt` 是给所有人用的通用列表，`requirements-vulcan.lock.txt` 是集群上的精确版本，两个都要更新。

## 工程上守的几条

研究代码也按工程代码的标准写。下面每条都对应这个仓库里的具体做法。

1. **不重复**：同一段逻辑只写一处。两个脚本里出现同样的读取、计分或渲染代码，就把它挪进 `gwm/` 对应的模块，脚本里只留调用。路径和阈值也一样，只在 `configs/` 和环境脚本里出现一次。
2. **关注点分离、单一职责**：一个包只管一件事（感知、合成、编译、反馈、评测各是一个包），一个文件只有一个改动的理由。实验脚本只负责「跑一次、写结果」，不在里面长出新算法；算法进 `gwm/`。
3. **小而稳的接口**：模块之间只通过下面「模块之间的边界」那张表里的契约交互，实现可以随便换。新加一条边界，就把它写进那张表。
4. **低耦合、高内聚**：一个包不去读另一个包的内部文件或私有格式，只用对方公开的函数或文件契约。比如评测（`gwm/bench`）不依赖几何管线的内部。
5. **无状态、能横向扩**：每个阶段都是「输入文件 → 输出文件」的步骤，幂等、能重跑、能按片段或场景拆成作业数组并行。不靠进程里的全局状态，也不靠手工步骤。
6. **可观测、可测试**：每次运行写 `run.json`（git commit、配置快照、模型与权重、随机种子、时间、节点）和日志。新模块配单元测试，跨模块的链路配一条能在 CPU 作业里跑的冒烟脚本。测试不依赖集群路径。
7. **保持简单，不为假想的需求设计**：先写最直接的线性版本。没有第二个实现之前不抽象，没人用的参数和开关不加。不写吞掉错误的 try/except，也不为不会发生的情况加兜底：错在哪就在哪报出来。

**研究上**

1. 实验可复现：每次运行写 `run.json`，产物按运行分目录，永不覆盖。
2. 配置驱动：选型、阈值、轮数、候选数、消融开关都在 `configs/*.yaml`，代码里没有魔法数。
3. 提示词文件化：`prompts/` 下每个角色一个文件，few-shot 单独存放，改动可 diff。
4. 先合成再真实：先用内核自渲染的合成数据（有真值）对齐，再上真实数据。
5. 先量后改：动之前先有能量出差别的量具，量具本身要先自检（预言机该满分、空对空不能算满分），看结果先看样本量和分布，别只看一个均值。
6. 每个实验做完都在 `experiments/index.csv` 登记一行，没成立的也要登记，写清是怎么排除的。
7. 数据许可随数据走：`data/clips/raw/SOURCES.md` 记录来源、作者、许可原文、直链、sha1。

**LLM 驱动的图形程序合成**

1. 固定内核、生成数据：模型只产出程序，运行时是固定代码，状态 schema 固化。
2. 校验先于渲染：schema 与语义校验失败的程序直接回喂，不进渲染。
3. 结构化反馈：数值指标先出逐物体 pass/fail 子句，VLM 只看失败区域，输出受 schema 约束的建议。
4. 有界循环：轮数、候选数、调用次数与 token 都有上限；不优则回退。

**集群**：npm 与浏览器在登录节点装好；Python venv 在作业里建或放 `/project`；权重预下载到 `/project`；中间产物在 `work/`（scratch 上）；任何超过几分钟的计算都进作业，登录节点上不跑 Python。

## 代码放在哪

目录只写到包这一级，文件清单不写进文档，写了就会过期。

```
GameWorldModel/
  README.md  RP.md  RUNNING.md  CONTRIBUTING.md  AGENTS.md
  gwm/            Python 代码，一个子包一件事
    perception/     视频 → 几何、掩码、轨迹（几何管线，已冻结作对照）
    synthesis/      证据 → 场景程序（VLM 写、评、改）
    compiler/       程序校验、素材解析、打包
    feedback/       渲染比对和指标
    binding/ playtest/   玩法绑定、自动试玩
    assets/         素材检索
    dataset/        合成数据集：样式与行为分两路采样
    bench/          两轴评测：场景包、智能体运行、保真度、几何、交互计分、报告
  kernel/         固定运行时（buildless ESM，three.js + 物理）
  harness/        Node + Playwright：渲染、试玩、录制、评测里的交互执行
  configs/        阈值与选型；local.yaml / vulcan.yaml 是站点路径；bench/ 是评测场景
  prompts/        每个角色一个提示词文件
  scripts/        入口脚本和冒烟脚本；一次性的探针放 scripts/probes/（已有的 probe_*.py 是之前留下的，不再挪）
  tests/          单元测试
  experiments/    实验索引 index.csv，以及需要单独说明的实验目录
  related-work/   相关工作：定位（README）、逐篇索引（index.csv）、PDF（本地）
  docs/           设计文档、状态（status.md）、要留的结果（results/）
  survey/         早期调研报告原文
  examples/       手写的示例程序
  data/           数据（大文件不入 git，来源和许可入 git）
```

## 新东西放哪

| 东西 | 放哪 | 进 git 吗 |
|---|---|---|
| 新功能、新算法 | `gwm/<包>/`，配 `tests/` 里的测试 | 进 |
| 一次性的探针脚本 | `scripts/probes/`，在 `experiments/index.csv` 里登记 | 进 |
| Slurm 作业脚本 | `jobs/`；每次运行把它复制一份进运行目录 | 不进 |
| 运行产物 | `work/<路线>/<YYYYMMDD>-<名字>/`，里面有 `run.json` 和作业脚本的副本 | 不进 |
| 要留下来给人看的结果 | `docs/results/<名字>/`，带 `run.json` | 进 |
| 实验结论 | `experiments/index.csv` 加一行（列的意思见 `experiments/README.md`） | 进 |
| 相关工作 | `related-work/index.csv` 加一行，PDF 放 `related-work/pdf/编号-简称.pdf` | CSV 进，PDF 不进 |
| 数据集 | `data/<数据集>/`，在 `data/` 里写明来源和怎么下载 | 不进 |
| 环境和权重 | `venv`、`models`、`deps` 是指向 `/project` 的软链接 | 不进 |
| 论文和汇报 | 各自是独立的 Overleaf 仓库，克隆在 `paper/`、`reports/` | 不进本仓库 |
| 会议转写、个人笔记 | 本地的 `notes/`，不放进本仓库，也不放进 Overleaf 仓库 | 不进 |
| API key | `.secrets/` | 不进 |

命名按内容起，不用 V0、M1、v2 这类版本号。`work/` 和 scratch 上的东西闲置会被清理，要留的结果一定挪进 `docs/results/`。

## 模块之间的边界

| 边界 | 契约 | 可替换实现 |
|---|---|---|
| 感知几何 | `GeometryBackend.estimate(frames, indices, cfg) -> Geometry`（内参、cam→world、深度、可选动态掩码、scale 标记） | VGGT / 静态相机 + 单目深度 / 真值（合成片段） |
| 感知分割 | `SegmentationBackend.segment(frames, indices, phrases, cfg) -> Tracks`（逐帧掩码 + 跨帧 ID） | Grounding DINO + SAM 2.1 / 真值（合成片段） |
| 证据 | `evidence.json`（字段见 `architecture.md` 5.1） | 只依赖上面两个协议 |
| 程序合成 | `VLMClient.chat(system, text, images, json_schema, temperature)` | OpenRouter（默认）/ 自托管 vLLM / 假模型 mock |
| 程序 | `program.json` + JSON Schema + 语义校验 | Writer / Critic / 直译（`synthesis/direct.py`） |
| 编译 | `validate → resolve_assets → bundle` 三个纯函数 | `AssetResolver` 策略链 |
| 运行时 | 内核解释 `program.json`；对外只有 `window.__game` | 模板（`templates/*.js`）与运动节点注册表可扩展 |
| 反馈 | harness 命令行契约：`render.mjs --game DIR --times ... --out DIR` | Python 不关心浏览器细节 |
| 评测场景包 | `gwm.bench.run pack` 写出的场景包（照片、相机、留出视角、观测点）；智能体只该看到 `pack/agent/`（隔离还没做，见 `docs/bench-demo.md`） | 任意能写 `scene.js` + `materials.json` 的智能体 |
| 评测计分 | 候选目录里的 `scene.js` + `materials.json` → `results.json`（保真度、几何、交互清单、连续序列） | — |
| 编排 | 每阶段幂等，产物存在即跳过，`--resume` 复用 | — |

## 环境怎么搭

看 [RUNNING.md](RUNNING.md)，里面分了在自己电脑上跑和在 HPC 上跑两种情况。两轴评测在 Vulcan 上的环境见 `scripts/setup_bench_env.sh` 和 `docs/bench-demo.md`。

只改场景程序、编译器或者 three.js 内核的话，不用下那 9.6 GB 的感知权重，装好 Python 依赖和 `npm install` 就能用 `examples/handwritten/program.json` 调试。
