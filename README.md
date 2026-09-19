# GameWorldModel

从一段无标定的单目视频推断带时间轴的可执行场景程序，编译到 three.js，绑定玩法模板后成为可玩的浏览器游戏。

## 文档

| 文档 | 读什么 |
|---|---|
| `RP.md` | 研究提案：动机、问题定义、方法、数据引擎、评估协议、应用、贡献、风险，以及第 9 节的实施路线图 |
| `docs/baseline.md` | 第一阶段计划：目标与验收、六个阶段的任务清单、备选方案 |
| `docs/architecture.md` | 系统架构：模块、数据流、产物目录、接口契约 |
| `docs/dsl.md` | 场景程序 DSL：文本与 JSON 形式、词表、校验、编译映射 |
| `docs/runtime.md` | 运行时与游玩层：three.js 栈、内核、玩法模板、无头渲染、试玩验收 |
| `docs/perception.md` | 感知与证据提取 |
| `docs/models.md` | 模型层：API 与开源 VLM 的视觉支持、部署、成本；训练栈与数据集 |
| `docs/feedback.md` | 视觉反馈与 agent 编排 |
| `docs/assets.md` | 素材层 |
| `docs/related-work.md` | 相关工作 |
| `docs/cluster.md` | 在 Vulcan 上运行 |
| `survey/` | 原始调研报告（2026-09-18）与集群探测记录 |

## 约定

- 阶段与产物用它们交付的东西命名，不编版本号。
- 生成侧只产出场景程序（数据），运行时内核是固定代码。
- 校验先于渲染，反馈必须结构化。
- 集群上登录节点只做编译与打包，一切渲染与模型推理进 Slurm 作业。
