# 原始调研报告（2026-09-18）

五份报告由调研 agent 生成，保留原貌作为一手材料；每份都标注了未核实项。整理后的内容最初按层写进 `../docs/` 下的多份文档，2026-09-20 合并进 `docs/architecture.md` 和 `docs/design-notes.md`；相关工作在 `../related-work/`：

| 报告 | 主题 | 整理到 |
|---|---|---|
| `01_generation_from_text_image_video.md` | 一句话 / 一张图 / 一段视频生成游戏或可探索世界的论文与产品 | `related-work/README.md` |
| `02_runtime_threejs_gamelayer.md` | 运行时选型、three.js 游戏栈、场景格式、无头渲染、内核与模板、试玩 agent | `docs/architecture.md`、`docs/design-notes.md` |
| `03_assets_and_3d_generation.md` | CC0 资产库、研究数据集、检索配方、image-to-3D、关节生成、许可 | `docs/architecture.md`、`docs/design-notes.md` |
| `04_agent_scaffold_visual_feedback.md` | agent 脚手架、多智能体有效性、视觉反馈机制、指标、试玩 agent | `docs/architecture.md`、`docs/design-notes.md` |
| `05_perception_and_model_layer.md` | 动态视频位姿与深度、分割与跟踪、4D 重建、VLM 对比、训练栈、数据集 | `docs/architecture.md`、`docs/design-notes.md` |
| `probe/` | Vulcan 登录节点与计算节点的 Node.js、网络、Playwright、Chromium 探测脚本与日志 | `docs/architecture.md`、`docs/design-notes.md` |
