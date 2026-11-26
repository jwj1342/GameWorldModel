# 智能体运行的结果

每个子目录是 `scripts/bench_agent_run.sh` 的一次运行，名字是 `<场景>-<label>`。里面是：
智能体写的 `scene.js` 和 `materials.json`，运行记录 `agent_run.json`（模型、接口地址、轮数、估算花费），
评测的 `results.json`、`fidelity_flat.json`、`geometry.json`、`interact.json`，room 还有 `sequence_lazy.json` 和 `sequence_settled.json`，
以及每一轮快照的 `trajectory.json`。渲染图和贴图不放进来，用同一份 `scene.js` 重跑 `eval` 就能得到。

## 中途快照和 Mac 上的运行记录

- `room-kimi-*/snapshots/NNN/`：智能体每次调用预览工具时留下的 `scene.js` 和 `materials.json`，编号和 `trajectory.json` 里的 `rows` 一一对应（最后一行 `final` 就是目录顶层的 `scene.js`）。
- `lighting-mid-source/`：光照归因里"中途"那一份，是第一次运行在 19:53 的快照；"最终版"就是 `docs/results/bench-room-20261125/` 里的那份。
  `lighting-<几何>-<灯光>/scene.js` 是实际拿去评测的文件，两份源码以 data URL 内嵌。
- `mac-logs/`：Mac 上这一轮的运行日志和排队脚本（`queue.log` 记了 Kimi 额度用完和恢复的时间），
  以及从嵌套会话记录生成各目录 `audit.json` 的 `audit.py`。脚本里写的是当时机器上的绝对路径，换机器要改。
  嵌套会话的原始记录没有放进仓库。
- `room-kimi-*/agent_scratch/`：智能体在工作区里自己写的分析脚本和笔记，目录结构照原样保留（`agent_scratch_first_attempt/` 来自撞上额度上限而中止的那一次）。
  第一次运行（Fable）的在 `docs/results/bench-room-20261125/agent_scratch/`。脚本产出的图片没有放进来。
