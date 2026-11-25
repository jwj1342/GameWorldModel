# 智能体运行的结果

每个子目录是 `scripts/bench_agent_run.sh` 的一次运行，名字是 `<场景>-<label>`。里面是：
智能体写的 `scene.js` 和 `materials.json`，运行记录 `agent_run.json`（模型、接口地址、轮数、估算花费），
评测的 `results.json`、`fidelity_flat.json`、`geometry.json`、`interact.json`，room 还有 `sequence_lazy.json` 和 `sequence_settled.json`，
以及每一轮快照的 `trajectory.json`。渲染图和贴图不放进来，用同一份 `scene.js` 重跑 `eval` 就能得到。
