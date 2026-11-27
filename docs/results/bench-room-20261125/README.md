# 两轴评测的第一次运行：room（2026-11-25）

这是 `docs/bench-demo.md` 里那张结果表「最终版」两行的原始产物，从
`out/bench/room/runs/20261125-200727-agent/` 拷出来的。

| 文件 | 内容 |
|---|---|
| `scene.js`、`materials.json` | 智能体写的场景：31 个物体、67 种材质 |
| `agent_run.json` | 这次是谁写的、怎么写的（会话内子智能体，不是 `claude -p`） |
| `results.json` | 两个点的汇总：纯色和生成贴图各自的保真度，加上共用的交互分 |
| `fidelity_flat.json`、`fidelity_textured.json` | 39 个留出视角逐个的 PSNR、SSIM、LPIPS |
| `interact.json` | 交互清单 30 条逐条的执行结果 |
| `sequence_lazy.json`、`sequence_settled.json` | 连续交互 24 条检查的结果，两种条件（见 `docs/bench-demo.md`）。这两份是后来在 Vulcan 上补跑的，不是 `eval` 当时的产物 |
| `package.json` | 只有 `{"type":"module"}`，让 Node 22 以前的版本也能把这里的 `scene.js` 当 ES 模块加载 |
| `run.json` | 运行时的配置快照和 git 提交 |
| `renders/` | 留出视角 `DSCF4699` 的两张渲染图（纯色、生成贴图） |

没有放进来的：留出视角的真实照片（数据集的图不进仓库，按 `docs/bench-demo.md` 下载）、
生成的贴图和其余 38 个视角的渲染（重跑 `eval` 即可得到，贴图在同一台机器上逐字节可复现）。

重新评测这份场景：

```bash
venv/bin/python -m gwm.bench.run eval --config configs/bench/room.yaml --cand docs/results/bench-room-20261125 --label rerun
```
