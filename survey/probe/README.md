# Vulcan 无头渲染探测（2026-09-18）

- `probe.sh`：第一轮探测（模块、网络出口、Chromium 依赖库、npm 直装）。结论：计算节点经代理无法访问 npm registry 与 Playwright CDN。
- `probe2.sh` + `probe_render.mjs`：第二轮，依赖在登录节点预装打包后解到 `$SLURM_TMPDIR`，用 Playwright + Chromium 渲染 three.js 立方体并回读 RGB / ID / 深度三个 pass。
- `probe1_cpu_992141.log`、`probe2_cpu_992151.log`、`probe2_gpu_992150.log`：作业日志。
- `render_gpu_angle_vulkan.png`：GPU 节点 ANGLE Vulkan 路径的截图。

结论见 RP.md 第 9.7 节。预装包（node_modules + 浏览器，tar 约 454 MB）未入库，需在登录节点重新生成：
```bash
module load StdEnv/2023 nodejs/20.16.0
export npm_config_cache=/scratch/$USER/.npm-cache PLAYWRIGHT_BROWSERS_PATH=$PWD/pw-browsers
npm init -y && npm install --no-audit --no-fund playwright three && npx playwright install chromium
tar -cf pwstage.tar node_modules pw-browsers package.json
```
