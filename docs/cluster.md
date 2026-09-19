# 在 Vulcan 上运行

> 登录节点与计算节点的差异、网络出口、无头渲染实测、依赖打包与作业约定。实测日期 2026-09-18，脚本与日志在 `../survey/probe/`。

## 1. 登录节点与计算节点

| 项目 | 登录节点 vulcan1 | 计算节点（CPU compute[1-2] / GPU rack*-*） |
|---|---|---|
| 允许的负载 | 编译、npm install、打包、提交作业；禁止渲染与模型推理 | 全部重负载：感知模型、VLM、无头渲染 |
| Node.js | 模块 nodejs/18.17.1、20.16.0、24.15.0 | 同一套 CVMFS 模块：`module load StdEnv/2023 nodejs/20.16.0` |
| Python | 模块 python/3.10.13 ~ 3.14.2 | 同上；虚拟环境建在 `/project` |
| 互联网 | 直连，无代理 | 无直连；squid 代理 `http://squid:3128`（prolog 自动注入 `http_proxy` / `https_proxy`） |
| 代理放行 | — | github.com、huggingface.co、api.anthropic.com、api.openai.com、Docker Hub、inference.vulcan.alliancecan.ca |
| 代理封禁 | — | registry.npmjs.org、Playwright 浏览器 CDN、poly.pizza、sketchfab.com（CONNECT 403） |
| Chromium 系统依赖 | 齐全（libnss3、libatk、libgbm、libxkbcommon、libasound、libEGL、libvulkan 等） | 齐全；NVIDIA EGL 库存在（driver 595.91.07）；Vulkan ICD 目录只有 asahi，但 ANGLE 仍能拿到 L40S |
| 容器 | apptainer 1.3.5 | apptainer 1.3.5，`docker://` 拉取经代理可用 |
| 本地盘 | 无 | `$SLURM_TMPDIR=/tmp`，作业结束即清空 |
| 现有缓存 | `~/.cache/ms-playwright` 已有 chromium-1228（663 MB，占 $HOME 配额，应迁走） | 无 |

## 2. 无头渲染实测

Playwright 1.63.0 + Chromium 1243 + three.js r186，320×240 单立方体，30 帧，颜色 / ID 掩码 / 深度三种 pass 回读均正确：

| 节点 | Chromium 参数 | WebGL2 渲染器 | 每帧 | 结论 |
|---|---|---|---|---|
| CPU（rack01-08） | `--use-gl=angle --use-angle=swiftshader` | SwiftShader（软件 Vulkan） | 0.95 ms | 可用；反馈循环默认走这里 |
| GPU（rack05-05, L40S） | 同上 | SwiftShader | 0.77 ms | 与 CPU 一致 |
| GPU | `--use-gl=angle --use-angle=vulkan --enable-features=Vulkan` | ANGLE (NVIDIA, Vulkan 1.4, NVIDIA L40S) | 1.17 ms | 硬件加速可用，无需 X server |
| GPU | `--use-gl=angle --use-angle=gl-egl` | ANGLE (NVIDIA L40S, OpenGL ES 3.2) | 3.87 ms | 硬件加速可用 |
| GPU | `--use-gl=egl` | 回退 SwiftShader 且 context lost | — | 不要用 |

Chromium 必须带 `--no-sandbox --disable-dev-shm-usage`。小场景下软件渲染已足够快，硬件加速的优势在大分辨率、批量帧时才体现。

## 3. 依赖打包

计算节点装不了 npm 包，也下不了浏览器，所以在登录节点预装后打包：

```bash
module load StdEnv/2023 nodejs/20.16.0
export npm_config_cache=/scratch/$USER/.npm-cache
export PLAYWRIGHT_BROWSERS_PATH=$PWD/pw-browsers
npm init -y && npm install --no-audit --no-fund playwright three @dimforge/rapier3d-compat @recast-navigation/three miniplex
npx playwright install chromium
tar -cf pwstage.tar node_modules pw-browsers package.json      # 实测约 454 MB
```

作业内：

```bash
module --force purge && module load StdEnv/2023 nodejs/20.16.0
tar -xf /project/aip-zhouyang/jwj/GameWorldModel/deps/pwstage.tar -C $SLURM_TMPDIR   # 约 1 秒
export PLAYWRIGHT_BROWSERS_PATH=$SLURM_TMPDIR/pw-browsers
```

Python 环境按集群规范：`module load python/3.11.5`，`virtualenv --no-download`，`pip install --no-index` 优先集群 wheelhouse；缺的包在登录节点用 `pip download` 预取到 `/project`。感知权重预下载到 `/project`，`HF_HOME` 与 `XDG_CACHE_HOME` 指向 `$SCRATCH`。

## 4. 作业模板

| 作业 | 资源参数 | 内容 |
|---|---|---|
| 感知 | `--gres=gpu:l40s:1 --cpus-per-task=16 --mem=64G --time=02:00:00` | ViPE、SAM 3.1、SpatialTrackerV2、证据提取；一次处理一批片段 |
| 生成与反馈 | `--cpus-per-task=8 --mem=32G --time=03:00:00`（CPU 节点；若 DINOv3 要 GPU 则加 1 张卡） | Pydantic AI 循环、编译、Playwright 渲染、指标、试玩回放 |
| 自托管 VLM | `--gres=gpu:l40s:2 --cpus-per-task=32 --mem=128G` | vLLM 起 Qwen3-VL-32B FP8 |
| 素材构建 | `--gres=gpu:l40s:1 --cpus-per-task=16 --mem=64G` | 缩略图渲染、CLIP 嵌入、TRELLIS.2 |

账户 `aip-zhouyang`；不设 `--partition`；`--time` 必设；作业输出写 `$SCRATCH`。CPU 分区只有两个节点，排队时可以把生成与反馈作业挂在 GPU 作业尾段跑。

## 5. 网络相关的工作分配

| 需要访问 | 在哪做 |
|---|---|
| npm registry、Playwright CDN | 登录节点 |
| poly.pizza、Sketchfab、Kenney 官网 | 登录节点 |
| GitHub、HuggingFace（模型权重、Kubric 等数据） | 作业内经代理，或登录节点 |
| Anthropic / OpenAI API | 作业内经代理 |
| Aleph 推理服务 | 作业内 |

## 6. 存储

| 位置 | 放什么 | 现状（2026-09-18） |
|---|---|---|
| `/project/aip-zhouyang/jwj/GameWorldModel/` | 代码依赖包、感知权重、素材库、最终结果 | project 已用 8586 / 10240 GiB |
| `$SCRATCH` | 感知中间产物、渲染帧、作业输出，定期清理 | scratch 已用 4076 / 5120 GiB |
| `$HOME` | 代码与配置 | 18 / 50 GiB；`~/.cache/ms-playwright` 663 MB 应迁走 |

素材库预计小于 3 GB，感知与检索权重约 10~20 GB，都放 project。
