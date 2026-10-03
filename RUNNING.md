# 怎么跑起来

这份文档讲怎么在自己的机器上或者在 HPC 集群上把这个项目跑起来。想先知道它是干什么的，看 README.md。

## 需要准备什么

在 Linux 和 macOS 上验证过。Windows 要用 WSL，因为渲染那部分是通过一个 bash 脚本调起来的。

不管在哪跑，下面这些是必须的。

Python 3.10 以上。代码里用了 `X | None` 这种类型写法，3.9 跑不了。我们在 3.12 上开发和测试。

Node.js 20 以上。three.js 内核和无头浏览器那部分是 Node 跑的。

ffmpeg 和 ffprobe。抽帧、合成检查视频都要用，用系统包管理器装就行。

一个能调视觉模型的 API key。默认走 OpenRouter，注册后拿到 key 放到 `.secrets/openrouter.key` 这个文件里。也可以不给，加 `--no-vlm` 参数让程序直接从感知结果翻译出场景，整条链一样能跑完，只是没有模型那一步的修正。

磁盘大约 12 GB。感知模型权重 9.6 GB，Node 依赖和 Chromium 加起来 1.1 GB，剩下是产物。

内存 8 GB 起步，16 GB 比较从容。实测一条 20 秒的视频跑完整管线峰值用了 8.5 GB。

CPU 多核会快很多。一条视频的感知部分大约要 35 分钟的单核计算量，16 核上 4 分钟跑完，8 核大概 6 分钟，4 核大概 10 分钟。

显卡不是必须的。有 8 GB 以上显存的 NVIDIA 卡会快不少，PyTorch 会自动用上，不用改配置。

## 在自己的电脑或普通服务器上跑

### 第一次装

```bash
git clone https://github.com/jwj1342/GameWorldModel.git
cd GameWorldModel

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/facebookresearch/vggt"

npm install                       # 会顺带下载 Chromium，大约 600 MB

python3 scripts/download_weights.py          # 下载感知模型权重，约 9.6 GB
```

VGGT 那一行必须带 `--no-deps`，它的安装脚本里钉死了 numpy 1.x 和很老的 torch，不加这个参数会把环境搞坏。

想先看看要下哪些权重，跑 `python3 scripts/download_weights.py --list`。VGGT 的 5 GB 权重走 HuggingFace 的 CDN，有的网络环境会被挡，脚本会打印出用 curl 直接下的命令。

要用模型的话，把 OpenRouter 的 key 存进去：

```bash
mkdir -p .secrets && echo "sk-or-v1-你的key" > .secrets/openrouter.key
```

### 跑一条视频

先把视频放到 `data/clips/trimmed/` 下面，命名成 `片段名.mp4`。长度十几秒到一分钟，720p 就够，太长太大只会更慢。然后

```bash
source venv/bin/activate
python3 -m gwm.run_clip --video data/clips/trimmed/我的视频.mp4 --clip 我的视频
```

第一次跑建议加上 `--phrases`，把画面里有什么用英文名词短语告诉检测器，逗号分隔，效果会好很多：

```bash
python3 -m gwm.run_clip --video data/clips/trimmed/我的视频.mp4 --clip 我的视频 \
  --phrases "toy train,train track,floor"
```

不给 `--phrases` 的话，程序会让模型看第一帧自己列，没有模型时用一张默认词表。

不想调模型就加 `--no-vlm`。想换模型加 `--config configs/vlm/openrouter_qwen122b.yaml`，`configs/vlm/` 下面还有 Kimi、Opus 和自托管 vLLM 的配置。

跑完之后产物在 `out/片段名/运行编号/`，进去执行

```bash
node harness/serve.mjs game 8000
```

然后浏览器打开 http://localhost:8000/ 就能玩。WASD 或方向键移动，空格跳，R 在回放和游玩之间切换。

### 先不下权重，只看看游戏部分

如果只想看 three.js 那一半是怎么回事，不装感知模型也行。装好 Python 依赖里 jsonschema 那一组和 npm 依赖之后：

```bash
python3 -m gwm.compiler.compile examples/handwritten/program.json /tmp/mygame
node harness/serve.mjs /tmp/mygame 8000
```

这会把仓库里手写的示例场景程序编译成游戏。改 `examples/handwritten/program.json` 里的数字再编译一次，游戏就变了，这是理解场景程序长什么样最快的办法。

### 常见问题

浏览器打开是空白，控制台报 ES 模块加载失败，说明是直接双击的 `index.html`。浏览器不允许本地文件加载 ES 模块，必须用上面的命令起一个服务器。

提示找不到 `node_modules`，在仓库根目录执行 `npm install`。

感知那一步报显存不足，把 `configs/perception/cpu.yaml` 加上：`--config configs/perception/cpu.yaml`，它把几何用的帧数降到 24、关键帧降到 8。或者设 `CUDA_VISIBLE_DEVICES=` 强制走 CPU。

## 在 HPC 集群上跑

下面以我们自己用的 Vulcan 为例。别的 Slurm 集群思路一样，改改账号和模块名就行。

集群和个人电脑最大的不同有两点。一是计算节点通常没有外网直连，装不了 npm 包也下不了 Chromium，所以这两样要在登录节点准备好、打成一个包，作业里再解开用。二是所有重活都必须通过作业提交，登录节点只能做安装、打包和提交。

### 第一次装

在登录节点上：

```bash
bash scripts/stage_node_deps.sh     # 装 Node 依赖和 Chromium，打包放到 project 空间
sbatch scripts/env_setup.sh         # 建 Python 环境，下载权重
```

`scripts/setup_env.sh` 里写了要加载哪些模块和各个目录在哪，换集群的话改这一个文件。加 `--with-vllm` 参数给 `env_setup.sh` 会再建一个自托管 vLLM 用的环境，不用自托管模型就不需要。

OpenRouter 的 key 同样放 `.secrets/openrouter.key`。计算节点经代理能访问 OpenRouter。

### 跑一条视频

```bash
sbatch scripts/pipeline.sh 片段名 "toy train,train track,floor" --config configs/perception/cpu.yaml
```

默认申请 16 核 64 GB 跑两小时，在 CPU 节点上就能跑完，不用排 GPU 的队。要用 GPU 就加 `--gres=gpu:l40s:1`。

其它常用的作业：`scripts/smoke_kernel.sh` 用手写的示例程序验证内核和无头渲染；`scripts/run_tests.sh` 跑单元测试；`scripts/make_release.sh` 把跑出来的结果打包；`scripts/serve_vlm.sh` 起自托管的 vLLM 服务。

产物在 `out/片段名/运行编号/`。想在浏览器里玩的话，要么把 `game` 目录下载到本地起服务器，要么用集群的 Open OnDemand。

### 登录节点和计算节点的区别（实测）

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

### 无头渲染实测

Playwright 1.63.0 + Chromium 1243 + three.js r186，320×240 单立方体，30 帧，颜色 / ID 掩码 / 深度三种 pass 回读均正确：

| 节点 | Chromium 参数 | WebGL2 渲染器 | 每帧 | 结论 |
|---|---|---|---|---|
| CPU（rack01-08） | `--use-gl=angle --use-angle=swiftshader` | SwiftShader（软件 Vulkan） | 0.95 ms | 可用；反馈循环默认走这里 |
| GPU（rack05-05, L40S） | 同上 | SwiftShader | 0.77 ms | 与 CPU 一致 |
| GPU | `--use-gl=angle --use-angle=vulkan --enable-features=Vulkan` | ANGLE (NVIDIA, Vulkan 1.4, NVIDIA L40S) | 1.17 ms | 硬件加速可用，无需 X server |
| GPU | `--use-gl=angle --use-angle=gl-egl` | ANGLE (NVIDIA L40S, OpenGL ES 3.2) | 3.87 ms | 硬件加速可用 |
| GPU | `--use-gl=egl` | 回退 SwiftShader 且 context lost | — | 不要用 |

Chromium 必须带 `--no-sandbox --disable-dev-shm-usage`。小场景下软件渲染已足够快，硬件加速的优势在大分辨率、批量帧时才体现。

### 换一个集群要改什么

`scripts/setup_env.sh` 里的模块名和目录。`configs/vulcan.yaml` 里的路径和 Slurm 账号，或者照着它新建一个自己的站点配置。各个 `scripts/*.sh` 顶上的 `#SBATCH` 参数。

程序怎么知道自己在哪个站点：看环境变量 `GWM_SITE`，没设的话检测到 Slurm 或 CVMFS 就当成 Vulcan，否则当成本地。想强制用本地配置就 `export GWM_SITE=local`。

## 一次运行产出什么

```
out/片段名/运行编号/
  report.md              汇总报告，先看这个
  program.json           场景程序，游戏的全部内容
  run.json               这次运行的完整配置和每一步耗时
  errors.jsonl           出过什么错、程序怎么处理的
  game/                  可玩的游戏，起服务器打开
  perception/
    evidence.json        感知结果
    overlay.mp4          把感知结果画回视频上，检查认得对不对
    keyframes/           送给模型看的那几张图
  generation/
    writer_log/          模型每一步的原始输出
    rounds/r0 r1 r2      每一轮的程序、渲染图、比对结果
  feedback/              最终一轮的 RGB 深度 物体ID 三个通道的渲染
  playtest/              自动试玩的截图、状态日志和判定
  vlm_calls.jsonl        每次调模型的输入输出和花了多少钱
```

`scripts/collect_results.sh out/片段名/运行编号` 可以把其中值得看的挑出来放到 `docs/results/`。
