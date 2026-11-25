# 两轴评测 demo

这是一个最小的可验证版本：给一个真实房间的多张带位姿照片，让无头的编码智能体把它写成一份 three.js 场景代码，然后**分开**量两件事。

- **像不像。** 在智能体没见过的留出视角下渲染场景，和那些位置拍的真实照片比（PSNR、SSIM、LPIPS）。
- **能不能动。** 一份对着真实房间写的交互清单，在物理引擎里逐条执行，报通过率。

两个数不合成一个。整件事的论点是一张二维图：高斯重建在「很像、完全不能动」的角上，纯色基本体在「能动、不太像」的角上，要看的是谁能往右上走。

第一个场景是 Mip-NeRF 360 的 `room`。

## 流程

```
带位姿的照片 ──► [pack] 坐标归一化、训练/留出划分 ──► 场景包
                                                      │
                 agent/（训练视角照片、相机、稀疏点、物体名单）
                                                      ▼
                         [agent] claude -p 写 scene.js + materials.json
                                                      │
                                                      ▼
[eval]  同一份代码渲染两遍：flat（纯色）、textured（本机扩散模型按文字生成贴图）
        ├─ 留出视角渲染 vs 真实照片 ──► 保真度
        └─ 交互清单逐条执行        ──► 通过率
                                                      │
                                                      ▼
                                    [report] 静态对比网页
```

## 怎么跑

第一次要装依赖和下数据：

```bash
npm install
uv venv venv --python 3.12 && uv pip install --python venv/bin/python -r requirements-bench.txt
mkdir -p data/mipnerf360 && cd data/mipnerf360
curl -L -O http://storage.googleapis.com/gresearch/refraw360/360_v2.zip     # 11.7 GB
unzip -q 360_v2.zip 'room/*'
```

然后：

```bash
venv/bin/python -m gwm.bench.run pack --config configs/bench/room.yaml

W=out/bench/room/workspaces/$(date +%Y%m%d-%H%M%S)
venv/bin/python -m gwm.bench.agent prepare --config configs/bench/room.yaml --workspace $W
venv/bin/python -m gwm.bench.agent run     --config configs/bench/room.yaml --workspace $W   # 需要 claude 命令行已登录

venv/bin/python -m gwm.bench.run eval   --config configs/bench/room.yaml --cand $W --label agent
venv/bin/python -m gwm.bench.run report --config configs/bench/room.yaml
```

网页在 `out/bench/room/report/index.html`。每次 `eval` 的产物在 `out/bench/room/runs/<时间>-<label>/`，不覆盖。

不想等智能体、只想确认管线没坏：`examples/bench_smoke/` 是一个手写的小场景，`--cand examples/bench_smoke --label smoke` 就能走完评测。它和 room 毫无关系，分数只是个下限。

## 智能体交什么

两个文件，约定写在 `prompts/bench_scene_writer.md` 里：

- `scene.js` 导出 `describe(THREE, ctx)`，返回物体数组。每个物体有 `id`、`kind`（`static` 或 `movable`）、一个由基本体网格拼成的 `object3D`，可选 `mass_kg` 和 `interaction`（门的铰链、抽屉的滑轨）。
- `materials.json` 每种表面一条：一段英文描述、一个种子、一个底色。

样式和行为在这里是分开的两半。外观只能通过 `ctx.material(名字)` 拿，纯色渲染用底色，贴图渲染用生成的贴图，场景代码一个字不改；碰撞体只看几何，每个网格一个盒子，没有画面也能算。所以同一份代码的两个点（flat 和 textured）交互分数必然相同。

不许把照片的像素当贴图：高熵的外观只能以「文字加种子」的形式存在于代码里。

## 贴图怎么来

每条材质是三样东西：一段文字、一个种子、一个底色。它们各管一件事。

- **文字加种子管纹理。** `gwm/bench/texture.py` 在本机用 `stabilityai/sd-turbo` 生成 512 像素的贴图，生成前把网络里每个卷积都换成环绕补边，所以出来的图上下左右首尾相接。同样的文字、种子和模型在同一台机器上得到逐字节相同的图。
- **底色管颜色。** 只凭文字，模型画得出「木地板的纹理」，画不准「这块地板的颜色」：同一句 cream wall 出来可能是灰的。而平均颜色恰恰是最容易从照片里估出来的。所以 `gwm/bench/appearance.py` 只保留贴图的明暗相对变化，把平均色换成底色。这就是现实输入和生成模型之间的那层中介：观测给低维的统计量，生成模型给观测里写不下来的细节。

`texture.detail_strength` 控制细节留多少，`texture.bind_base_color: false` 可以关掉绑定做对照。生成的原图留在每次运行的 `candidate/textures_raw/`，渲染用的是 `candidate/textures/`。

## 两根轴各自怎么量

**保真度**（`gwm/bench/fidelity.py`）。留出视角按三维高斯那边的通用约定取：文件名排序后每 8 张留 1 张，room 是 311 张里的 39 张。渲染分辨率和照片一致。主看 LPIPS：生成的贴图是「合理」而不是「相同」，PSNR 和 SSIM 天然吃亏，一起报是为了能和别人的表对上。

**交互**（`harness/bench_interact.mjs`）。清单每条是 `{ target, expect }`，五种期望：

| expect | 怎么执行 | 通过条件 |
|---|---|---|
| `movable` | 抓住、提起、横拖、松手、等它稳（和 `interaction_probe.mjs` 是同一套）。依次试四个水平方向，再试不提起直接平推 | 有一种方式跟得住手、松手后不乱飞不穿地 |
| `fixed` | 看它能不能被抓 | 是固定刚体 |
| `rests` | 放开，让它在重力下待 2.5 秒 | 位移不超过 5 厘米、倾斜不超过 10 度 |
| `hinge` | 往限位外面拖 | 声明的是铰链，转得动，不冲过限位 |
| `slide` | 往限位外面拖 | 声明的是滑轨，沿轴走，不冲过限位 |

清单不给智能体看。它只拿到物体名单（场景里要有哪些 id），哪些能动要自己判断。它能用 `tools/physics.sh` 把自己声明成能动的物体执行一遍，但不知道清单期望的是哪些。

## 第一次运行的结果（2026-11-25）

`room` 场景，39 个留出视角，`images_4` 分辨率。场景由 Claude Fable 5.2 写成：31 个物体、469 行代码、67 种材质，对着训练视角渲染比对了 12 轮。

| 点 | LPIPS ↓ | PSNR ↑ | SSIM ↑ | 交互清单 |
|---|---|---|---|---|
| 最终版，纯色 | 0.613 | 13.89 | 0.488 | 29/30 |
| 最终版，生成贴图 | 0.602 | 13.72 | 0.342 | 29/30 |
| 中途快照，纯色 | 0.593 | 15.42 | 0.513 | 28/30 |
| 中途快照，生成贴图 | 0.579 | 15.21 | 0.401 | 28/30 |
| 3DGS（论文值，`images_2`） | 0.220 | 30.63 | 0.914 | 0（按定义） |

读这张表要注意的几件事：

- **贴图几乎不改变 LPIPS**（0.613 到 0.602）。瓶颈在几何和光照：物体是盒子拼的，灯光是均匀的。先别急着微调贴图模型。
- **最终版在保真度上不如中途快照。** 智能体后期把灯光换成了均匀的环境光，画面比照片亮，像素指标掉了 1.5 dB。它在训练视角上自己看着更好，留出视角上并没有。这说明只靠智能体目测对比是不够的，它需要拿到数值反馈。
- **交互清单偏容易。** 30 条过了 29 条（没过的是沙发被判成固定），这个分数现在分不开好坏，清单需要加难：更多铰接物体、质量和摩擦、物体之间的支撑关系。
- **只有一个场景、一次运行**，没有方差。
- **这次写场景的不是 `claude -p`**，而是会话内的子智能体（当时命令行没登录）。任务说明、输入和工具与命令行路径完全相同，运行记录在 `out/bench/room/workspaces/subagent-001/agent_run.json`。命令行登录后用上面「怎么跑」里的 `agent run` 可以原样重跑。
- **运行中途改过一次交互判定。** 原来 `movable` 只朝 +x 一个方向拖，贴墙的家具被误判成拖不动；改成多方向并允许平推之后，智能体看到的自检结论前后不一致，它因此调低了几个物体的质量、缩短了边几的底杆。这两处改动是被旧判定带出来的。

网页在 `out/bench/room/report/index.html`。

## 现在要知道的局限

- **尺度是先验，不是量出来的。** COLMAP 给不出绝对尺度，这里假定相机离地高度的中位数是 1.4 米。物理只需要大致的米制尺度，但质量、摩擦这类量因此没有真值可比。
- **交互清单是对着照片写的初版，待人工复核。** 只收了没有争议的条目，钢琴、电视和地毯不计分。清单能查「判得对不对、执行出来成不成立」，查不了质量和摩擦系数准不准。
- **高斯重建那个参考点是论文值。** 不是在这台机器上算的，分辨率也不同（论文用 `images_2`，这里用 `images_4`），网页上单独标了出来。它的交互分是按定义记的 0。
- **没有沙箱。** 智能体被允许运行 node，原则上读得到磁盘上别的文件。demo 阶段靠不告诉它留出照片在哪，正式做 benchmark 要进容器。
- **相机主点偏移没有建模。** 渲染相机只用了焦距，主点按图像中心算。
- **贴图模型的许可是非商用为主。** `sd-turbo` 用的是 Stability AI Community License，做研究和 demo 可以，商用前要核对条款或者换模型。
- **铰链探针是抓着物体原点去转的。** 原点正好落在转轴上时力臂为零，门转不动。这次智能体靠把原点挪到门板中间绕过去了，探针应该改成抓在离轴最远的点上。
- **碰撞体是轴对齐的盒子。** 斜放的网格会得到偏大的包围盒，圆的东西也是方的。

## 代码在哪

| 文件 | 做什么 |
|---|---|
| `gwm/bench/data.py` | 读 COLMAP、坐标归一化、划分视角、生成场景包 |
| `gwm/bench/agent.py` | 准备工作区、调起 `claude -p` |
| `gwm/bench/texture.py` | 本机扩散模型把「文字加种子」变成可平铺贴图 |
| `gwm/bench/appearance.py` | 把贴图的平均色绑到从照片估的底色上 |
| `gwm/bench/fidelity.py` | PSNR、SSIM、LPIPS |
| `gwm/bench/run.py` | 命令行入口：`pack`、`eval`、`report` |
| `gwm/bench/report.py` | 生成对比网页 |
| `kernel/bench_scene.js` | 把智能体写的模块变成场景和物理注册表（浏览器和 Node 共用） |
| `kernel/bench_view.{html,js}` | 渲染页 |
| `harness/bench_render.mjs` | 无头浏览器按给定相机出图 |
| `harness/bench_interact.mjs` | 执行交互清单 |
| `configs/bench/room.*` | 场景配置、物体名单、交互清单 |
| `prompts/bench_scene_writer.md` | 给智能体的任务说明 |
| `examples/bench_smoke/` | 管线的冒烟场景 |
