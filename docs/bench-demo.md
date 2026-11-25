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
venv/bin/python -m gwm.bench.agent prepare --config configs/bench/room.yaml --workspace $W   # 加 --no-manifest 则不给物体名单
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

**几何**（`gwm/bench/geometry.py`）。PSNR、SSIM、LPIPS 把几何、光照和贴图混在一起量，单看它们分不清瓶颈在哪。几何单独量：留出视角再按深度模式渲染一遍（`kernel/bench_view.js` 的 `mode=depth`，视空间深度按毫米写进 RGB），然后取 COLMAP 在这张照片里实际观测到的三维点（场景包的 `eval/observed.json`，被挡住的和画面外的不算），投到同一个相机里，和那个像素上渲染出来的深度比。报四个数：`coverage` 是投进画面的点里有几何的比例，`median_abs_m` 是深度差绝对值的中位数，`within_10cm` 是深度差不超过 10 厘米的比例（没几何的点算不合格），`median_rel` 是相对误差的中位数。单位是场景包的先验尺度，大致是米。三维高斯基线用同一套点、同一个函数量。

**快照轨迹**（`python -m gwm.bench.run trajectory`）。智能体每次跑 `tools/preview.sh` 都会先把当前的 `scene.js` 和 `materials.json` 存进工作区的 `snapshots/`。事后把每一份快照都在留出视角和给智能体看的训练视角上评一遍（纯色模式），连同几何和代码行数写进 `trajectory.json`。它回答两件事：代码写到多少行以后保真度不再涨，以及智能体对着训练视角越改越好的时候，留出视角是不是也在变好。

**交互**（`harness/bench_interact.mjs`）。清单每条是 `{ target, expect }`，五种期望：

| expect | 怎么执行 | 通过条件 |
|---|---|---|
| `movable` | 抓住、提起、横拖、松手、等它稳（和 `interaction_probe.mjs` 是同一套）。依次试四个水平方向，再试不提起直接平推 | 有一种方式跟得住手、松手后不乱飞不穿地 |
| `fixed` | 看它能不能被抓 | 是固定刚体 |
| `rests` | 放开，让它在重力下待 2.5 秒 | 位移不超过 5 厘米、倾斜不超过 10 度 |
| `hinge` | 往限位外面拖 | 声明的是铰链，转得动，不冲过限位 |
| `slide` | 往限位外面拖 | 声明的是滑轨，沿轴走，不冲过限位 |

清单不给智能体看。它只拿到物体名单（场景里要有哪些 id），哪些能动要自己判断。它能用 `tools/physics.sh` 把自己声明成能动的物体执行一遍，但不知道清单期望的是哪些。

**连续交互**（`harness/bench_sequence.mjs`，清单在 `configs/bench/room.sequence.json`）。上面那份清单每条各开一个新世界，只能说明单个物体判得对不对，说明不了动作之间的后果。连续交互在同一个世界里按顺序做一串动作，中途不重置：推边几，看上面的碗、水瓶、杯子是不是跟着走；把杯子端到地毯上、把碗端到脚凳上；推躺椅，看靠垫还在不在；开门；再把边几推回去，看前面放下的东西还在不在原处；最后看没碰过的东西有没有被带动。每一步检查四种事：被推的东西走到了没有（`followed`），某样东西是不是放在某样东西上（`on`，从它中心往下打射线看正下方是谁、隔了多远），有没有倒（`upright`），门转了没有（`turned`），没碰过的有没有动（`stayed`）。

「手」的定义会直接影响分数，这里写明：内核的抓取是给鼠标拖动用的，抓住期间关掉重力、不管朝向；照原样平推一张悬臂边几，它会被台面上东西的重量带着慢慢转起来，把东西甩出去。连续交互里的手稳住抓住那一刻的朝向，移动缓起缓停，端东西时先抬到 1.2 米以上再平移，不去扫别的物体。

同一串动作跑两种条件。默认（lazy）是内核原本的样子：没被碰到的物体是焊死的，只有手上那个和它直接碰到的会被唤醒。`settled` 在开始前把所有能动的物体交给物理落稳一遍。两种都跑，才分得清失败是场景写得不对，还是运行时只唤醒直接接触者。

**按位置找目标**（`configs/bench/room.anchors.json`）。清单和连续交互里的物体名字只是标签，评测按锚点（房间坐标系里的一个点）去找场景里占着那个点的物体。两个标签落到同一个物体上记为 `merged`，除了「推不动」那类，其余几类都判不通过：碗和边几写成了一个物体，拖「碗」拖的是一整张桌子，不能算碗能拿起来。这样就可以不给智能体物体名单，让它自己决定房间里有哪些物体、每个拆成多细，粒度切得对不对会直接反映在分数里。锚点目前取自第一次运行的场景（每个物体最大那块网格的中心），待对照照片复核；在第一次运行的场景上，按位置找和按 id 找的结果完全相同（29/30）。

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
- **单步清单把后果记下来了，但没计分。** 拖边几那一条的记录里，台面上 3 样东西被撞开了 0.86 米；拖躺椅那一条，靠垫被甩出去 2.3 米（`interact.json` 里的 `max_disturbance`）。两条都算通过。
- **连续交互是 22/24（lazy）和 19/24（settled）。** 两种条件下都是推边几时杯子从台面上滑了下去：它的包围盒边缘正好和台面边缘重合。之后把它端到地毯上时它是躺着的，这是前一步的连带。settled 下另有三条：边几推完松手后退回了约 8 厘米（底杆落稳后贴着高出地面 3 毫米的地毯边，退回的确切原因还没查清）、碗放到脚凳上是倒扣的、推回边几时水瓶掉了。结果在 `docs/results/bench-room-20261125/sequence_*.json`。这一串动作和期望是 Claude 按常识写的初版，待人工复核。
- **只有一个场景、一次运行**，没有方差。
- **这次写场景的不是 `claude -p`**，而是会话内的子智能体（当时命令行没登录）。任务说明、输入和工具与命令行路径完全相同，运行记录在 `out/bench/room/workspaces/subagent-001/agent_run.json`。命令行登录后用上面「怎么跑」里的 `agent run` 可以原样重跑。
- **运行中途改过一次交互判定。** 原来 `movable` 只朝 +x 一个方向拖，贴墙的家具被误判成拖不动；改成多方向并允许平推之后，智能体看到的自检结论前后不一致，它因此调低了几个物体的质量、缩短了边几的底杆。这两处改动是被旧判定带出来的。

网页在 `out/bench/room/report/index.html`。

## 在 Vulcan 上跑

环境在 `scripts/setup_bench_env.sh`（模块、venv、Chromium 和缓存的路径）。计算节点的代理不放行 npm 源，所以 npm 包和 Playwright 的 Chromium 要先在登录节点下载（`npm install --ignore-scripts`，再 `npx playwright install chromium`，`PLAYWRIGHT_BROWSERS_PATH` 指到 scratch）；Python 依赖全部能从集群轮子库装（`pip install --no-index -r requirements-bench.txt`，gsplat 也在）。

三维高斯基线：`python -m gwm.bench.gs_baseline --config configs/bench/room.yaml`。它用 gsplat 的光栅化和默认致密化策略按标准设置训练 30k 步，坐标系、相机（竖直视场角、主点在图像中心）和留出视角都和场景包一致，评测直接用 `fidelity.py` 和 `geometry.py`，产物在 `out/bench/<scene>/baselines/`。

## 三维高斯基线和几何（2026-11-25，Vulcan）

在 Vulcan 上重新评测了入库的场景，纯色模式的三项指标和第一次运行一致（PSNR 13.886、SSIM 0.4876、LPIPS 0.6128），说明两台机器的环境对得上。同一个场景包上训练了本机的三维高斯基线（gsplat，30k 步，L40S 上 13 分钟，最后 147 万个高斯）。两边都加了几何这一列。

| 点 | LPIPS ↓ | PSNR ↑ | SSIM ↑ | 深度差中位数 | 10 厘米内 | 交互清单 |
|---|---|---|---|---|---|---|
| 智能体写的场景，纯色 | 0.613 | 13.89 | 0.488 | 3.8 厘米 | 77% | 29/30 |
| 三维高斯（本机，`images_4`） | 0.171 | 32.23 | 0.940 | 1.6 厘米 | 88% | 0（按定义） |
| 三维高斯（论文值，`images_2`） | 0.220 | 30.63 | 0.914 | — | — | 0（按定义） |

几何上两边差得没有外观那么多。代码场景深度差的中位数是三维高斯的 2.4 倍，10 厘米内的比例是 77% 对 88%；把观测点按误差上色叠到图上看，墙、地板、地毯这些大面基本都对，误差集中在几个摆错位置或者尺寸偏大的前景物体上（比如左前方那块黑色的音箱，把后面的书架和窗户整片挡住了）。外观上差得远得多：LPIPS 0.61 对 0.17。所以之前「瓶颈在几何和光照」的说法要改：布局已经大致对了，拉开差距的主要是外观和光照，几何上要补的是前景物体的位置和尺寸。

读这张表要注意两点。一是几何只在 COLMAP 的特征点上量，这些点偏向有纹理的表面，大片白墙上点很少。二是三维高斯本来就是从这些稀疏点初始化的，智能体也拿到了其中的两万个，所以这一列对两边都不算盲测。产物在 `docs/results/bench-room-3dgs/`。

## 现在要知道的局限

- **尺度是先验，不是量出来的。** COLMAP 给不出绝对尺度，这里假定相机离地高度的中位数是 1.4 米。物理只需要大致的米制尺度，但质量、摩擦这类量因此没有真值可比。
- **交互清单是对着照片写的初版，待人工复核。** 只收了没有争议的条目，钢琴、电视和地毯不计分。清单能查「判得对不对、执行出来成不成立」，查不了质量和摩擦系数准不准。
- **高斯重建的参考点现在有本机的数了**（上一节），论文值留作对照。它的交互分是按定义记的 0。
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
| `gwm/bench/geometry.py` | 渲染深度和 COLMAP 观测点之间的几何误差 |
| `gwm/bench/gs_baseline.py` | 本机的三维高斯基线，同分辨率、同留出视角 |
| `gwm/bench/run.py` | 命令行入口：`pack`、`eval`、`trajectory`、`report` |
| `gwm/bench/report.py` | 生成对比网页 |
| `kernel/bench_scene.js` | 把智能体写的模块变成场景和物理注册表（浏览器和 Node 共用） |
| `kernel/bench_view.{html,js}` | 渲染页 |
| `harness/bench_render.mjs` | 无头浏览器按给定相机出图 |
| `harness/bench_interact.mjs` | 执行交互清单；按锚点把标签对到场景物体上 |
| `harness/bench_sequence.mjs` | 连续交互：同一个世界里按顺序做一串动作 |
| `configs/bench/room.*` | 场景配置、物体名单、交互清单、锚点、连续交互清单 |
| `prompts/bench_scene_writer.md` | 给智能体的任务说明 |
| `examples/bench_smoke/` | 管线的冒烟场景 |
