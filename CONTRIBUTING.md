# 怎么参与

main 分支开了保护，不能直接往上推，改动都通过 Pull Request 进来，需要一个人批准之后才能合并。

## 流程

```bash
git checkout -b 你的分支名        # 起个能看出在做什么的名字，比如 fix-motion-classification
# 改代码
python -m pytest -q tests        # 开 PR 之前先跑一遍，11 个测试应该全过
git commit -am "一句话说清楚改了什么"
git push -u origin 你的分支名
gh pr create                     # 或者去网页上开
```

PR 里写清楚改了什么、为什么改。如果动了感知或者生成那部分，最好附上改动前后跑同一段视频的对比，`docs/status.md` 里有当前的分数可以对照。

评论都解决掉、有一个批准之后就能合并。

## 提交之前注意

不要提交 API key。`.secrets/` 已经在 `.gitignore` 里了，但还是自己确认一下。

不要提交视频和运行产物。`data/clips/` 和 `out/` 也都忽略了。想分享结果的话用 `scripts/collect_results.sh` 把该看的挑到 `docs/results/`，那个目录是进 git 的。

依赖有变动的话，`requirements.txt` 是给所有人用的通用列表，`requirements-vulcan.lock.txt` 是集群上的精确版本，两个都要更新。

## 写代码时守的几条

**科研项目**

1. 一切实验可复现：每次运行写 `run.json`（git commit、配置快照、模型与权重、随机种子、时间、节点），产物按 `out/<clip>/<run_id>/` 分目录，永不覆盖。
2. 配置驱动：所有选型、阈值、轮数、候选数、消融开关都在 `configs/*.yaml`；代码里没有魔法数。
3. 评测钩子从第一天就在：指标脚本与消融开关（`no_evidence`、`no_feedback`、`single_stage`）和主流程一起写，后面做 benchmark 不用重构。
4. 提示词文件化：`prompts/` 下每个角色一个文件，few-shot 单独存放，改动可 diff。
5. 先合成再真实：先用内核自渲染的合成片段（有真值）对齐管线，再上真实视频。
6. 依赖锁定：`requirements.txt`、`requirements-vllm.txt`、`package-lock.json`，`/project` 下的 tar 包与权重目录记录 sha。
7. 数据许可随数据走：`data/clips/raw/SOURCES.md` 记录来源、作者、许可原文、直链、sha1。

**LLM 驱动的图形程序合成**

1. 固定内核、生成数据：模型只产出 `program.json`，运行时是固定代码，状态 schema 固化。
2. 校验先于渲染：schema 与语义校验失败的程序直接回喂，不进渲染。
3. 结构化反馈：数值指标先出逐物体 pass/fail 子句，VLM 只看失败区域，输出受 schema 约束的建议。
4. 有界循环：轮数、候选数、调用次数与 token 都有上限；不优则回退。
5. 每个阶段一个 smoke 脚本，能在 CPU 作业里独立跑。

**集群**：npm 与浏览器在登录节点装好打包；Python venv 在作业里建；权重预下载到 `/project`；中间产物在 `$SCRATCH`；GPU 只给感知、VLM 与批量渲染；任何超过几分钟的计算都进作业。

## 代码放在哪

```
GameWorldModel/
  RP.md  README.md  RUNNING.md  docs/  survey/
  configs/     default.yaml（阈值与选型） local.yaml / vulcan.yaml（站点路径，按 GWM_SITE 或自动检测选）
               perception/{cpu,gt}.yaml vlm/{openrouter_*,vllm_local,mock}.yaml ablations/*.yaml
  prompts/     common_dsl.md writer_*.md critic.md namer.md playtest_review.md fewshot/
  gwm/         Python 管线
    config.py errors.py run_clip.py
    perception/  frames.py base.py masks.py vggt_backend.py static_backend.py grounded_sam2_backend.py gt_backend.py evidence.py run.py
    synthesis/   vlm.py mock_vlm.py direct.py writer.py critic.py loop.py
    compiler/    schema/program.schema.json validate.py assets.py bundle.py compile.py ids.py
    feedback/    render.py metrics.py
    binding/     platformer.py
    playtest/    autopilot.py
  kernel/      固定运行时（buildless ESM）：main.js scene.js motions.js physics.js passes.js replay_camera.js templates/platformer_3p.js index.html
  harness/     Node + Playwright：common.mjs render.mjs playtest.mjs record.mjs serve.mjs
  examples/handwritten/program.json
  scripts/     setup_env.sh env_setup.sh stage_node_deps.sh download_weights.py pipeline.sh smoke_kernel.sh
               run_tests.sh serve_vlm.sh trim_clips.sh node_harness.sh collect_results.sh make_release.sh release_files/
  tests/       单元测试（CPU 作业跑）
  data/clips/{raw,trimmed}   视频（不入 git；SOURCES.md 入 git）
  out/<clip>/<run_id>/        产物（不入 git）
  .secrets/    API 密钥（不入 git）
  requirements.txt            通用依赖；requirements-vulcan.lock.txt 是集群上的精确版本
  models/ deps/ venv/         软链接到 /project/aip-zhouyang/jwj/GameWorldModel/{weights,deps,venv}
```

## 模块之间的边界

| 边界 | 契约 | 可替换实现 |
|---|---|---|
| 感知几何 | `GeometryBackend.estimate(frames, indices, cfg) -> Geometry`（内参、cam→world、深度、可选动态掩码、scale 标记） | VGGT / 静态相机 + 单目深度 / 真值（合成片段） |
| 感知分割 | `SegmentationBackend.segment(frames, indices, phrases, cfg) -> Tracks`（逐帧掩码 + 跨帧 ID） | Grounding DINO + SAM 2.1 / 真值（合成片段） |
| 证据 | `evidence.json`（字段见 `architecture.md` 5.1） | 只依赖上面两个协议 |
| 程序合成 | `VLMClient.chat(system, text, images, json_schema, temperature)` | OpenRouter（默认）/ 自托管 vLLM / 假模型 mock |
| 程序 | `program.json` + JSON Schema + 语义校验 | Writer / Critic / 直译（`synthesis/direct.py`） |
| 编译 | `validate → resolve_assets → bundle` 三个纯函数 | `AssetResolver` 策略链（MVP 只有 primitive） |
| 运行时 | 内核解释 `program.json`；对外只有 `window.__game` | 模板（`templates/*.js`）与运动节点注册表可扩展 |
| 反馈 | harness 命令行契约：`render.mjs --game DIR --times ... --out DIR` | Python 不关心浏览器细节 |
| 编排 | 每阶段幂等，产物存在即跳过，`--resume` 复用 | — |

## 环境怎么搭

看 [RUNNING.md](RUNNING.md)，里面分了在自己电脑上跑和在 HPC 上跑两种情况。

只改场景程序、编译器或者 three.js 内核的话，不用下那 9.6 GB 的感知权重，装好 Python 依赖和 `npm install` 就能用 `examples/handwritten/program.json` 调试。
