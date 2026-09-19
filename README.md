# GameWorldModel

这个项目做一件事。给它一段十几秒到一分钟的单目视频，它把视频里的场景和运动恢复成一份可执行的场景程序，再把程序编译成一个能在浏览器里玩的 three.js 小游戏。研究提案在 RP.md，设计文档在 docs 目录，目前已经有一个跑通的 MVP，两段真实视频和一段合成片段都能从视频一路走到可玩的游戏。

## 它是怎么工作的

视频先抽帧。VGGT 估计相机位姿和深度，Grounding DINO 加 SAM 2.1 把画面里的物体分割出来并跨帧跟踪。这些结果被整理成一份证据文件，里面有地面平面、每个物体随时间的三维包围盒和运动类型猜测。然后一个视觉语言模型根据关键帧和证据分三步写出场景程序，先写相机和静态结构，再写物体，最后写运动和事件。默认用 OpenRouter 上的 GLM-4.6V，一条视频大约六次调用，费用一美分左右。程序经过 JSON Schema 和语义校验后编译成一个 game 目录，固定的 three.js 内核负责解释它，Rapier 提供物理。编译好的场景在无头浏览器里按视频的相机重新渲染，和视频逐物体比较掩码重叠、位置和轨迹误差，不合格的地方交给模型修订，最多三轮。通过之后绑定第三人称平台跳跃模板，自动试玩一遍确认能走到终点，最后产出报告。模型不可用时整条链也能走完，程序直接从证据翻译得到。

## 目录

RP.md 是研究提案。docs 放设计文档，其中 status.md 记录进展和已知问题，engineering.md 讲项目怎么组织，results 放每次跑出来的结果。survey 放前期调研的原始报告。gwm 是 Python 管线，按感知、程序生成、编译、反馈、绑定、试玩分成子包，run_clip.py 是入口。kernel 是在浏览器里运行的固定内核，harness 是用 Playwright 驱动内核做渲染、试玩和录制的脚本。configs 放所有可调参数和选型，prompts 放提示词，examples 放手写的示例程序，tests 放单元测试，scripts 放集群作业脚本。data/clips 放视频，out 放每次运行的产物，.secrets 放 API 密钥，这三个目录都不进 git。

## 在 Vulcan 上跑

所有重计算都通过 Slurm 作业提交，登录节点只做安装、打包和提交。第一次使用先在登录节点执行 scripts/stage_node_deps.sh，把 Node 依赖和浏览器打包放到 project 空间，然后提交 scripts/env_setup.sh 建好 Python 环境并下载权重。OpenRouter 的密钥放在 .secrets/openrouter.key。

跑一条视频用下面这条命令。第一个参数是 data/clips/trimmed 里的文件名，第二个参数是给检测器的名词短语，可以留空。

    sbatch scripts/pipeline.sh plarail_train "toy train,train track,floor" --config configs/perception/cpu.yaml

感知模型在 CPU 节点上就能跑，一条视频大约三分钟加上模型调用的一两分钟。加 --no-vlm 就不调用模型，用证据直译。加 --gres=gpu:l40s:1 可以用 GPU。换模型用 --config configs/vlm/openrouter_qwen122b.yaml 这类文件，configs/vlm 下还有 Kimi、Opus、自托管 vLLM 和一个不花钱的假模型。合成片段可以用 configs/perception/gt.yaml 让感知直接取源程序的真值，用来单独测试后面的阶段。手写程序的内核冒烟用 scripts/smoke_kernel.sh，单元测试用 scripts/run_tests.sh。新加的视频先放到 data/clips/raw，在 scripts/trim_clips.sh 里加一行裁剪规则。

产物在 out 下按视频名和运行编号分目录。report.md 是汇总，program.json 是场景程序，game 目录用任意静态服务器打开 index.html 就能玩，WASD 移动，空格跳，R 在回放和游玩之间切换。perception/overlay.mp4 把感知结果画回视频上，方便检查。scripts/collect_results.sh 可以把一次运行里值得看的东西复制到 docs/results。

## 现在做到了什么

Plarail 玩具火车和一条工业输送线这两段版权干净的真实视频，加上一段合成片段，都能从视频一路跑到可玩的游戏，自动试玩都能走到终点。合成片段用真值感知时重建分数 0.71，说明感知之后的各个阶段是对的。用神经感知和真实视频时分数低很多，主要原因是几何体替身和真实画面在外观上不可比，以及运动类型的判断还不稳。模型写出来的程序目前和直接从证据翻译差不多，修订环节还没带来提升。这些数字、与设计的偏差和已知问题都写在 docs/status.md 里。
