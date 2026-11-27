#!/bin/bash
# 两轴评测（gwm/bench）在 Vulcan 上的环境。登录节点和计算节点都 source 这个文件：
#   source scripts/setup_bench_env.sh
# venv 用集群轮子库建（--no-index），cuda 模块要和 torch 编译时的 CUDA 版本一致（13.2）；Chromium 和 npm 包要在登录节点下载，计算节点的代理不放行 npm 源。
# 评测自己的东西都在主仓库的 work/bench/ 下：out、venv、node（node_modules 和 Chromium）、models、torch_extensions。
# worktree 里的 out、venv、node_modules、models、data/mipnerf360 是指过去的相对软链接。HF 缓存和主管线共用，不在这里。
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 python/3.12.4 nodejs/24.22.3
BENCH=/scratch/$USER/research/GameWorldModel/work/bench
export PLAYWRIGHT_BROWSERS_PATH=$BENCH/node/ms-playwright
export HF_HOME=/scratch/$USER/hf_cache
export XDG_CACHE_HOME=/scratch/$USER/cache
# 轮子库里 gsplat 的预编译内核对不上 torch 2.14，第一次用时会用 nvcc 现场编译（约 7 分钟），缓存在这里
export TORCH_EXTENSIONS_DIR=$BENCH/torch_extensions MAX_JOBS=6
source $BENCH/venv/bin/activate
