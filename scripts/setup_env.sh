#!/bin/bash
# 统一的模块加载脚本：登录节点与作业脚本都 source 这个文件。
# 用法：source scripts/setup_env.sh [--no-venv]
module --force purge
module load StdEnv/2023 gcc/12.3 opencv/4.11.0 python/3.12.4 ffmpeg/7.1.1 nodejs/20.16.0
export GWM_PROJECT=/project/aip-zhouyang/jwj/GameWorldModel
export GWM_VENV=$GWM_PROJECT/venv
export GWM_WEIGHTS=$GWM_PROJECT/weights
export GWM_DEPS=$GWM_PROJECT/deps
export GWM_SCRATCH=${SCRATCH:-/scratch/$USER}/gwm
export HF_HOME=${SCRATCH:-/scratch/$USER}/hf_cache
export XDG_CACHE_HOME=${SCRATCH:-/scratch/$USER}/cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$GWM_SCRATCH" "$HF_HOME" "$XDG_CACHE_HOME"
if [ "${1:-}" != "--no-venv" ] && [ -f "$GWM_VENV/bin/activate" ]; then
  source "$GWM_VENV/bin/activate"; unset PIP_PREFIX
fi
# HuggingFace token（文件在 /project 下，权限 600，不入 git，不打印）
if [ -f "$GWM_PROJECT/.hf_token" ]; then export HF_TOKEN="$(cat "$GWM_PROJECT/.hf_token")"; fi
