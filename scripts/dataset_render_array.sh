#!/bin/bash
# 数据集渲染的作业数组。Vulcan 只有两个 CPU 节点，所以要切成多个小作业去抢 backfill，
# 一个大作业会一直排在那里。
#
# 用法：
#   sbatch --array=0-7 scripts/dataset_render_array.sh $GWM_SCRATCH/dataset/train
#   （数组下标就是 --shard，数组长度就是 --shards，两者由 SLURM_ARRAY_TASK_COUNT 对上）
#
#SBATCH --job-name=gwm-render
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%A_%a.out
set -eu
ROOT=${1:?用法: sbatch --array=0-N scripts/dataset_render_array.sh <generate_dataset 的输出目录>}
source "$(dirname "$0")/setup_env.sh"
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
python -m scripts.render_dataset \
  --root "$ROOT" --out "${ROOT}_rendered" \
  --width "${GWM_RENDER_WIDTH:-640}" --height "${GWM_RENDER_HEIGHT:-360}" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT"
