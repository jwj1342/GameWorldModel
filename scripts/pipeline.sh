#!/bin/bash
#SBATCH --job-name=gwm-pipe
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 端到端：视频 -> 游戏。用法: sbatch scripts/pipeline.sh <clip_name> [phrases] [extra run_clip args...]
#   clip_name 对应 data/clips/trimmed/<clip_name>.mp4 ；phrases 逗号分隔（可为 ""）
#   默认 CPU 节点（感知模型在 CPU 上约 3 分钟）；要用 GPU 加 sbatch --gres=gpu:l40s:1
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
export GWM_NODE_ROOT=$SLURM_TMPDIR/node
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT && export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
CLIP=${1:?clip name}; PHR=${2:-}; shift 2 || true
# 需要 VLM 时等待端点就绪（vLLM 作业启动与编译约 5~8 分钟；最多等 40 分钟）
if [[ " $* " != *" --no-vlm "* ]] && [ "${GWM_VLM_LOCAL:-0}" = "1" ]; then
  for i in $(seq 1 80); do
    EP=$(cat $GWM_SCRATCH/vlm_endpoint.txt 2>/dev/null || true)
    if [ -n "$EP" ] && curl -s -m 5 "$EP/models" | grep -q qwen; then echo "VLM endpoint ready: $EP"; break; fi
    [ $i -eq 80 ] && echo "WARNING: VLM endpoint not ready after 40 min; run_clip will fall back to direct translation"
    sleep 30
  done
fi
echo "=== $(date) on $(hostname) clip=$CLIP ==="; command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "(CPU job)"
bash $REPO/scripts/trim_clips.sh
cd $REPO
ARGS=(--video data/clips/trimmed/$CLIP.mp4 --clip $CLIP)
[ -n "$PHR" ] && ARGS+=(--phrases "$PHR")
python -m gwm.run_clip "${ARGS[@]}" "$@"
echo "=== DONE $(date) ==="
