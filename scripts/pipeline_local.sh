#!/bin/bash
#SBATCH --job-name=gwm-pipeline-local
#SBATCH --account=aip-zhouyang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 用本地自托管的开源模型跑完整管线，不依赖任何外部 API。
# 服务和管线放同一个作业，省掉跨节点端口。
# 用途：#3 重跑、#15 本地可复现、以及产出 #17 需要的感知产物（evidence + 掩码）。
# 用法：sbatch scripts/pipeline_local.sh <clip 名> <视频路径> [额外 --config]
set -u
CLIP=${1:?需要 clip 名}; VIDEO=${2:?需要视频路径}; shift 2 || true
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh --no-venv
module load cuda/12.6
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=$GWM_WEIGHTS/qwen3.5-27b-fp8
PORT=${PORT:-8312}
LOG=$GWM_SCRATCH/logs/vllm_pipeline_$SLURM_JOB_ID.log
mkdir -p $GWM_SCRATCH/logs
echo "=== $(date) on $(hostname) ==="; nvidia-smi -L

(
  source $GWM_PROJECT/venv-vllm/bin/activate; unset PIP_PREFIX
  vllm serve "$MODEL" --served-model-name qwen3.5-27b-fp8 --host 127.0.0.1 --port $PORT \
    --max-model-len 24576 --max-num-seqs 4 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":12}' \
    --structured-outputs-config '{"backend":"guidance"}' \
    --enable-prefix-caching --trust-remote-code > $LOG 2>&1
) &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT

echo "=== 等服务（最多 20 分钟）==="
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q qwen && { echo "服务就绪 $((i*10))s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "vllm 挂了:"; tail -40 $LOG; exit 1; }
  sleep 10
done
curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q qwen || { echo "超时"; tail -40 $LOG; exit 1; }

# 管线用主 venv。vLLM 已经吃掉整张卡（27B FP8 光权重就 33G），所以感知这步走 CPU。
# vLLM 的子 shell 启动时已经拿到卡了，之后改这个环境变量不影响它。
# 感知在 CPU 上一条视频约五分钟，可以接受。
source $GWM_VENV/bin/activate; unset PIP_PREFIX
export CUDA_VISIBLE_DEVICES=""
cd $REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export GWM_VLM_ENDPOINT=http://127.0.0.1:$PORT/v1
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers

python -m gwm.run_clip --video "$VIDEO" --clip "$CLIP" \
  --config configs/vlm/local_inline.yaml "$@"
# 注：随机种子（--seed）来自 PR #22，合并之后在上面补 --seed "${SEED:-1}"
echo "=== DONE $(date) ==="
