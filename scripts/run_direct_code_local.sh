#!/bin/bash
#SBATCH --job-name=gwm-direct-code-vlm
#SBATCH --account=aip-zhouyang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# issue #11 的直接出代码对照，用本地自托管的开源模型跑。
# 服务和实验放在同一个作业里，省掉跨节点端口的麻烦。
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh --no-venv
module load cuda/12.6                       # flashinfer 的 JIT 需要 nvcc
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
export VLLM_USE_FLASHINFER_SAMPLER=0        # 避免 JIT 编译采样器，PyTorch 的够用
MODEL=$GWM_WEIGHTS/qwen3.5-27b-fp8
PORT=${PORT:-8313}
LOG=$GWM_SCRATCH/logs/vllm_dc_$SLURM_JOB_ID.log
mkdir -p $GWM_SCRATCH/logs
echo "=== $(date) on $(hostname) ==="; nvidia-smi -L

# ---- 起服务（用 venv-vllm）----
(
  source $GWM_PROJECT/venv-vllm/bin/activate; unset PIP_PREFIX
  vllm serve "$MODEL" --served-model-name qwen3.5-27b-fp8 --host 127.0.0.1 --port $PORT \
    --max-model-len 24576 --max-num-seqs 4 --gpu-memory-utilization 0.90 \
    --structured-outputs-config '{"backend":"guidance"}' \
    --enable-prefix-caching --trust-remote-code > $LOG 2>&1
) &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT

echo "=== 等服务起来（最多 20 分钟）==="
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q qwen && { echo "服务就绪，用了 $((i*10)) 秒"; break; }
  kill -0 $VPID 2>/dev/null || { echo "vllm 挂了:"; tail -40 $LOG; exit 1; }
  sleep 10
done
curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q qwen || { echo "等超时了"; tail -40 $LOG; exit 1; }

# ---- 跑实验（用主 venv，它有 gwm 的依赖）----
source $GWM_VENV/bin/activate; unset PIP_PREFIX
cd $REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export GWM_VLM_ENDPOINT=http://127.0.0.1:$PORT/v1
RUN=${RUN:?需要设 RUN 指向一次完整运行的目录}
python scripts/run_direct_code.py \
  --evidence $RUN/perception/evidence.json \
  --keyframes $RUN/perception/keyframes \
  --out $GWM_SCRATCH/direct_code_vlm/$SLURM_JOB_ID \
  --config configs/vlm/local_inline.yaml --config configs/ablations/direct_code.yaml
echo "=== DONE $(date) ==="
