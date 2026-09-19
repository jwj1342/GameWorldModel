#!/bin/bash
#SBATCH --job-name=gwm-vlm
#SBATCH --account=aip-zhouyang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:59:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# vLLM 服务：Qwen3.5-27B-FP8 单卡；把端点写到 $GWM_SCRATCH/vlm_endpoint.txt；作业内先做一次单图 + JSON 约束冒烟
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh --no-venv
module load cuda/12.6   # flashinfer JIT needs nvcc / CUDA_HOME
source $GWM_PROJECT/venv-vllm/bin/activate; unset PIP_PREFIX
export VLLM_USE_FLASHINFER_SAMPLER=0   # avoid JIT-compiling the sampler; PyTorch sampler is fine here
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
MODEL=$GWM_WEIGHTS/qwen3.5-27b-fp8
PORT=${PORT:-8000}; MAXLEN=${MAXLEN:-24576}; MAXSEQS=${MAXSEQS:-4}
HOST=$(hostname -i | awk '{print $1}')
echo "=== $(date) on $(hostname) ($HOST) ==="; nvidia-smi -L; python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'torch cuda',torch.version.cuda)"; echo "CUDA_HOME=$CUDA_HOME"; which nvcc && nvcc --version | tail -1
rm -f $GWM_SCRATCH/vlm_endpoint.txt
vllm serve "$MODEL" --served-model-name qwen3.5-27b-fp8 --host 0.0.0.0 --port $PORT \
  --max-model-len $MAXLEN --max-num-seqs $MAXSEQS --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":12}' \
  --reasoning-parser qwen3 --structured-outputs-config '{"backend":"guidance"}' \
  --enable-prefix-caching --trust-remote-code > $GWM_SCRATCH/logs/vllm_server_$SLURM_JOB_ID.log 2>&1 &
VPID=$!
for i in $(seq 1 120); do
  if curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q qwen; then break; fi
  if ! kill -0 $VPID 2>/dev/null; then echo "vllm died:"; tail -60 $GWM_SCRATCH/logs/vllm_server_$SLURM_JOB_ID.log; exit 1; fi
  sleep 10
done
echo "http://$HOST:$PORT/v1" > $GWM_SCRATCH/vlm_endpoint.txt
echo "=== endpoint http://$HOST:$PORT/v1 ready after $((i*10))s ==="
echo "=== smoke: text ==="
curl -s -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{"model":"qwen3.5-27b-fp8","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":20,"chat_template_kwargs":{"enable_thinking":false}}' | head -c 600; echo
echo "=== smoke: image + json schema ==="
IMG=$(ls $REPO/data/clips/keyframes_smoke/*.jpg 2>/dev/null | head -1)
if [ -z "$IMG" ]; then mkdir -p $GWM_SCRATCH/smoke_img; ffmpeg -loglevel error -y -ss 5 -i $REPO/data/clips/raw/plarail_osaka_metro_ccbysa40.webm -frames:v 1 -vf scale=640:-1 $GWM_SCRATCH/smoke_img/f.jpg; IMG=$GWM_SCRATCH/smoke_img/f.jpg; fi
B64=$(base64 -w0 "$IMG")
python - "$B64" <<'PY'
import json, sys, urllib.request
b64 = sys.argv[1]
body = {"model": "qwen3.5-27b-fp8", "max_tokens": 400, "temperature": 0.2,
  "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}, {"type": "text", "text": "List the distinct physical objects in this frame as noun phrases with a rough 3D size in metres."}]}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "objs", "schema": {"type": "object", "properties": {"objects": {"type": "array", "items": {"type": "object", "properties": {"phrase": {"type": "string"}, "size_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "moving": {"type": "boolean"}}, "required": ["phrase", "size_m", "moving"]}}}, "required": ["objects"]}}},
  "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % __import__('os').environ.get('PORT','8000'), data=json.dumps(body).encode(), headers={"content-type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=300))
print(json.dumps(r["choices"][0]["message"], ensure_ascii=False)[:1500]); print("usage:", r.get("usage"))
PY
echo "=== serving until job end; endpoint file: $GWM_SCRATCH/vlm_endpoint.txt ==="
wait $VPID
