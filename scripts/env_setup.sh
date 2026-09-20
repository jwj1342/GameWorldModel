#!/bin/bash
#SBATCH --job-name=gwm-env
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 建 Python 环境并下载权重，可重复运行。默认只建感知环境 venv；加参数 --with-vllm 再建自托管 vLLM 用的 venv-vllm。
# 依赖优先来自集群 wheelhouse（--no-index），缺的经代理从 PyPI 装。
set -u
REPO=/scratch/jwj/research/GameWorldModel
mkdir -p /scratch/jwj/gwm/logs
source $REPO/scripts/setup_env.sh --no-venv
echo "=== $(date) on $(hostname) ==="

echo "=== perception venv ($GWM_VENV) ==="
[ -f "$GWM_VENV/bin/activate" ] || python -m venv "$GWM_VENV"
source "$GWM_VENV/bin/activate"; unset PIP_PREFIX
pip install --no-index --upgrade pip 2>&1 | tail -1
# torch 钉 2.14 与 torchvision 0.29 配对；不装 flash_attn（它会把 torch 拉回 2.9），注意力用 sdpa
PKGS="torch==2.14.0 torchvision==0.29.0 transformers openai jsonschema pydantic pyyaml numpy scipy pillow av decord huggingface_hub safetensors timm einops imageio imageio_ffmpeg accelerate tqdm pytest jsonpatch matplotlib scikit_learn"
for p in $PKGS; do pip install --no-index "$p" > /tmp/pip.log 2>&1 && echo "  ok      $p" || echo "  FAILED  $p ($(tail -1 /tmp/pip.log | cut -c1-100))"; done
pip install --no-deps "git+https://github.com/facebookresearch/vggt" > /tmp/pip_vggt.log 2>&1 && echo "  ok      vggt (no-deps)" || { echo "  FAILED  vggt"; tail -2 /tmp/pip_vggt.log; }
python - <<'PY'
import importlib
for m in ["torch", "torchvision", "transformers", "cv2", "vggt", "jsonschema", "jsonpatch", "openai"]:
    try: mod = importlib.import_module(m); print("  import ok", m, getattr(mod, "__version__", ""))
    except Exception as e: print("  import FAILED", m, repr(e)[:120])
from torchvision.ops import nms; print("  torchvision ops ok")
PY
pip freeze --local > $REPO/requirements-vulcan.lock.txt   # requirements.txt 是手写的通用清单，别覆盖
deactivate

echo "=== weights -> $GWM_WEIGHTS ==="
source "$GWM_VENV/bin/activate"
mkdir -p "$GWM_WEIGHTS"
dl() { repo=$1; name=$2; shift 2; [ -f "$GWM_WEIGHTS/$name/.complete" ] && { echo "  skip    $name"; return; }
  hf download "$repo" --local-dir "$GWM_WEIGHTS/$name" "$@" > /tmp/hf.log 2>&1 && { touch "$GWM_WEIGHTS/$name/.complete"; echo "  ok      $name $(du -sh $GWM_WEIGHTS/$name | cut -f1)"; } || { echo "  FAILED  $name"; tail -2 /tmp/hf.log; }; }
dl facebook/dinov2-base dinov2-base
dl depth-anything/Depth-Anything-V2-Small-hf depth-anything-v2-small
dl IDEA-Research/grounding-dino-tiny grounding-dino-tiny
dl facebook/sam2.1-hiera-large sam2.1-hiera-large
dl facebook/VGGT-1B vggt-1b --include "*.safetensors" "*.json" "*.md"
# VGGT 的 5 GB 权重走 cdn.hf.co，计算节点代理会拦；若上面没下到 model.safetensors，在登录节点执行：
#   curl -L -o $GWM_WEIGHTS/vggt-1b/model.safetensors https://huggingface.co/facebook/VGGT-1B/resolve/main/model.safetensors
[ -f "$GWM_WEIGHTS/vggt-1b/model.safetensors" ] || echo "  NOTE    vggt-1b/model.safetensors missing: download it on the login node (see comment above)"
deactivate

if [ "${1:-}" = "--with-vllm" ]; then
  echo "=== vllm venv ($GWM_PROJECT/venv-vllm) ==="
  VV=$GWM_PROJECT/venv-vllm; [ -f $VV/bin/activate ] || python -m venv $VV
  source $VV/bin/activate; unset PIP_PREFIX
  pip install --no-index --upgrade pip 2>&1 | tail -1
  pip install --no-index "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" 2>&1 | tail -1
  pip install --no-index --no-deps vllm 2>&1 | tail -1
  python - <<'PY' > /tmp/vllm_deps.txt
import re, zipfile, glob
w = glob.glob('/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/*/*/vllm-0.25.0*.whl')[0]; z = zipfile.ZipFile(w)
meta = z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode().splitlines()
for l in meta:
    if l.startswith('Requires-Dist:') and 'extra ==' not in l:
        req = l[len('Requires-Dist: '):].split(';')[0].strip(); name = re.split(r'[<>=!~\[ ]', req)[0].lower()
        if not name.startswith(('opencv', 'torch', 'torchvision', 'torchaudio')): print(req)
PY
  while read -r req; do [ -z "$req" ] && continue; pip install --no-index "$req" > /tmp/dep.log 2>&1 || pip install "$req" > /tmp/dep.log 2>&1 || echo "  FAILED  $req"; done < /tmp/vllm_deps.txt
  python -c "import vllm, torch; print('  vllm', vllm.__version__, 'torch', torch.__version__)"
  pip freeze --local > $REPO/requirements-vllm.lock.txt
  hf download Qwen/Qwen3.5-27B-FP8 --local-dir "$GWM_WEIGHTS/qwen3.5-27b-fp8" > /tmp/hf_qwen.log 2>&1 && echo "  ok      qwen3.5-27b-fp8" || echo "  FAILED  qwen3.5-27b-fp8"
  deactivate
fi
echo "=== DONE $(date) ==="
