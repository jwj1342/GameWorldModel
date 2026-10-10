#!/bin/bash
#SBATCH --job-name=gwm-collision-chain
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 验证 DSL 的 dynamic 运动类型：自由刚体与碰撞能不能编译、跑起来、物理行为正确
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
echo "=== $(date) on $(hostname) ==="
OUT=${GWM_SCRATCH}/collision_chain/$SLURM_JOB_ID; mkdir -p $OUT
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
H=$REPO/scripts/node_harness.sh
if ! python -c "import jsonschema" 2>/dev/null; then python -m venv $SLURM_TMPDIR/venv && source $SLURM_TMPDIR/venv/bin/activate && pip install --no-index jsonschema pyyaml numpy >/dev/null; fi
cd $REPO

echo "=== 编译手写的弹性碰撞链场景 ==="
python -m gwm.compiler.compile examples/collision_chain/program.json $OUT/game || exit 1

echo "=== 录制 6 秒，拿到逐帧状态 ==="
$H record.mjs --game $OUT/game --out $OUT/record --fps 30 --duration 6 --width 640 --height 360 \
  || { echo RECORD FAILED; tail -30 $OUT/record/browser.log; exit 1; }

echo "=== 检查物理行为 ==="
python $REPO/scripts/check_collision_chain.py $OUT/record/gt_states.jsonl
echo "=== DONE $(date) === 产物在 $OUT"
