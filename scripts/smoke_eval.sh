#!/bin/bash
#SBATCH --job-name=gwm-eval
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 评测入口的端到端验证：给一批运行目录，出一张表
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
cd $REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
echo "=== $(date) on $(hostname) ==="
python scripts/eval.py --work $SLURM_TMPDIR/ev "$@"
echo "=== DONE $(date) ==="
