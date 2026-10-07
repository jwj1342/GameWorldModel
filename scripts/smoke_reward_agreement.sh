#!/bin/bash
#SBATCH --job-name=gwm-reward-agree
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# issue #17：无真值的奖励信号和有真值的指标，排序一不一致
set -u
RUN=${1:?需要一次完整运行的目录}
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
cd $REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
echo "=== $(date) on $(hostname) ==="
python scripts/check_reward_agreement.py --run "$RUN" --work $SLURM_TMPDIR/ra
echo "=== DONE $(date) ==="
