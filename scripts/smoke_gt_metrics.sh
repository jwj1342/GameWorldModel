#!/bin/bash
#SBATCH --job-name=gwm-gt-metrics
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# issue #4 的敏感性检查：把正确程序改坏，确认真值指标确实能测到
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
echo "=== $(date) on $(hostname) ==="
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
cd $REPO
# 追加而不是覆盖：模块系统用 PYTHONPATH 提供 numpy 等包，整个覆盖会把它们弄丢
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
# 依赖直接用 /project 下的项目 venv，setup_env.sh 已经激活；
# 不要再建临时 venv，空 venv 会把项目 venv 盖住
python -c "import numpy, scipy, jsonschema" || { echo "项目 venv 缺依赖"; exit 1; }
python scripts/check_gt_metrics.py --work $SLURM_TMPDIR/gtm
echo "=== DONE $(date) ==="
