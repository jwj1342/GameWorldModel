#!/bin/bash
#SBATCH --job-name=gwm-dc-compare
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# issue #11：把模型直接写的代码渲染出来，和 DSL 路径的产物做同口径比较
set -u
GAME=${1:?需要直接出代码的 game 目录}
DSLRUN=${2:?需要 DSL 路径的运行目录}
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
cd $REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
OUT=${GWM_SCRATCH}/dc_compare/$SLURM_JOB_ID; mkdir -p $OUT
H=$REPO/scripts/node_harness.sh
echo "=== $(date) on $(hostname) ==="

echo "=== 直接出代码：能不能启动、能不能渲染 ==="
if $H render.mjs --game "$GAME" --times 0,2,4,6 --out $OUT/dc --width 480 --height 270 2>$OUT/dc_err.txt; then
  echo "  渲染成功"; ls $OUT/dc | head -4
else
  echo "  渲染失败："; tail -12 $OUT/dc_err.txt; tail -12 "$GAME/../browser.log" 2>/dev/null
fi

echo "=== 自动试玩：生成的场景能不能玩 ==="
if $H playtest.mjs --game "$GAME" --out $OUT/dc_play --seconds 60 2>$OUT/dc_play_err.txt; then
  cat $OUT/dc_play/verdict.json
else
  echo "  试玩失败："; tail -12 $OUT/dc_play_err.txt
fi

echo "=== DSL 路径同口径（同一段视频的产物）==="
cat $DSLRUN/playtest/verdict.json 2>/dev/null || cat $DSLRUN/verdict.json 2>/dev/null || echo "  找不到 DSL 的试玩结果"
echo "=== DONE $(date) ==="
