#!/bin/bash
#SBATCH --job-name=gwm-smoke
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 阶段「手写程序跑通」的冒烟：编译手写程序 → 三 pass 渲染 → 自动试玩 → 录制合成片段
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
echo "=== $(date) on $(hostname) ==="
OUT=${GWM_SCRATCH}/smoke/$SLURM_JOB_ID; mkdir -p $OUT
# node deps -> local disk
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
H=$REPO/scripts/node_harness.sh
# python: project venv if ready, else a throwaway venv with jsonschema
if ! python -c "import jsonschema" 2>/dev/null; then python -m venv $SLURM_TMPDIR/venv && source $SLURM_TMPDIR/venv/bin/activate && pip install --no-index jsonschema pyyaml >/dev/null; fi
cd $REPO
echo "=== compile ==="; python -m gwm.compiler.compile examples/handwritten/program.json $OUT/game || exit 1
ls $OUT/game; du -sh $OUT/game
echo "=== render 3 passes at 4 times ==="; t0=$(date +%s)
$H render.mjs --game $OUT/game --times 0,2.5,5,7.5 --out $OUT/render --width 640 --height 360 || { echo RENDER FAILED; cat $OUT/render/browser.log 2>/dev/null | tail -30; exit 1; }
echo "render took $(( $(date +%s)-t0 ))s"; ls $OUT/render | head -20; tail -5 $OUT/render/browser.log 2>/dev/null
echo "=== 检查 ID pass 是否干净 ==="
python $REPO/scripts/check_id_pass.py $OUT/game $OUT/render || echo "（上面的越界像素会压低所有掩码 IoU，需要修）"

echo "=== playtest ==="; t0=$(date +%s)
$H playtest.mjs --game $OUT/game --out $OUT/playtest --seconds 90 || { echo PLAYTEST FAILED; tail -30 $OUT/playtest/browser.log; }
echo "playtest took $(( $(date +%s)-t0 ))s"; cat $OUT/playtest/verdict.json; tail -3 $OUT/playtest/states.jsonl | cut -c1-200
echo "=== record synthetic clip (8 s @ 30 fps) ==="; t0=$(date +%s)
$H record.mjs --game $OUT/game --out $OUT/record --fps 30 --duration 8 --width 960 --height 540 || { echo RECORD FAILED; tail -30 $OUT/record/browser.log; }
echo "record took $(( $(date +%s)-t0 ))s"
ffmpeg -loglevel error -y -framerate 30 -i $OUT/record/frames/f_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 $OUT/record/handwritten_playground.mp4 && ls -la $OUT/record/handwritten_playground.mp4
mkdir -p $REPO/data/clips/trimmed && cp $OUT/record/handwritten_playground.mp4 $REPO/data/clips/trimmed/ && cp $OUT/record/gt_states.jsonl $REPO/data/clips/trimmed/handwritten_playground.gt.jsonl
echo "=== DONE $(date) === outputs in $OUT"
