#!/bin/bash
#SBATCH --job-name=gwm-direct-code
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 用手写的样例验证直接出代码这条路径的宿主能跑通，不花模型调用。
set -u
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
export GWM_NODE_ROOT=$SLURM_TMPDIR/node; mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
cd $REPO
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
OUT=${GWM_SCRATCH}/direct_code/$SLURM_JOB_ID; mkdir -p $OUT
echo "=== $(date) on $(hostname) ==="

python - "$OUT" <<'PY'
import json, sys, pathlib
sys.path.insert(0, "/scratch/jwj/research/GameWorldModel/scripts")
from run_direct_code import build_game_dir
out = pathlib.Path(sys.argv[1])
code = pathlib.Path("examples/direct_code/model_scene.js").read_text()
stub = {"meta": {"clip": "direct_code_demo", "fps": 30, "duration": 8.0, "units": "m", "up": "y"},
        "style": {"background": "#dddde3"},
        "camera": {"intrinsics": {"fov_deg": 50, "aspect": 1.7778, "far": 80},
                   "keyframes": [{"t": 0.0, "pos": [0, 3, 10], "quat": [0, 0, 0, 1]}], "interp": "linear"},
        "static": [], "objects": [],
        "binding": {"template": "platformer_3p", "slots": {"walkable": "auto"}}}
build_game_dir(code, stub, out / "game")
print("  打包好了:", out / "game")
PY

H=$REPO/scripts/node_harness.sh
echo "=== 渲染三个通道 ==="
$H render.mjs --game $OUT/game --times 0,2,4,6 --out $OUT/render --width 480 --height 270 \
  || { echo RENDER FAILED; tail -40 $OUT/render/browser.log 2>/dev/null; exit 1; }
ls $OUT/render | head -8
echo "=== 录状态，确认 pose(t) 真的在动 ==="
$H record.mjs --game $OUT/game --out $OUT/rec --fps 10 --duration 8 --width 320 --height 180 \
  || { echo RECORD FAILED; tail -30 $OUT/rec/browser.log 2>/dev/null; exit 1; }
python - "$OUT" <<'PY'
import json, sys, pathlib
st = [json.loads(l) for l in (pathlib.Path(sys.argv[1]) / "rec" / "gt_states.jsonl").read_text().splitlines() if l.strip()]
ys = [next(o for o in s["objects"] if o["id"] == "lift")["pos"][1] for s in st]
print(f"  升降平台 y 范围 {min(ys):.2f} .. {max(ys):.2f}（手写的是中心 1.2、幅度 1.2，所以应当约 0.0 .. 2.4）")
print("  宿主可用" if max(ys) - min(ys) > 1.5 else "  宿主有问题：平台没动起来")
PY
echo "=== DONE $(date) ==="
