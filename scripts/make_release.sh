#!/bin/bash
#SBATCH --job-name=gwm-release
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
# 把跑出来的游戏和视频打包成可以发给别人的 zip。
# 用法: sbatch scripts/make_release.sh  （在 scripts/release_runs.txt 里写「片段名 运行目录」，一行一个）
set -eu
REPO=/scratch/jwj/research/GameWorldModel
source $REPO/scripts/setup_env.sh
export GWM_NODE_ROOT=$SLURM_TMPDIR/node
mkdir -p $GWM_NODE_ROOT && tar -xf $GWM_DEPS/node-playwright-three.tar -C $GWM_NODE_ROOT
export GWM_NODE_MODULES=$GWM_NODE_ROOT/node_modules PLAYWRIGHT_BROWSERS_PATH=$GWM_NODE_ROOT/pw-browsers
cd $REPO
OUT=${GWM_SCRATCH}/release; rm -rf $OUT; mkdir -p $OUT
H=$REPO/scripts/node_harness.sh

while read -r CLIP RUN TITLE <&3; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  echo "=== $CLIP from $RUN ==="
  PKG=$OUT/gwm-${CLIP//_/-}; mkdir -p $PKG
  # 1. 重新编译游戏（新的打包只带真正用到的依赖）
  python -m gwm.compiler.compile $RUN/program.json $PKG/game >/dev/null
  cp $RUN/program.json $RUN/report.md $PKG/
  du -sh $PKG/game
  # 2. 按视频原来的相机路径回放，得到可以和原视频并排看的画面
  DUR=$(python -c "import json;print(round(json.load(open('$RUN/program.json'))['meta']['duration'],2))")
  $H record.mjs --game $PKG/game --out $SLURM_TMPDIR/rec_$CLIP --fps 24 --duration $DUR --width 640 --height 360 >/dev/null
  ffmpeg -nostdin -loglevel error -y -framerate 24 -i $SLURM_TMPDIR/rec_$CLIP/frames/f_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 $PKG/replay.mp4
  # 3. 自动试玩，录成视频
  $H playtest.mjs --game $PKG/game --out $SLURM_TMPDIR/play_$CLIP --seconds 60 --shot-every 0.0667 >/dev/null || true
  python - "$SLURM_TMPDIR/play_$CLIP/frames" "$SLURM_TMPDIR/seq_$CLIP" <<'PY'
import re, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("rgb_*.png"), key=lambda p: float(re.search(r"rgb_([\d.]+)\.png", p.name).group(1)))
for i, f in enumerate(files): shutil.copy2(f, dst / f"f_{i:05d}.png")
print(f"  {len(files)} playthrough frames")
PY
  ffmpeg -nostdin -loglevel error -y -framerate 15 -i $SLURM_TMPDIR/seq_$CLIP/f_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 $PKG/gameplay.mp4
  # 4. 原视频、感知叠加视频
  cp data/clips/trimmed/$CLIP.mp4 $PKG/source.mp4
  [ -f $RUN/perception/overlay.mp4 ] && cp $RUN/perception/overlay.mp4 $PKG/overlay.mp4 || true
  # 5. 并排对比：左边原视频，右边同一相机下的游戏画面
  ffmpeg -nostdin -loglevel error -y -i $PKG/source.mp4 -i $PKG/replay.mp4 -filter_complex \
    "[0:v]scale=640:360,setsar=1[l];[1:v]scale=640:360,setsar=1[r];[l][r]hstack=inputs=2" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 -an $PKG/side-by-side.mp4
  # 6. 说明文件、启动脚本，打成 zip
  cp $REPO/scripts/release_files/serve.py $PKG/
  python $REPO/scripts/release_files/make_package_docs.py $PKG $CLIP "$TITLE"
  (cd $OUT && zip -qr "$(basename $PKG).zip" "$(basename $PKG)")
  echo "  packaged $(du -h $OUT/$(basename $PKG).zip | cut -f1)"
done 3< $REPO/scripts/release_runs.txt

# 把每段的并排对比接成一个总的 demo
echo "=== demo.mp4 ==="
: > $SLURM_TMPDIR/concat.txt
for f in $OUT/*/side-by-side.mp4; do echo "file '$f'" >> $SLURM_TMPDIR/concat.txt; done
ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i $SLURM_TMPDIR/concat.txt -c:v libx264 -pix_fmt yuv420p -crf 20 $OUT/demo.mp4
ls -la $OUT/demo.mp4

echo "=== DONE, packages in $OUT ==="
ls -la $OUT
