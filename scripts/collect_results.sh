#!/bin/bash
# 把一次运行的可读产物（report、程序、关键渲染、试玩帧、叠加视频）复制到 docs/results/<clip>/ 供阅读与入库。
# 用法: scripts/collect_results.sh out/<clip>/<run_id> [目标文件夹名，默认用 clip 名]
set -eu
RUN=$1; CLIP=$(basename $(dirname $RUN)); DST=docs/results/${2:-$CLIP}; rm -rf $DST; mkdir -p $DST
cp $RUN/report.md $DST/report.md
cp $RUN/program.json $DST/program.json
cp $RUN/run.json $DST/run.json 2>/dev/null || true
[ -f $RUN/errors.jsonl ] && cp $RUN/errors.jsonl $DST/ || true
# perception
[ -f $RUN/perception/overlay.mp4 ] && cp $RUN/perception/overlay.mp4 $DST/overlay.mp4 || true
[ -d $RUN/perception/keyframes ] && { mkdir -p $DST/keyframes; cp $RUN/perception/keyframes/*.jpg $DST/keyframes/ 2>/dev/null || true; }
# final feedback renders (rgb + id for the first and last key time)
if [ -d $RUN/feedback ]; then mkdir -p $DST/render; ls $RUN/feedback/rgb_*.png | head -1 | xargs -I{} cp {} $DST/render/; ls $RUN/feedback/rgb_*.png | tail -1 | xargs -I{} cp {} $DST/render/; ls $RUN/feedback/id_*.png | head -1 | xargs -I{} cp {} $DST/render/; ls $RUN/feedback/depth_*.png | head -1 | xargs -I{} cp {} $DST/render/; fi
# playtest
[ -f $RUN/playtest/verdict.json ] && cp $RUN/playtest/verdict.json $DST/ || true
if [ -d $RUN/playtest/frames ]; then mkdir -p $DST/playtest; n=$(ls $RUN/playtest/frames | wc -l); ls $RUN/playtest/frames/*.png | awk -v n=$n 'NR==1||NR==int(n/3)||NR==int(2*n/3)||NR==n' | xargs -I{} cp {} $DST/playtest/; fi
echo "collected -> $DST"; du -sh $DST
