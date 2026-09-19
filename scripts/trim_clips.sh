#!/bin/bash
# 裁剪真实视频到 10~60 s、720p、30 fps（在作业内调用；ffmpeg 模块由 setup_env.sh 加载）
set -eu
REPO=/scratch/jwj/research/GameWorldModel; RAW=$REPO/data/clips/raw; OUT=$REPO/data/clips/trimmed; mkdir -p $OUT
trim() { # src dst start duration
  [ -f "$OUT/$2" ] && { echo "skip $2"; return; }
  ffmpeg -loglevel error -y -ss "$3" -i "$RAW/$1" -t "$4" -vf "scale=-2:720,fps=30" -an -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/$2"; echo "wrote $2 ($(du -h $OUT/$2 | cut -f1))"
}
trim plarail_osaka_metro_ccbysa40.webm plarail_train.mp4 6 20
trim conveyor_belt_boxes_gigaset_ccby30.webm conveyor_boxes.mp4 2 16
trim newtons_cradle_5balls_slowmo_cc0.webm newtons_cradle.mp4 0 12
