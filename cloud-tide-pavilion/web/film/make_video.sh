#!/bin/bash
# 帧序列 + 配乐 → MP4（H.264 / AAC，可在 PPT、手机与网页里直接播放）
#   web/film/make_video.sh 帧目录 输出前缀 [python]
# 生成 前缀_1080p.mp4（高画质，汇报用）与 前缀_720p.mp4（体积小，方便微信等发送）
set -e
FRAMES=$1; OUT=$2; PY=${3:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
N=$(ls "$FRAMES"/f*.jpg | wc -l)
DUR=$(awk "BEGIN { printf \"%.3f\", $N / 30 }")
"$PY" "$HERE/audio.py" "$FRAMES/audio.wav" "$DUR"
ffmpeg -y -hide_banner -loglevel error -framerate 30 -i "$FRAMES/f%05d.jpg" -i "$FRAMES/audio.wav" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -profile:v high -level 4.1 \
  -c:a aac -b:a 192k -movflags +faststart -shortest \
  -metadata title="云上潮汐馆 · 导览影片" "${OUT}_1080p.mp4"
ffmpeg -y -hide_banner -loglevel error -i "${OUT}_1080p.mp4" -vf scale=1280:720:flags=lanczos \
  -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -profile:v high -level 4.0 \
  -c:a aac -b:a 128k -movflags +faststart "${OUT}_720p.mp4"
ls -la "${OUT}_1080p.mp4" "${OUT}_720p.mp4"
