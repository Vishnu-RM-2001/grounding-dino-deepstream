#!/usr/bin/env bash
# Download the two demo clips into data/ (gitignored, re-fetched any time):
#   data/dog_park.mp4  — a dog and its owner in a park   (prompt: "dog . person .")
#   data/tomatoes.mp4  — hands washing tomatoes in a bowl (prompt: "tomato . hand .")
# Both are short Pexels clips with distinct, well-separated objects.
set -e
cd "$(dirname "$0")/.."
mkdir -p data

fetch() {  # name  url
  local out="data/$1"
  [ -s "$out" ] && { echo "have $out"; return; }
  echo "downloading $out ..."
  curl -fL -A "Mozilla/5.0" -o "$out" "$2"
}

fetch dog_park.mp4 "${GDINO_DOG_URL:-https://www.pexels.com/download/video/3191251/}"
fetch tomatoes.mp4 "${GDINO_TOMATO_URL:-https://www.pexels.com/download/video/5945027/}"

ls -lh data/dog_park.mp4 data/tomatoes.mp4
echo
echo "run:  ./scripts/run.sh --video file:///workspace/data/dog_park.mp4 \"dog . person .\""
echo "      ./scripts/run.sh --video file:///workspace/data/tomatoes.mp4 \"tomato . hand .\""
