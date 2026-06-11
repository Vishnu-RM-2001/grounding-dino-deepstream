#!/usr/bin/env bash
# Download a Grounding-DINO demo clip into data/sample.mp4. It's the short (~14 s)
# overhead-concourse "people-walking" clip used throughout GroundingDINO / supervision
# demos — a crowd of pedestrians with bags, great for open-vocabulary prompts.
# Source: Roboflow supervision video-examples (people-walking.mp4).
# The file lands in data/ (gitignored) — kept locally, re-fetched by this script.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
URL=${GDINO_SAMPLE_URL:-https://media.roboflow.com/supervision/video-examples/people-walking.mp4}
if [ ! -s data/sample.mp4 ]; then
  echo "downloading sample clip..."
  curl -L -o data/sample.mp4 "$URL"
fi
ls -lh data/sample.mp4
echo
echo "run:  ./scripts/run.sh \"person . backpack . handbag .\""
