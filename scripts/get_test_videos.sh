#!/usr/bin/env bash
# Copy a sample video out of the DeepStream image into data/ (no external
# download — these ship inside the container). sample_720p.mp4 is a street scene
# with cars and people; run.sh also uses it by default.
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}
mkdir -p data
docker run --rm -v "$PWD/data":/out "$IMAGE" \
  bash -lc 'cp /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 /out/'
echo "done:"; ls -lh data/*.mp4
echo
echo "run:  ./scripts/run.sh \"car, man\" file:///work/data/sample_720p.mp4 out/sample.mp4"
echo "(other clips live in /opt/nvidia/deepstream/deepstream/samples/streams/ inside the image)"
