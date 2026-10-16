#!/usr/bin/env bash
# One-command pipeline: WorldCover labels -> models -> risk map.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.data build
python3 -m src.data build-patches
python3 -m src.train --model all --device auto --folds 5 --epochs 200
python3 -m src.mapping --model cnn --aoi default
echo "RUN_ALL OK — see runs/metrics.json and runs/susceptibility_map_cnn_default.png"
