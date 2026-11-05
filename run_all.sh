#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-runs}"
common=(--epochs "${EPOCHS:-15}" --results-dir "$RUN_DIR/results" --models-dir "$RUN_DIR/models")
"$PYTHON" src/train.py --name baseline "${common[@]}"
"$PYTHON" src/train.py --name aug --augment "${common[@]}"
"$PYTHON" src/train.py --name aug_bn_drop --augment --bn --dropout 0.3 "${common[@]}"
"$PYTHON" src/evaluate.py --results-dir "$RUN_DIR/results" --models-dir "$RUN_DIR/models"
