#!/usr/bin/env bash
# Reproduce every number and figure in the README. About 8 minutes on two CPU cores.
set -euo pipefail
python src/download_data.py
python src/prepare_beats.py --mitdb data/mitdb --out data/beats.npz
python src/baseline.py
python src/train_cnn.py --split inter --stage validate --seed 0
for seed in 0 1 2; do python src/train_cnn.py --split inter --stage final --seed "$seed"; done
python src/train_cnn.py --split inter --stage final --seed 0 --no-template
python src/train_cnn.py --split inter --stage final --seed 0 --no-template --no-rr
python src/train_cnn.py --split intra --seed 0
python src/figures.py
