#!/usr/bin/env bash
# Regenerates the headline comparison from committed configs and seeds.
set -euo pipefail

python train/grpo_train.py --config configs/baseline.yaml   --seed 1
python train/grpo_train.py --config configs/structural.yaml --seed 1
python train/grpo_train.py --config configs/ablation.yaml   --seed 1

python eval/bbh_eval.py --runs results/e0_baseline results/e1_structural results/e2_ablation \
                        --splits docs/splits.json --out results/headline.csv
echo "Headline table written to results/headline.csv"
