#!/usr/bin/env bash
# Exploratory short/long ratio analysis (pre-registered as a note after Amendment 3 in results/phase56_prereg.md):
# short-term share 0.65 and 0.85 at 200 context rows, KDD dual memory and regime-balanced dual memory, 4 streams, seeds 0-9.
# The 0.75 column is round 3 (results/round3), the Insects sliding window round 2 (results/round2): run those first.
#
#   bash scripts/reproduce_round4.sh            # all runs on a CUDA GPU, then the report, CSV and figures
#   DEVICE=cpu bash scripts/reproduce_round4.sh # on CPU (very slow)
set -euo pipefail
cd "$(dirname "$0")/.."

NP="${NP:-2}"
PY="${PYTHON:-python3}"
DEV="${DEVICE:-cuda}"
M=("$PY" scripts/run_round2_multiseed.py --device "$DEV" --n_parallel "$NP" --out_dir results/round4 --n_estimators 4 --seeds 0-9)
for ds in gas electricity insects_gradual insects; do
  "${M[@]}" --dataset "$ds" --methods dual200_s65,dual200_s85,dual200_rb_s65,dual200_rb_s85
done
"$PY" scripts/analyze_round4.py      # results/round4_analysis.md and results/round4_per_seed.csv
"$PY" scripts/plot_round4.py         # results/figures_round4/
