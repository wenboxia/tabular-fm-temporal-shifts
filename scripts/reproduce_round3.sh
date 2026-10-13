#!/usr/bin/env bash
# Round 3: the round-2 comparison on more streams (pre-registered in results/phase56_prereg.md, Amendment 3).
# Streams: Gas Sensor Array Drift (OpenML 1476), Electricity (OpenML 151), the gradual-drift slice of
# Insects gradual_imbalanced (two memory-boundary arms), and two extra 200-row methods on the Insects abrupt stream.
#
#   bash scripts/reproduce_round3.sh            # all runs on a CUDA GPU
#   DEVICE=cpu bash scripts/reproduce_round3.sh # all runs on CPU (very slow; only for machines without CUDA)
#   SMOKE=1 bash scripts/reproduce_round3.sh    # ~3-minute CPU check of the pipeline: Electricity, 3 methods, 1 seed, first 3000 rows
#
# The first run downloads OpenML 151 / 1476 and the Insects files; their content is checked against the
# fingerprints in Amendment 3. Runs that already have a result file are skipped, so the script can be restarted.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/round3
NP="${NP:-2}"                       # runs in parallel
PY="${PYTHON:-python3}"
SIX="sw200,sw1000,dual200,dual1000,dual200_rb,dual1000_rb"

if [ "${SMOKE:-0}" = "1" ]; then
  "$PY" scripts/run_round2_multiseed.py --dataset electricity --methods sw200,dual200,dual200_rb --seeds 0 \
    --n_estimators 1 --device cpu --n_parallel "$NP" --out_dir "$OUT" --max_stream_rows 3000 --tag _smoke
  exit 0
fi

DEV="${DEVICE:-cuda}"
[ "$DEV" = cpu ] && export TABPFN_ALLOW_CPU_LARGE_DATASET=1
M=("$PY" scripts/run_round2_multiseed.py --device "$DEV" --n_parallel "$NP" --out_dir "$OUT" --n_estimators 4 --seeds 0-9)
"${M[@]}" --dataset electricity --methods "$SIX"
"${M[@]}" --dataset gas --methods "$SIX"
"${M[@]}" --dataset insects_gradual --methods "$SIX,dual200_rb_adwin,dual1000_rb_adwin"   # memory closes at the documented change point
"${M[@]}" --dataset insects_gradual_start --methods dual200_rb,dual1000_rb               # memory closes where the mixing starts
"${M[@]}" --dataset insects --methods dual200,dual200_rb                                  # completes the 200-row comparison on round 2's stream
