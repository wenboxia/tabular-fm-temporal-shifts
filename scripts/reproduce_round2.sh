#!/usr/bin/env bash
# Main experiment (round 2): adaptation vs. forgetting of a frozen TabPFN under 16 context-memory policies.
# Reproduces every table and figure of the main results (results/phase56_round2.md). The pilot runs and the
# per-batch integrity checks described there used separate tooling and are not part of this script.
#
#   bash scripts/reproduce_round2.sh            # full run on a CUDA GPU (~91 GPU-hours on an RTX 2060 6 GB)
#   DEVICE=cpu bash scripts/reproduce_round2.sh # full run on CPU (very slow; only for machines without CUDA)
#   SMOKE=1 bash scripts/reproduce_round2.sh    # ~3-minute CPU check of the pipeline: 3 methods, 1 seed, first 3000 rows
#                                               # (no regime ends within 3000 rows, so retention is reported as nan)
#
# Runs that already have a result file are skipped, so the script can be interrupted and restarted.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/round2
NP="${NP:-2}"                       # runs in parallel
PY="${PYTHON:-python3}"
ALL="sw100,sw200,sw300,sw400,sw600,sw1000,sw1500,sw2000,dual400,dual1000,cbfifo400,arch_union,arch_routed,arch_routed_adwin,dual1000_rb,dual1000_rb_adwin"

if [ "${SMOKE:-0}" = "1" ]; then
  export TABPFN_ALLOW_CPU_LARGE_DATASET=1      # TabPFN refuses >1000-row contexts on CPU unless this is set
  "$PY" scripts/run_round2_multiseed.py --methods sw200,dual1000,dual1000_rb --seeds 0 --n_estimators 1 \
    --device cpu --n_parallel "$NP" --out_dir "$OUT" --max_stream_rows 3000 --tag _smoke
  "$PY" scripts/analyze_round2.py --dir "$OUT" --n_estimators 1 --tag _smoke
  exit 0
fi

DEV="${DEVICE:-cuda}"
[ "$DEV" = cpu ] && export TABPFN_ALLOW_CPU_LARGE_DATASET=1
M=("$PY" scripts/run_round2_multiseed.py --device "$DEV" --n_parallel "$NP" --out_dir "$OUT" --n_estimators 4)
"${M[@]}" --methods "$ALL" --seeds 0-9                                   # main grid: 16 methods x 10 seeds, S = 50
"${M[@]}" --methods sw200,dual1000_rb --seeds 0-2 --stride 10            # sensitivity: batch size S = 10
"$PY" scripts/run_round2_transfer.py --seeds 0-9 --n_estimators 4 --device "$DEV" --out_dir "$OUT"   # 6x6 transfer matrix
"$PY" scripts/analyze_round2.py --dir "$OUT" --n_estimators 4            # tables + pre-registered hypotheses H1-H8
"$PY" scripts/analyze_round2.py --dir "$OUT" --n_estimators 4 --stride 10   # sensitivity check (3 seeds; no formal verdicts)
"$PY" scripts/plot_round2.py --dir "$OUT" --out results/figures_round2 --csv results/round2_per_seed.csv
