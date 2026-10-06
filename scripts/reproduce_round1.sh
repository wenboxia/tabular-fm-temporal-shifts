#!/usr/bin/env bash
# Preliminary study (round 1): three-level system (frozen TabPFN + gated adapter + fast corrector)
# with a river-ADWIN drift detector and four actions on alarm, on drift-aligned Insects segments (seed 42).
# 53 runs (9 TabPFN-alone baselines + 44 three-level runs; the original batch also had one duplicate
# calibration run); ~16 GPU-hours on an RTX 2060. Ends with scripts/analyze_round1.py, which prints the
# numbers quoted in the README (results/round1_summary.md).
#
#   bash scripts/reproduce_round1.sh             # full run (GPU recommended; device is chosen automatically)
#   SMOKE=1 bash scripts/reproduce_round1.sh     # ~7-minute CPU check: 300 steps of the no-drift segment, 1 estimator
#   NP=4 bash scripts/reproduce_round1.sh        # runs in parallel (default 2); PYTHON=... picks the interpreter
#
# Runs that already have a result file are skipped, so the script can be interrupted and restarted.
set -euo pipefail
cd "$(dirname "$0")/.."

SEGS="d3_33240,d4_double,d0_control"     # two drift segments + one no-drift control
SEGS_ORACLE="d3_33240,d4_double"         # the oracle trigger needs a known change point
SEGS_PARITY="d2_19500,d3_33240,d0_control"
XA=""; NP="${NP:-2}"; PY="${PYTHON:-python3}"
if [ "${SMOKE:-0}" = "1" ]; then XA="--max_eval_steps 300 --n_estimators 1"; fi
MS=("$PY" scripts/run_multiseed.py --dataset_source real --datasets insects --aligned_v2 --seeds 42 --n_parallel "$NP")
xa() { local a="$*"; [ -n "$XA" ] && a="${a:+$a }$XA"; printf '%s' "$a"; }

if [ "${SMOKE:-0}" = "1" ]; then
  "${MS[@]}" --label_scheme pair_A_vs_B --segments d0_control --configs phase1 \
    --variant_tag v2AvsB_base_smoke --partial_tag _smoke --extra_args "$(xa)"
  "${MS[@]}" --label_scheme pair_A_vs_B --segments d0_control --configs phase4a --variant_tag b2_detector_pred1_context_reset_smoke \
    --partial_tag _smoke --extra_args "$(xa --detector_impl river --detector_input pred1 --trigger_source detector --action_on_alarm context_reset)"
  "${MS[@]}" --label_scheme pair_A_vs_B --segments d0_control --configs phase4a --variant_tag b2_detector_pred1_none_smoke \
    --partial_tag _smoke --extra_args "$(xa --detector_impl river --detector_input pred1 --trigger_source detector --action_on_alarm none)"
  "$PY" scripts/analyze_round1.py --dir results --suffix _smoke
  exit 0
fi

# 1. Baselines: TabPFN alone (sliding window), two label schemes, and a dual-memory context (short-term ratio 0.5)
"${MS[@]}" --label_scheme pair_A_vs_B --segments "$SEGS" --configs phase1 --variant_tag v2AvsB_base --partial_tag _r1_base --extra_args ""
"${MS[@]}" --label_scheme pair_parity --segments "$SEGS_PARITY" --configs phase1 --variant_tag v2parity_base --partial_tag _r1_parity --extra_args ""
"${MS[@]}" --label_scheme pair_A_vs_B --segments "$SEGS" --configs phase1 --variant_tag v2AvsB_dual --partial_tag _r1_dual \
  --extra_args "--context_loader dual --short_ratio 0.5 --long_max_age 2000"

# 2. Discriminative control: oracle trigger at the official change point vs. the real detector, x 4 actions
for ACT in context_reset route_adapter buffer_clear none; do
  "${MS[@]}" --label_scheme pair_A_vs_B --segments "$SEGS_ORACLE" --configs phase4a --variant_tag "b2_oracle_${ACT}" --partial_tag _r1_b2 \
    --extra_args "--detector_impl river --detector_input pred1 --trigger_source oracle --action_on_alarm ${ACT}"
done

# 3. Detector inputs: prediction-based (pred1), contrast signal (contrast_prob) and 0/1 error indicator, x 4 actions
for INP in pred1 contrast_prob indicator; do
  for ACT in context_reset route_adapter buffer_clear none; do
    "${MS[@]}" --label_scheme pair_A_vs_B --segments "$SEGS" --configs phase4a --variant_tag "b2_detector_${INP}_${ACT}" --partial_tag _r1_b2 \
      --extra_args "--detector_impl river --detector_input ${INP} --trigger_source detector --action_on_alarm ${ACT}"
  done
done

"$PY" scripts/analyze_round1.py --dir results | tee results/round1_summary.md
