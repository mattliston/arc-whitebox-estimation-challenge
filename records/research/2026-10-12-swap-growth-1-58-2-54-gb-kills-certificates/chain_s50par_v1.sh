#!/bin/bash
# TAG opus_engineer a62 (file written 2026-10-12T04:59Z; lead: S51 paused, rerun the S50 parity as ONE process): waits until S51's workers are down,
# then ONE native process: S50 (s49/jmS50_v1, 10209ba7) then S48b on 7100521; parity against the trainer's check records
# (a_RS50C_ab / pt, final layer) and, when present, the all-layer pair (a_RS50P_ab / pt). chain_s50cert_v1.sh's runs died to the
# 256 MiB system-swap guard at 04:50Z.
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; A=$O/engineer/a62_modules; R=$O/trainer/run_RAD_S50_gen/results
run() {   # <candidate.py> <out tag> <seeds>
  bash $O/gate.sh 7 engineer || return 3
  cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    /Users/matt/arc-white-box-estimation/.venv/bin/python -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py \
    --record $A/gate_rdz_$2_guard.json --seconds 3600 --rss-limit 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $1 none none $3 $A/gate_rdz_$2.json \
    > $A/gate_rdz_$2.log 2>&1
  echo "$2 exit=$? $(date -u +%H:%M:%SZ)"
}
until ! pgrep -f 'worker_generated_v12.py run_RAD_S51_gen' > /dev/null; do sleep 10; done
echo "S51 workers down $(date -u +%H:%M:%SZ); swap $(sysctl -n vm.swapusage | cut -c1-60)"
run $A/s49/jmS50_v1/candidate.py s50_521 7100521 || exit 1
run $A/s48/s48b/candidate.py s48b_521 7100521 || exit 1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
echo "== against the check records (final layer)"
$PY -B $A/parity_v2.py $R/a_RS50C_ab_r7100521.json $R/a_RS50C_pt_r7100521.json $A/gate_rdz_s48b_521.json $A/gate_rdz_s50_521.json 7100521
if [ -f $R/a_RS50P_ab_r7100521.json ] && [ -f $R/a_RS50P_pt_r7100521.json ]; then
  echo "== against the all-layer pair"
  $PY -B $A/parity_v2.py $R/a_RS50P_ab_r7100521.json $R/a_RS50P_pt_r7100521.json $A/gate_rdz_s48b_521.json $A/gate_rdz_s50_521.json 7100521
else
  echo "== all-layer pair not in results yet (rerun parity_v2.py when it lands; no native run needed)"
fi
echo "chain done $(date -u +%H:%M:%SZ)"
