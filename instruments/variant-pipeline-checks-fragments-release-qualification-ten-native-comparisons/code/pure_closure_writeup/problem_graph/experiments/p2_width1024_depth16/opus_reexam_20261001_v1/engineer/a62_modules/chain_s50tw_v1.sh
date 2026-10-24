#!/bin/bash
# TAG opus_engineer a62 (file written 2026-10-12T04:34Z; lead's conditional GO: only after S50 is ACCEPTED and chain_s50cert_v1.sh has run): S50's
# TWIN to a release-ready aggressive file, the S48 twin's recipe (s48tw_pbt + XSG = twin v2), two native processes at most:
#   1. key census of the twin source s49/jmS50_v1tw (port_twin_v1.sh, 6af01487) on list a (651 378 424 328 287 641 548 557 316 390) and
#      660-669 (run_quick.sh --keys, one process each)
#   2. prebind_targeted_v2.py -> fragment_prebind_s50tw_v1.py (binding costs and budgets from the S48 twin's stage C trace, as for s48tw_pbt)
#   3. s49/jmS50_v1tw_pbt = twin + that list; s49/jmS50_v1tw_x = + XSG (make_xsg_any_v1.py)
#   4. certificate on 7100136: twin source vs jmS50_v1tw_x (predictions must be byte-identical; bill about -0.07%)
#   5. platform-pace counts of jmS50_v1tw_x on 35 / 100, normal and 1.3x slow setup
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
TW=$A/s49/jmS50_v1tw/candidate.py
run() {   # <candidate.py> <out tag> <seeds>
  bash $O/gate.sh 7 engineer || return 3
  cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    /Users/matt/arc-white-box-estimation/.venv/bin/python -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py \
    --record $A/gate_rdz_$2_guard.json --seconds 3600 --rss-limit 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $1 none none $3 $A/gate_rdz_$2.json \
    > $A/gate_rdz_$2.log 2>&1
  echo "$2 exit=$? $(date -u +%H:%M:%SZ)"
}
echo "start $(date -u +%H:%M:%SZ)"
bash $E/run_quick.sh $TW 651,378,424,328,287,641,548,557,316,390 keys_s50tw_a --keys > $A/census_s50tw_a.out 2>&1 &
sleep 5
bash $E/run_quick.sh $TW 660,661,662,663,664,665,666,667,668,669 keys_s50tw_660 --keys > $A/census_s50tw_660.out 2>&1 &
wait
echo "census done $(date -u +%H:%M:%SZ)"
cd $A && $PY -B $A/prebind_targeted_v2.py $TW $O/release/runs/c_s48tw_safe/C0/manifest.json $A/fragment_prebind_s50tw_v1.py \
  $E/logs/quick_keys_s50tw_a.json $E/logs/quick_keys_s50tw_660.json | tee $A/prebind_s50tw_v1.out | tail -12 || exit 4
$PY -B $A/make_s49_v1.py $TW jmS50_v1tw_pbt $A/fragment_prebind_s50tw_v1.py > /dev/null || exit 5
$PY -B $A/make_xsg_any_v1.py $A/s49/jmS50_v1tw_pbt/candidate.py jmS50_v1tw_x | tail -3 || exit 6
X=$A/s49/jmS50_v1tw_x/candidate.py
echo "aggressive file $(shasum -a 256 $X | cut -c1-8) $X; release fragment lines $(grep -c 'BEGIN OPUS-RELEASE FRAGMENT' $X)"
run $TW s50tw_src 7100136 &
sleep 5
run $X s50tw_x 7100136 &
wait
python3 -c "
import json
a = json.load(open('$A/gate_rdz_s50tw_src.json'))['rows'][0]; b = json.load(open('$A/gate_rdz_s50tw_x.json'))['rows'][0]
print('twin source vs twin + prebind + XSG on 7100136: pred_sha', a['pred_sha'], b['pred_sha'], 'BYTE-IDENTICAL' if a['pred_sha'] == b['pred_sha'] else 'DIFFERENT',
      '| bill %+.4f%%' % (100 * (b['flops'] / a['flops'] - 1)), '| ops', a['ops'], b['ops'])"
bash $A/run_boundary_counts_v3.sh $X ps_s50tw_x_n 35,100 &
sleep 5
bash $A/run_boundary_counts_v3.sh $X ps_s50tw_x_s 35,100 "--setup-slowdown 1.3" &
wait
echo "chain done $(date -u +%H:%M:%SZ)"
