#!/bin/bash
# TAG opus_engineer2 a62 smoke_pe34k21k_v1 (file written 2026-10-13T00:18:44Z): the NON-EMPTY paths of the two S54 routes before the trainer's arms land:
# s49/pe34smk_s51k1twx (s51k1tw_x fb4456a7 + make_pe34_v1 with e3 and e4 doubled at layer 12) and s49/k21ksmk_s51k1twx (+ make_k21k_v1 with k21k = 0.5
# at layer 12), each on 7100264 with gate_rdz_v1 (the transfer test's harness), against the transfer test's own run of s51k1tw_x there
# (xfer_v1/gate_rdz_xfer_K_7100264): a crash shows here, not in a parity run; and the layer pattern checks the layer keying: a covariance change at the
# pair map / GN correction of layer 12 must leave the predictions of layers 0-12 bit for bit and move them from layer 13 on. Waits for the transfer test,
# idc1 and the no-op certificates (one native process); one [engineer2] note with the per-layer pattern.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; O=$P2/opus_reexam_20261001_v1; A=$O/engineer/a62_modules
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; NPY=$P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
NATIVE="^($P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B .*/engineer/|$P2/data_gate/.client-venv/bin/python -B -u .*boundary_client[.]py .*/engineer/)"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
say "armed: waiting for the transfer test, idc1 and the no-op certificates"
while pgrep -f 'xfer_run_v[0-9]+[.]sh|after_xfer_cert_idc_v[0-9]+[.]sh|cert_cut_v[0-9]+[.]sh|noop_pe34k21k_v[0-9]+[.]sh' > /dev/null; do sleep 60; done
for t in pe34smk_s51k1twx k21ksmk_s51k1twx; do
  [ -e $A/gate_rdz_${t}_264.json ] && continue
  until [ $(ps -axo command= | grep -c -E "$NATIVE") -lt 2 ]; do sleep 30; done
  bash $O/gate.sh 7 engineer2 > /dev/null
  say "smoke $t on 7100264"
  ( cd $P2 && $PY -B $O/release/guard_footprint_v3.py --record $A/gate_rdz_${t}_264_guard.json --seconds 7200 --rss-backstop 7500000000 \
    $NPY -B $A/gate_rdz_v1.py $A/s49/$t/candidate.py none none 7100264 $A/gate_rdz_${t}_264.json > $A/gate_rdz_${t}_264.log 2>&1 )
done
res=$($PY -c "
import json, sys
import numpy as np
A = sys.argv[1]
base = np.load(A + '/xfer_v1/gate_rdz_xfer_K_7100264_pred.npz')['7100264'].astype(float)
mb = json.load(open(A + '/xfer_v1/gate_rdz_xfer_K_7100264.json'))['rows'][0]['mse_final']
out = []
for t, what in (('pe34smk_s51k1twx', 'pe34 e3 / e4 x2 at L12'), ('k21ksmk_s51k1twx', 'k21k +0.5 at L12')):
    try:
        p = np.load('%s/gate_rdz_%s_264_pred.npz' % (A, t))['7100264'].astype(float)
        r = json.load(open('%s/gate_rdz_%s_264.json' % (A, t)))['rows'][0]
    except Exception as e:
        out.append('%s: NO RESULT (crash? a62_modules/gate_rdz_%s_264.log)' % (what, t)); continue
    same = [l for l in range(16) if np.array_equal(p[l], base[l])]
    first = min([l for l in range(16) if l not in same], default=None)
    rel = ' '.join('L%d %.1e' % (l, np.linalg.norm(p[l] - base[l]) / np.linalg.norm(base[l])) for l in range(13, 16))
    out.append('%s: layers bit for bit %s, first changed layer %s, relative change %s, final MSE %+.3f%%' % (
        what, ('0-%d' % max(same)) if same == list(range(max(same) + 1)) else same, first, rel, 100 * (r['mse_final'] / mb - 1)))
print('; '.join(out))" $A)
say "$res"
echo "[engineer2] $(date -u +%FT%TZ) S54 ROUTES SMOKE (the non-empty paths on the fired twin s51k1tw_x, 7100264, vs the transfer test's own run there; smoke_pe34k21k_v1.sh): $res. Expected: layers 0-12 bit for bit (both edits move only the off-diagonal output covariance of layer 12), change from 13." >> $O/TEAM_LOG.md
say "done"
