#!/bin/bash
# [opus-s3pb] S3's own-census prebind refit: census (quick_s3_c*.json) -> prebind_targeted_v2 (budgets from S3's release stage C manifests)
# + k-split refit (_KS_ACC_MN: (M, N) used by >= 2 census networks, most used first) -> cand_s3pb_v1.py = S3 6921feaa + both fragments.
set -u
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1; W=$O/lead/runs/opus_s3pb_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; NPY=$O/../astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python
C=$W/cand_s3_v1.py; R=$O/lead/runs/opus_release_v1/rel_stks3_20261016_opus_v1/boundary
Q="$(ls $W/quick_s3_c*.json | grep -v _guard | tr "\n" " ")"
BN=$($PY -c "import json;print(json.load(open('$R/normal/manifest.json'))['setup']['binding_requests']-573)"); BS=$($PY -c "import json;print(json.load(open('$R/slow/manifest.json'))['setup']['binding_requests']-573)")
echo "budgets normal $BN slow $BS; census: $Q"
cd $O && BUDGET_NORMAL=$BN BUDGET_SLOW=$BS nice -n 10 $NPY -B $O/engineer/a62_modules/prebind_targeted_v2.py $C $R/normal/manifest.json $W/fragment_prebind_s3_v1.py.txt $Q 2>&1 | grep -E "^(OLD|NEW|SWAPS|old list|binding cost|written)"
$PY - <<PYEOF
import json, collections
c = collections.Counter()
for f in "$Q".split():
    d = json.load(open(f))
    for r in (d['rows'] if isinstance(d, dict) else d): c.update({(k[2], k[3]) for k in r['key_uses'] if k[0] == 3})
keep = sorted((k for k in c if c[k] >= 2), key=lambda k: (-c[k], k))
open('$W/fragment_ks_acc_s3_v1.py.txt', 'w').write('\n\n# BEGIN OPUS-S3PB FRAGMENT ks_acc_refit_s3_v1 (2026-10-16; append LAST): k-split leaf-buffer views setup binds, refit to S3\'s own\n# census (20 public ranks, 35/100 excluded): every (M, N) used by >= 2 networks, most used first. Exact.\n_KS_ACC_MN = ' + repr(tuple(keep)) + '\n# END OPUS-S3PB FRAGMENT ks_acc_refit_s3_v1\n')
print('ks_acc keys', len(keep))
PYEOF
cat $C $W/fragment_prebind_s3_v1.py.txt $W/fragment_ks_acc_s3_v1.py.txt > $W/cand_s3pb_v1.py
$PY -c "compile(open('$W/cand_s3pb_v1.py').read(), 'c', 'exec')" && echo "cand_s3pb_v1 $(shasum -a 256 $W/cand_s3pb_v1.py | cut -c1-8)"
