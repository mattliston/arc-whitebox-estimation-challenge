#!/bin/zsh
# TAG fable-covhunter (CH-70): ALTERNATING LOOP ROUND 3 on the S23 STRUCTURE (lead's option (1)), automatic, under the machine rule.
# (1) wait for the physicist's loopR2 (s6b_loopR2_fckpt0_psloop_s1zero: loop3 card = s1=0 star, round-1 dials) VALID16 + params; stop unless
# VALID16 <= 0.995 x its valid_before. (2) machine gate, then the learned-ON joint GN dial fit on 16 TRAIN nets, card = joint2 + nystrom join
# + lrstar s1=0 + gn_linear_hook + fine_ckpt=False + pair_series loop form, loopR2 params, linearized at the round-1 dials. (3) only if the fit's
# VALID16 (round-1 dials -> fit, same card) <= -0.3%: write dials_r3_loopR2.json + s6bcard_jfitR3.py. (4) machine gate, then VALID40 round 3
# (loopR2 + round-3 dials) vs the S23 torch state (loopR1 + round-1 dials), both on the S23-structure card.
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_covhunter_v1; R=$P2/fable_physicist_v1/refit6b
T=s6b_loopR2_fckpt0_psloop_s1zero
cd $H
until grep -q '"valid_epoch0"' $R/$T.log 2>/dev/null && [ -f $R/$T/params_epoch0.pt ]; do sleep 30; done
VB=$(grep '"valid_before"' $R/$T.log | python3 -c "import json,sys; print(json.loads(sys.stdin.readline())['mean'])")
V=$(grep '"valid_epoch0"' $R/$T.log | python3 -c "import json,sys; print(json.loads(sys.stdin.readline())['mean'])")
echo "$(date '+%F %T') loopR2 VALID16 $V (valid_before $VB)"
if python3 -c "import sys; sys.exit(0 if float('$V') <= 0.995 * float('$VB') else 1)"; then echo "loopR2 passes its bar"; else echo "STOP: loopR2 failed its bar"; exit 3; fi
sleep 30
./machine_gate.sh 6
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
PARAMS=$R/$T/params_epoch0.pt STAR=s1zero JOIN=nystrom GL=1 PS=1 BASE_DIALS=$H/dials_r1_ep2.json nice -n 10 python3 gn_joint_fit.py 16 3 gnjoint_r3_loopR2_on_shared_s23 on shared > gnjoint_r3_loopR2_on_shared_s23.log 2>&1
echo "$(date '+%F %T') fit exit $?"
python3 - <<'PY' || exit 4
import json
from pathlib import Path
H = Path('.'); d = json.load(open('gnjoint_r3_loopR2_on_shared_s23.json')); v = d['valid16']
print('fit VALID16 (round-1 dials -> fit, loopR2 params, S23-structure card): %+.3f%%, %d/16' % (v['pooled'] * 100, v['wins']))
if v['pooled'] > -0.003: raise SystemExit('STOP: round-3 fit VALID16 not <= -0.3%; not writing the round-3 files')
fit = d['chosen']['dials']
g = d['gn']; lo = min(g, key=lambda x: x['loo'])
src = ('"""TAG fable-covhunter (CH-70): ALTERNATING LOOP round 3 dials (S23 structure), FROZEN as literals, for the physicist refit (LOOP_DIALMOD). Semantics:\n'
       's6bcard_joint2 (CH-60, native-consistent). Fit: round3_chain.sh -> gn_joint_fit.py (gnjoint_r3_loopR2_on_shared_s23: learned ON, 16 TRAIN nets, the loopR2\n'
       'params, linearized at the round-1 dials, card = joint2 + nystrom join + lrstar s1=0 + gn_linear_hook + fine_ckpt=False + pair_series loop form):\n'
       f'LOO {lo["loo"] * 100:+.2f}%, VALID16 {v["pooled"] * 100:+.2f}% ({v["wins"]}/16) vs the round-1 dials. STAR lr32 (the loop3/loop4 wrappers set STAR None and install the s1=0 star) installs the S18\n'
       'low-rank star hook after the card; the refit wrapper installs gn_linear_hook / pair_series and the _ckpt re-wrap. Drop-in card module."""\n'
       'import sys\nfrom pathlib import Path\n'
       "sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'fable_engineer_v1'))\n"
       'import s6bcard_joint2 as SJ\n'
       'FIT = ' + '{' + ', '.join(f'{k!r}: {v!r}' for k, v in fit.items()) + '}\n'
       "STAR = 'lr32'\n"
       'model, weights, truthpost, payload, PARAMS = SJ.model, SJ.weights, SJ.truthpost, SJ.payload, SJ.PARAMS\n\n\n'
       'def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):\n'
       '    SJ.DIALS = dict(FIT)\n'
       '    cd = SJ.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, **kw)\n'
       "    if STAR == 'lr32':\n"
       '        import lrstar_torch_hook as lh\n'
       '        lh.install(cd, torch)\n'
       '    return cd\n')
assert not (H / 's6bcard_jfitR3.py').exists() and not (H / 'dials_r3_loopR2.json').exists()
(H / 's6bcard_jfitR3.py').write_text(src); json.dump(fit, open(H / 'dials_r3_loopR2.json', 'x'), indent=1); print('wrote dials_r3_loopR2.json + s6bcard_jfitR3.py', fit)
PY
python3 -m py_compile s6bcard_jfitR3.py || exit 5
./machine_gate.sh 6
STAR=s1zero JOIN=nystrom GL=1 PS=1 nice -n 10 python3 valid40_dials.py 3 $R/s6b_loopR1/params_epoch0.pt $H/dials_r1_ep2.json $R/$T/params_epoch0.pt $H/dials_r3_loopR2.json r3_loopR2dials_vs_S23 > r3_loopR2dials_vs_S23.log 2>&1
echo "$(date '+%F %T') valid40 exit $?"; grep summary r3_loopR2dials_vs_S23.log | cut -c1-1200
