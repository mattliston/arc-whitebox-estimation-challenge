#!/bin/bash
# TAG opus-reexam E60 (lead, 2026-10-02): THE C^2 TERM IN ITS NATIVE FORM: compression of C^2 on a K-dimensional subspace from a warm-started subspace iteration (previous block pushed through the gates and W, s sweeps with C): arms k16s1, k16s2, k24s1, k32s1, k8s2 (K, sweeps); all form coefficients from the diagonal regression with a constant and (C^2)_ii. Reference E59: exact C^2 (c2full), exact top-16 eigenpairs (c2k16). Header below is E59 text: FORM COEFFICIENTS FROM THE DIAGONAL REGRESSION WITH A CONSTANT (physicist 13:46Z) AND THE C^2 FORM FROM ITS OWN DIAGONAL REGRESSOR: arms cst1 (constant regressor, multiple of C removed), cst3 (+ the two mean forms), c2full (+ regressor (C^2)_ii and b C^2 removed, exact C^2), c2k16 / c2k8 / c2k4 (C^2 from its top 16 / 8 / 4 eigenpairs). Header below is E57 text: SPECTRAL REPAIR of the fold of the pair-carried law (O/ncar6.py, level 'read'). E56 spectrum (e56_spectrum.py): in the eigenbasis of the
# covariance the carried matrix is 80% diagonal (a function of C), and the multiple of C it carries falls from the middle of the spectrum to the top modes; the exact four-form
# fit follows that with a + b lambda. Native candidate: remove the BULK multiple (x the diagonal regression's coefficient) and then only the Rayleigh quotients of what is
# left on the top-K modes (one pass over the columns per mode), keeping the cross terms between modes.
# Arms: d95q4 / q8 / q16 / q32 = bulk factor 0.95, top-K quotients out;  d85q8, d105q8 = bulk factor 0.85 / 1.05 at K 8.   (E51 references: readf4, readf1, readtab; E56: d95, d95p8.)
# usage: e60_c2sub.sh <rank> ...  (one torch process of 4 threads; idle gate 30%; no quiet window)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
quiet() { while window_open || ! $O/idle_gate.sh 30 > /dev/null; do sleep 20; done; }
cd $O; export OMP_NUM_THREADS=4
B="f4=0.6,age=2,c=0.75"; N6="nc=1.0,nclaw=6,nclo=1,nca=0.75,ncar=0,nclev=1"
for r in "$@"; do
  R4=$(printf %04d $r); [ -s e60_c2sub/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e60 S37:$B k16s1:$B,$N6,ncbf=3,ncdf=4,ncc2=16,ncc2s=1 k16s2:$B,$N6,ncbf=3,ncdf=4,ncc2=16,ncc2s=2 k24s1:$B,$N6,ncbf=3,ncdf=4,ncc2=24,ncc2s=1 k32s1:$B,$N6,ncbf=3,ncdf=4,ncc2=32,ncc2s=1 k8s2:$B,$N6,ncbf=3,ncdf=4,ncc2=8,ncc2s=2 > e60_c2sub/rank$R4.tmp 2> e60_c2sub/rank$R4.err && mv e60_c2sub/rank$R4.tmp e60_c2sub/rank$R4.log
done
echo ALLDONE > e60_c2sub/done
