#!/bin/bash
# TAG opus-reexam E57 (lead, 2026-10-02): SPECTRAL REPAIR of the fold of the pair-carried law (O/ncar6.py, level 'read'). E56 spectrum (e56_spectrum.py): in the eigenbasis of the
# covariance the carried matrix is 80% diagonal (a function of C), and the multiple of C it carries falls from the middle of the spectrum to the top modes; the exact four-form
# fit follows that with a + b lambda. Native candidate: remove the BULK multiple (x the diagonal regression's coefficient) and then only the Rayleigh quotients of what is
# left on the top-K modes (one pass over the columns per mode), keeping the cross terms between modes.
# Arms: d95q4 / q8 / q16 / q32 = bulk factor 0.95, top-K quotients out;  d85q8, d105q8 = bulk factor 0.85 / 1.05 at K 8.   (E51 references: readf4, readf1, readtab; E56: d95, d95p8.)
# usage: e57_pdiag.sh <rank> ...  (one torch process of 4 threads; idle gate 30%; no quiet window)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
quiet() { while window_open || ! $O/idle_gate.sh 30 > /dev/null; do sleep 20; done; }
cd $O; export OMP_NUM_THREADS=4
B="f4=0.6,age=2,c=0.75"; N6="nc=1.0,nclaw=6,nclo=1,nca=0.75,ncar=0,nclev=1,ncbf=1,ncdf=1"
for r in "$@"; do
  R4=$(printf %04d $r); [ -s e57_pdiag/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e57 S37:$B d95q4:$B,$N6,ncds=0.95,ncpd=4 d95q8:$B,$N6,ncds=0.95,ncpd=8 d95q16:$B,$N6,ncds=0.95,ncpd=16 d95q32:$B,$N6,ncds=0.95,ncpd=32 d85q8:$B,$N6,ncds=0.85,ncpd=8 d105q8:$B,$N6,ncds=1.05,ncpd=8 > e57_pdiag/rank$R4.tmp 2> e57_pdiag/rank$R4.err && mv e57_pdiag/rank$R4.tmp e57_pdiag/rank$R4.log
done
echo ALLDONE > e57_pdiag/done
