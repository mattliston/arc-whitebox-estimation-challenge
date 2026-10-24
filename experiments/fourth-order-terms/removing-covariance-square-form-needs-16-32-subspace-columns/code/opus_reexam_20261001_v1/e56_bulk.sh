#!/bin/bash
# TAG opus-reexam E56 (lead, 2026-10-02): THE BULK MULTIPLE OF C PLUS A TOP-MODE REPAIR in the fold of the pair-carried law (O/ncar6.py, level 'read').
# Reading of E51/E53/E55: the exact four-form fit differs from the one-form fit by its LARGER multiple of C (a4 about 1.5 x a1 at depth; the C^2 term gives the top modes
# back), and a4 is 0.93-0.97 x the DIAGONAL regression's coefficient on 651 and 35 (e45_651/rank0651_e50.err). So natively: remove 0.95 x the diagonal coefficient x C
# (the bulk value) and repair the top modes by taking out the two-sided block of what is left on the top-K covariance eigenvectors.
# Arms: d95 = bulk multiple only (no repair); d95p1 / p2 / p4 / p8 = with the top-K block out.  (E51 reference: readf4 exact four forms, readf1, readtab.)
# usage: e56_bulk.sh <rank> ...  (one torch process of 4 threads; idle gate 30%; no quiet window)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
quiet() { while window_open || ! $O/idle_gate.sh 30 > /dev/null; do sleep 20; done; }
cd $O; export OMP_NUM_THREADS=4
B="f4=0.6,age=2,c=0.75"; N6="nc=1.0,nclaw=6,nclo=1,nca=0.75,ncar=0,nclev=1,ncbf=1,ncdf=1,ncds=0.95"
for r in "$@"; do
  R4=$(printf %04d $r); [ -s e56_bulk/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e56 S37:$B d95:$B,$N6 d95p1:$B,$N6,ncpt=1 d95p2:$B,$N6,ncpt=2 d95p4:$B,$N6,ncpt=4 d95p8:$B,$N6,ncpt=8 > e56_bulk/rank$R4.tmp 2> e56_bulk/rank$R4.err && mv e56_bulk/rank$R4.tmp e56_bulk/rank$R4.log
done
echo ALLDONE > e56_bulk/done
