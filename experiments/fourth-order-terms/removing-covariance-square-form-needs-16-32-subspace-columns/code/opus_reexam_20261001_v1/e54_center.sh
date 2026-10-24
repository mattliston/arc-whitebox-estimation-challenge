#!/bin/bash
# TAG opus-reexam E54 (lead, 2026-10-02): SOURCE CENTERING in the pair-carried law (O/ncar6.py, fold inside the read). E51/E53: the exact four-form removal is worth 1.7 points
# over the multiple of C alone, all of it the C^2 form (networks 287, 641), which costs an n^3 product natively. The C^2-like part is MADE by the Gaussian births
# (sum_h S2_h c_h c_h' = G Co S2 Co G): centering S2 removes it at the source. Arms (exact least squares for what is still removed, to see the potential):
#   f1 = multiple of C removed (E51's readf1);  cb = births centered, multiple of C removed;  cbr = births and row forms centered, multiple of C removed;
#   cbr0 = both centered, NOTHING removed;  nb = no births at all, multiple of C removed;  f4 = four forms removed (E51's readf4, the target).
# usage: e54_center.sh <rank> ...  (one torch process of 4 threads; idle gate 30%; no quiet window)
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
  R4=$(printf %04d $r); [ -s e54_center/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e54 S37:$B cb:$B,$N6,ncbf=1,nccb=1 cbr:$B,$N6,ncbf=1,nccb=1,nccr=1 cbr0:$B,$N6,ncbf=0,nccb=1,nccr=1 nb:$B,$N6,ncbf=1,ncbi=0 > e54_center/rank$R4.tmp 2> e54_center/rank$R4.err && mv e54_center/rank$R4.tmp e54_center/rank$R4.log
done
echo ALLDONE > e54_center/done
