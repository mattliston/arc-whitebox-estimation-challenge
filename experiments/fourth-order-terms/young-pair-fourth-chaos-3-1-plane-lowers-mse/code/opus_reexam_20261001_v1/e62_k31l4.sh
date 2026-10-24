#!/bin/bash
# TAG opus-reexam E62 (lead, 2026-10-02): THE FOURTH-CHAOS PART OF THE YOUNG PAIRS' LOCAL QUARTIC AS A (3,1) PLANE (O/k31l.py, chaos4, nothing removed, ages <= 2, strength 1) ON TOP OF
# THE PAIR-CARRIED LAW (O/ncar6.py, fold in the read, constant regressor, three forms). Network 651 (e61_k31l/): -1.96% final layer, -2.27% all layers against the law alone;
# age 1 only +0.96%, ages <= 3 / 4 -1.23 / -1.51%, strength 0.5 / 1.5 -1.38 / -1.52%; the full local form on top of the law (second chaos counted twice) -0.4%.
# Arms: S37 (base F4 0.6 + K4L 0.75), law, law_k4 (law + the plane).  usage: e62_k31l4.sh <rank> ...  (one torch process of 4 threads; idle gate 30%; no quiet window)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
quiet() { while window_open || ! $O/idle_gate.sh 30 > /dev/null; do sleep 20; done; }
cd $O; export OMP_NUM_THREADS=4
B="f4=0.6,age=2,c=0.75"; N6="nc=1.0,nclaw=6,nclo=1,nca=0.75,ncar=0,nclev=1,ncbf=3,ncdf=3"
for r in "$@"; do
  R4=$(printf %04d $r); [ -s e62_k31l4/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e62 S37:$B law:$B,$N6 law_k4:$B,$N6,k3l=1,k3lc=1,k3lf=0 > e62_k31l4/rank$R4.tmp 2> e62_k31l4/rank$R4.err && mv e62_k31l4/rank$R4.tmp e62_k31l4/rank$R4.log
done
echo ALLDONE > e62_k31l4/done
