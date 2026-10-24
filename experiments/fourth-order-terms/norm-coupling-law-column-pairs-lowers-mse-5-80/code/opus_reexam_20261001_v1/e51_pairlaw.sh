#!/bin/bash
# TAG opus-reexam E51 (lead, 2026-10-02): THE PAIR-CARRIED LEAN LAW (O/ncar6.py) on held-out ranks, torch, no refit, base = F4 0.6 + K4L 0.75 (S37 physics).
# Arms: r31f4 = the plane through remainder31, four forms removed exactly (research reference); readf4 / readf1 = the fold INSIDE the bank read with four forms / the
# multiple of C removed by the exact off-diagonal fit; readtab = the native candidate: multiple of C = per-layer literal x diagonal regression (dfit 2).
# usage: e51_pairlaw.sh <rank> ...   one torch process of 4 threads, each start waits for a 1-minute load under 26 and no quiet window.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
quiet() { while window_open || [ "$(sysctl -n vm.loadavg | awk '{print ($2 >= 26)}')" = "1" ]; do sleep 20; done; }
cd $O; export OMP_NUM_THREADS=4
B="f4=0.6,age=2,c=0.75"; N6="nc=1.0,nclaw=6,nclo=1,nca=0.75,ncar=0"
for r in "$@"; do
  R4=$(printf %04d $r); [ -s e51_pairlaw/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e51 S37:$B r31f4:$B,$N6 readf4:$B,$N6,nclev=1 readf1:$B,$N6,nclev=1,ncbf=1 readtab:$B,$N6,nclev=1,ncbf=1,ncdf=2 > e51_pairlaw/rank$R4.tmp 2> e51_pairlaw/rank$R4.err && mv e51_pairlaw/rank$R4.tmp e51_pairlaw/rank$R4.log
done
echo ALLDONE > e51_pairlaw/done
