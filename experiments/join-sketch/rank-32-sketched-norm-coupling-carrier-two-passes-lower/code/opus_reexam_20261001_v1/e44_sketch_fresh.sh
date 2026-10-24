#!/bin/bash
# TAG opus-reexam E44: the sketched rank-r carrier (O/ncar4.py, lean law) as an extra K31 plane on held-out ranks, torch, no refit.
# Arms: base; r 32 / 24 probes two-pass; the same starting at layer 6; single pass; conversion from the young tier only; no conversion.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
cd $O; mkdir -p e44_sketch_fresh; B="f4=0.6,age=2,c=0.75"; R="nc=1.0,nclaw=4"
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
for r in "$@"; do
  f=e44_sketch_fresh/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "noconv"' $f && continue
  while window_open || [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -ge ${GATE:-22} ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e44 S37:$B s4:$B,$R s6:$B,$R,ncstart=6 p1:$B,$R,ncpass=1 young:$B,$R,ncconv=2 noconv:$B,$R,ncconv=0 > $f 2>&1
done
echo E44_DONE >> e44_sketch_fresh/done.txt
