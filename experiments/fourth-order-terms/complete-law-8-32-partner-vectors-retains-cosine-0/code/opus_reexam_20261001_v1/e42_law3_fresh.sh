#!/bin/bash
# TAG opus-reexam E42: the complete law on K vectors (O/ncar3.py) as an extra K31 plane on held-out ranks, torch, no refit.
# Arms: base; K = 16 and 8 (all bank columns in the conversion); K = 16 with the conversion from the young tier only / none; K = 16 with the
# native subspace form (pushed vectors + 2 power sweeps); K = 16 without the K21 terms.   usage: e42_law3_fresh.sh <rank> ...
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
cd $O; mkdir -p e42_law3_fresh; B="f4=0.6,age=2,c=0.75"; R="nc=1.0,nclaw=3,nck=16"
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
for r in "$@"; do
  f=e42_law3_fresh/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "nok21"' $f && continue
  while window_open || [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -ge ${GATE:-22} ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e42 S37:$B K16:$B,$R K8:$B,nc=1.0,nclaw=3,nck=8 young:$B,$R,ncconv=2 noconv:$B,$R,ncconv=0 sw2:$B,$R,ncsw=2 nok21:$B,$R,ncskip=1 > $f 2>&1
done
echo E42_DONE >> e42_law3_fresh/done.txt
