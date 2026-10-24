#!/bin/bash
# TAG opus-reexam E37: the free-running vector carrier (O/ncar.py, law 2) as an extra K31 plane on held-out newer ranks, torch, no refit.
# Arms: base (F4 0.6 + K4L 0.75); modes 2-8 at strength 1 and 1.5; modes 1-8 at 1; K = 16 without and with mode 1.
# usage: e37_ncar_fresh.sh <rank> [<rank> ...]   (4 threads, nice 15, each network starts at a 1-minute load under 14 and outside quiet windows)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
cd $O; mkdir -p e37_ncar_fresh; B="f4=0.6,age=2,c=0.75"
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
for r in "$@"; do
  f=e37_ncar_fresh/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "K16m1"' $f && continue
  while window_open || [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -ge ${GATE:-14} ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e37 S37:$B k10:$B,nc=1.0 k15:$B,nc=1.5 k10m1:$B,nc=1.0,ncm1=1 K16:$B,nc=1.0,nck=16 K16m1:$B,nc=1.0,nck=16,ncm1=1 > $f 2>&1
done
echo E37_DONE >> e37_ncar_fresh/done.txt
