#!/bin/bash
# TAG opus-reexam E39: design variants of the vector carrier (O/ncar.py law 2) on held-out ranks: what the native port must contain.
# Arms: base; K16 with mode 1 (reference); without the bank pass (conv 0); without the isotropic transport terms (a = b = 0);
# K = 24 and 32; transport factor 1.0; plane strength 1.5.   usage: e39_ncar_design.sh <rank> ...   (waits for E37 to finish; load-gated)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
cd $O; mkdir -p e39_ncar_design; B="f4=0.6,age=2,c=0.75"; R="nc=1.0,nck=16,ncm1=1"
until grep -q E37_DONE e37_ncar_fresh/done.txt 2>/dev/null; do sleep 20; done
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
for r in "$@"; do
  f=e39_ncar_design/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "s15"' $f && continue
  while window_open || [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -ge ${GATE:-14} ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e39 S37:$B ref:$B,$R noconv:$B,$R,ncconv=0 noiso:$B,$R,nca=0,ncb=0 K24:$B,nc=1.0,nck=24,ncm1=1 K32:$B,nc=1.0,nck=32,ncm1=1 t10:$B,$R,nct=1.0 s15:$B,nc=1.5,nck=16,ncm1=1 > $f 2>&1
done
echo E39_DONE >> e39_ncar_design/done.txt
