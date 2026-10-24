#!/bin/bash
# TAG opus-reexam E52 (lead, 2026-10-02): which sources of the pair-carried law (O/ncar6.py) a native port needs. Fold inside the bank read, multiple of C by the
# per-layer table x diagonal regression (the native candidate 'readtab' of E51), base F4 0.6 + K4L 0.75, torch, no refit.
# Arms: tab = all three sources; nodiag = births + row forms (D = 0: six of the eight n^2 products go); nobirth = row forms + diagonal; rowsonly = row forms;
# s125 = all sources at fold strength 1.25; tab09 / tab11 = the table x 0.9 / x 1.1.
# usage: e52_pairlaw_ablate.sh <rank> ...  (waits for E51 to finish; one torch process of 4 threads; load gate 26; no quiet window)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
quiet() { while window_open || ! $O/idle_gate.sh 30 > /dev/null; do sleep 20; done; }   # gate on the idle CPU share (idle_gate.sh), not on the load average
cd $O; export OMP_NUM_THREADS=4
until [ -f e51_pairlaw/done ]; do sleep 30; done
B="f4=0.6,age=2,c=0.75"; N6="nclaw=6,nclo=1,nca=0.75,ncar=0,nclev=1,ncbf=1,ncdf=2"
for r in "$@"; do
  R4=$(printf %04d $r); [ -s e52_pairlaw_ablate/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e52 S37:$B tab:$B,nc=1.0,$N6 nodiag:$B,nc=1.0,$N6,ncdg=0 nobirth:$B,nc=1.0,$N6,ncbi=0 rowsonly:$B,nc=1.0,$N6,ncbi=0,ncdg=0 s125:$B,nc=1.25,$N6 tab09:$B,nc=1.0,$N6,ncds=0.9 tab11:$B,nc=1.0,$N6,ncds=1.1 > e52_pairlaw_ablate/rank$R4.tmp 2> e52_pairlaw_ablate/rank$R4.err && mv e52_pairlaw_ablate/rank$R4.tmp e52_pairlaw_ablate/rank$R4.log
done
echo ALLDONE > e52_pairlaw_ablate/done
