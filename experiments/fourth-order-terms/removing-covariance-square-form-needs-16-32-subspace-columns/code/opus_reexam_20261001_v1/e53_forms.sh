#!/bin/bash
# TAG opus-reexam E53 (lead, 2026-10-02): WHICH FORMS the fold of the pair-carried law (O/ncar6.py, level 'read') must remove. E51: exact four forms -7.46% pooled on seven
# networks, only the multiple of C -5.78% (networks 287 and 641 lose everything). Arms, exact off-diagonal least squares: readf2 = tr C x C and mu mu'; readf3 = + mu (C mu)' + sym;
# (readf1 and readf4 are in e51_pairlaw/). The C^2 form needs an n^3 product natively; the mean forms are rank one and two.
# usage: e53_forms.sh <rank> ...  (one torch process of 4 threads; load gate 30; no quiet window)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
quiet() { while window_open || ! $O/idle_gate.sh 30 > /dev/null; do sleep 20; done; }   # gate on the idle CPU share (idle_gate.sh), not on the load average
cd $O; export OMP_NUM_THREADS=4
B="f4=0.6,age=2,c=0.75"; N6="nc=1.0,nclaw=6,nclo=1,nca=0.75,ncar=0,nclev=1"
for r in "$@"; do
  R4=$(printf %04d $r); [ -s e53_forms/rank$R4.log ] && continue
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e53 S37:$B readf2:$B,$N6,ncbf=2 readf3:$B,$N6,ncbf=3 > e53_forms/rank$R4.tmp 2> e53_forms/rank$R4.err && mv e53_forms/rank$R4.tmp e53_forms/rank$R4.log
done
echo ALLDONE > e53_forms/done
