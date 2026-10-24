#!/bin/bash
# TAG opus-physicist t63 chain: adjoint dumps of the S42 torch card (t63_adjoint_dump.py, trainer's gradient mode) on the trainer's 36 gradient ranks, one at a time,
# idle gate and quiet window before each; also waits while a [lead] MACHINE PRIORITY line is the newest priority line and its end time has not passed.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
window_open() {
  local o c
  o=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW OPEN" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1); c=$(grep -n -E "^\[lead\] [0-9T:Z-]+ QUIET WINDOW CLOSED" $O/TEAM_LOG.md | tail -n 1 | cut -d: -f1)
  [ -n "$o" ] && { [ -z "$c" ] || [ "$o" -gt "$c" ]; }
}
priority() {      # true while the newest '[lead] ... MACHINE PRIORITY UNTIL hh:mmZ' line has not run out (same UTC day)
  local u now
  u=$(grep -E "^\[lead\] [0-9T:Z-]+ MACHINE PRIORITY UNTIL [0-9]{2}:[0-9]{2}Z" $O/TEAM_LOG.md | tail -n 1 | sed -E 's/.*MACHINE PRIORITY UNTIL ([0-9]{2}):([0-9]{2})Z.*/\1\2/')
  now=$(date -u +%H%M)
  [ -n "$u" ] && [ "$now" -lt "$u" ]
}
quiet() { while window_open || priority || ! bash $O/idle_gate.sh 30 > /dev/null; do sleep 20; done; }
cd $O/physicist
for r in "$@"; do
  R4=$(printf %04d $r); [ -f adump_rank${R4}.npz ] && continue
  [ -f STOP_T63 ] && { echo "STOP_T63 present, stopping" >> t63_chain.log; break; }
  quiet; nice -n 15 /Users/matt/anaconda3/bin/python -B t63_adjoint_dump.py $r 2 >> t63_chain.log 2>&1
done
echo ALLDONE >> t63_chain.log
