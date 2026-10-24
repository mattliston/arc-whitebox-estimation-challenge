#!/bin/bash
# TAG analyst A11 waiter v5 (2026-10-13 02:1xZ: v4 restarted after the lead's accidental stop at 02:08Z, with a11_watch_v5.py = NO forum reads, Matt 10-12;
# single-thread BLAS env): every hour on the hour a11_watch_v5.py (new flopscope / whestbench release, selection text on the challenge page; HTTP only), and at 02 / 06 / 10 / 14 / 18 / 22:00 UTC the board scan a11_rescan_v4.py
# (HTTP only; it logs every top-six change with the new best's final / layers-0-14 ratio next to ours). Both write TEAM_LOG lines on change only.
# A scan slot is skipped when another a11_rescan_v4.py is running. Ends at the end time given (default 2026-10-18T02:00:00Z).
# Stop: touch O/analyst/A11_STOP (read every hour). Log: a11_waiter_v5.out; scan output: a11_waiter_v5.scans.out.
# usage: nohup bash a11_waiter_v5.sh [end ISO] >> a11_waiter_v5.out 2>&1 &
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst
PY=/Users/matt/anaconda3/bin/python; END=${1:-2026-10-18T02:00:00Z}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
while true; do
  [ -e $A/A11_STOP ] && { echo "$(date -u +%FT%TZ) A11_STOP present: waiter ends"; exit 0; }
  read secs slot hour <<< "$($PY -c "
import datetime as d, sys
now = d.datetime.now(d.timezone.utc); end = d.datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
t = now.replace(minute=0, second=0, microsecond=0) + d.timedelta(hours=1)
print(int((t - now).total_seconds()) if t <= end else -1, t.strftime('%Y-%m-%dT%H:%MZ'), t.hour)" $END)"
  [ "$secs" -lt 0 ] && { echo "$(date -u +%FT%TZ) next slot $slot is after the end $END: waiter ends"; exit 0; }
  sleep $secs
  [ -e $A/A11_STOP ] && { echo "$(date -u +%FT%TZ) A11_STOP present: waiter ends"; exit 0; }
  if [ $((hour % 4)) -eq 2 ]; then
    if pgrep -f "[a]11_rescan(_v[0-9]+)?[.]py" > /dev/null; then echo "$(date -u +%FT%TZ) another a11_rescan_v4.py is running: scan slot $slot skipped"
    else echo "$(date -u +%FT%TZ) re-scan at $slot"; ( cd $A && $PY $A/a11_rescan_v4.py >> $A/a11_waiter_v5.scans.out 2>&1 ); echo "$(date -u +%FT%TZ) re-scan done: $(tail -n 1 $A/a11_rescan.log | cut -c1-200)"; fi
  fi
  ( cd $A && $PY $A/a11_watch_v5.py 2>&1 | tail -n 1 )
  sleep 5
done
