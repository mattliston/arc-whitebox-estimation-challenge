#!/bin/bash
# TAG analyst A11 waiter v2 (2026-10-10, final week; lead 10-10 02:11Z: board scans every 4 h by themselves). Runs a11_rescan.py (HTTP only, new ids, polite)
# at 02:00 / 06:00 / 10:00 / 14:00 / 18:00 / 22:00 UTC until the end time given (default 2026-10-18T02:00:00Z). a11_rescan.py itself writes a TEAM_LOG line
# for every top-six change and rebuilds the derived tables (a16_loss_risk.md, a18_third_place.md).
# Never two scans at once: when another a11_rescan.py is running at a slot (someone started one by hand), that slot is skipped.
# Stop: touch O/analyst/A11_STOP (read before each slot). Log: a11_waiter_v2.out; each scan's own output: a11_waiter_v2.scans.out.
# usage: nohup bash a11_waiter_v2.sh [end ISO] >> a11_waiter_v2.out 2>&1 &
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst
PY=/Users/matt/anaconda3/bin/python; END=${1:-2026-10-18T02:00:00Z}
while true; do
  [ -e $A/A11_STOP ] && { echo "$(date -u +%FT%TZ) A11_STOP present: waiter ends"; exit 0; }
  read secs slot <<< "$($PY -c "
import datetime as d, sys
now = d.datetime.now(d.timezone.utc); end = d.datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
t = now.replace(minute=0, second=0, microsecond=0)
while t <= now or t.hour % 4 != 2: t += d.timedelta(hours=1)
print(int((t - now).total_seconds()) if t <= end else -1, t.strftime('%Y-%m-%dT%H:%MZ'))" $END)"
  [ "$secs" -lt 0 ] && { echo "$(date -u +%FT%TZ) next slot $slot is after the end $END: waiter ends"; exit 0; }
  echo "$(date -u +%FT%TZ) waiting ${secs}s until $slot"
  sleep $secs
  [ -e $A/A11_STOP ] && { echo "$(date -u +%FT%TZ) A11_STOP present: waiter ends"; exit 0; }
  if pgrep -f "a11_rescan.py" > /dev/null; then echo "$(date -u +%FT%TZ) another a11_rescan.py is running: slot $slot skipped"; sleep 60; continue; fi
  echo "$(date -u +%FT%TZ) re-scan at $slot"
  ( cd $A && $PY $A/a11_rescan.py >> $A/a11_waiter_v2.scans.out 2>&1 )
  echo "$(date -u +%FT%TZ) re-scan done: $(tail -n 1 $A/a11_rescan.log | cut -c1-200)"
  sleep 60
done
