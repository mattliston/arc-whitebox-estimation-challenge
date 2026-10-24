#!/bin/bash
# TAG analyst A11: run a11_rescan.py (new submission ids only, polite, sequential) at each UTC time given as HH:MM, then exit.
# It only sleeps in between: no compute, no memory. Log: a11_waiter.out; each run appends one line to a11_rescan.log.
# usage: nohup bash a11_waiter.sh 02:00 08:00 > a11_waiter.out 2>&1 &
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst
PY=/Users/matt/anaconda3/bin/python
for t in "$@"; do
  secs=$($PY -c "import sys,datetime as d; now=d.datetime.now(d.timezone.utc); h,m=map(int,sys.argv[1].split(':')); t=now.replace(hour=h,minute=m,second=0,microsecond=0); t=t+d.timedelta(days=1) if t<=now else t; print(int((t-now).total_seconds()))" $t)
  echo "$(date -u +%FT%TZ) waiting ${secs}s until $t UTC"
  sleep $secs
  echo "$(date -u +%FT%TZ) re-scan"; cd $A && $PY $A/a11_rescan.py
done
echo "$(date -u +%FT%TZ) A11_WAITER_DONE"
