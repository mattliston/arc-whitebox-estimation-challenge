#!/bin/bash
# TAG opus-lead: poll a submission's public evaluation page until its state leaves 'running' (reader: P2/fable_fire_v1/read_eval_page.py,
# which writes eval_<id>.json into the current directory). usage: poll_eval.sh <tag dir> <submission id> [seconds between polls]
cd "$(dirname "$0")/$1" || exit 1
ID=$2; EVERY=${3:-150}
for i in $(seq 1 80); do
  /Users/matt/arc-white-box-estimation/.venv/bin/python -B ../../../fable_fire_v1/read_eval_page.py $ID > eval_summary.txt 2> eval_err.txt
  state=$(/Users/matt/arc-white-box-estimation/.venv/bin/python -c "import json;ev=json.load(open('eval_$ID.json'))['evaluation'];print(ev.get('state'), (ev.get('progress') or {}).get('mlps_completed'), (ev.get('progress') or {}).get('mlps_failed'))" 2>/dev/null)
  echo "$(date -u +%H:%M:%S) $state" >> poll.log
  case "$state" in running*|queued*|pending*|"") sleep $EVERY ;; *) break ;; esac
done
date -u +%H:%M:%S; tail -n 3 poll.log; cat eval_summary.txt
