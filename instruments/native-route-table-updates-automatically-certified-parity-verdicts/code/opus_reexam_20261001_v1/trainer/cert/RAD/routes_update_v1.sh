#!/bin/bash
# TAG opus-trainer RAD (10-12 17:0xZ): runs cert/RAD/routes_update_v1.py every 2 min until 10-17 12:00Z (stop: cert/RAD/routes_update_STOP).
# usage (from O/trainer): nohup bash cert/RAD/routes_update_v1.sh > cert/RAD/routes_update_v1.log 2>&1 &
cd "$(dirname "$0")/../.."
END=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-17T12:00:00Z" +%s)
while [ $(date +%s) -lt $END ] && [ ! -e cert/RAD/routes_update_STOP ]; do
  /Users/matt/anaconda3/bin/python -B cert/RAD/routes_update_v1.py || echo "$(date -u +%T) pass failed"
  sleep 120
done
