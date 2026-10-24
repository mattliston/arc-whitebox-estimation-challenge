#!/bin/bash
# TAG opus-trainer RAD (10-12 13:2xZ; lead 13:3xZ): one memory / lane sample a minute (cert/RAD/memlog_v1.py -> cert/RAD/memlog_v1.jsonl) until
# 2026-10-12T21:00Z or the file cert/RAD/memlog_STOP, to measure S51's downstream (stage 2, port, release gates) for the hold rule from S53.
# usage (from O/trainer): nohup bash cert/RAD/memlog_v1.sh > cert/RAD/memlog_v1.out 2>&1 &
cd "$(dirname "$0")/../.."
END=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-12T21:00:00Z" +%s)
while [ $(date +%s) -lt $END ] && [ ! -e cert/RAD/memlog_STOP ]; do
  /Users/matt/anaconda3/bin/python -B cert/RAD/memlog_v1.py || echo "$(date -u +%T) sample failed"
  sleep 60
done
