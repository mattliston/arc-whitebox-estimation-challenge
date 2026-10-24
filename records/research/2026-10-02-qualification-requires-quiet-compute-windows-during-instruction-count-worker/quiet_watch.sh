#!/bin/bash
# TAG opus-trainer (2026-10-02): keeps the trainer's pool idle inside the lead's quiet windows without anyone having to be awake.
# Every 15 s: looks at the [lead] lines of O/TEAM_LOG.md written after line $2. When the newest of them that names the window says
# QUIET WINDOW OPEN, every queued request of the run is moved from queue/ to held/ (the two workers finish the request they are on,
# at most about two minutes, and then only poll); when it says QUIET WINDOW CLOSED they are moved back. Nothing is deleted.
# (third argument: the state to start in, open or closed; default closed)
# usage: nohup bash quiet_watch.sh <run dir> <first TEAM_LOG line to read> [open] >> <run dir>/quiet_watch.log 2>&1 &
RUN=$1; FROM=$2; LOG=$(dirname "$0")/../TEAM_LOG.md
mkdir -p "$RUN/held"
state=${3:-closed}
while [ ! -e "$RUN/STOP" ]; do
  # only a line that BEGINS with the announcement counts ([lead] <time> QUIET WINDOW OPEN ...): the OPEN line itself mentions CLOSED
  last=$(tail -n +"$FROM" "$LOG" | grep -E -o '^\[lead\] [^ ]+ QUIET WINDOW (OPEN|CLOSED)' | grep -E -o 'QUIET WINDOW (OPEN|CLOSED)' | tail -1)
  if [ "$last" = "QUIET WINDOW OPEN" ] && [ "$state" = closed ]; then
    n=0; for f in "$RUN"/queue/*.json; do [ -e "$f" ] && mv "$f" "$RUN/held/" && n=$((n+1)); done
    state=open; echo "$(date -u +%FT%TZ) window OPEN: $n queued requests held"
  elif [ "$last" = "QUIET WINDOW OPEN" ] && [ "$state" = open ]; then
    for f in "$RUN"/queue/*.json; do [ -e "$f" ] && mv "$f" "$RUN/held/"; done
  elif [ "$last" = "QUIET WINDOW CLOSED" ] && [ "$state" = open ]; then
    n=0; for f in "$RUN"/held/*.json; do [ -e "$f" ] && mv "$f" "$RUN/queue/" && n=$((n+1)); done
    state=closed; echo "$(date -u +%FT%TZ) window CLOSED: $n requests back in the queue"
  fi
  sleep 15
done
