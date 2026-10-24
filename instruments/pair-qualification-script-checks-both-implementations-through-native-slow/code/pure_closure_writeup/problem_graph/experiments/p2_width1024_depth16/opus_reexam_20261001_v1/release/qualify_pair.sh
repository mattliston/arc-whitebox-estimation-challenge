#!/bin/bash
# TAG opus-release (2026-10-10): qualify a new physics file from the engineer as TWO submissions in one go, one process at a time:
#   BANK file = <file> + Rl3_bind128                                   (the public bank, as S41 / S42 were fired)
#   SAFE file = <file> + the fragments in SAFE_FRAGS (default 'l3_bind128 lvfs_limit_off sym_two_strips'; the file to designate:
#               fewer requests on every network)
# Fragments are appended by build_variant.py (variants/<tag>_bank, variants/<tag>_safe). Steps: stage A of both with the reviewed
# signatures (nothing is uploaded); then the BANK file in full (stage C session with the setup under the platform clock with the
# constants charged, parent = the stored legs of the bank in force; stage D; the 1.3x slow-setup session; its judge lines =
# 'BANK VERDICT'), then the SAFE file the same way ('SAFE VERDICT'). Stops within 5 s of a QUIET WINDOW OPEN line of the lead;
# every step logs to runs/qualify_<tag>.log.
# usage: [ONLY=safe|bank] [SAFE_FRAGS=...] [DRY=1] bash qualify_pair.sh <engineer file.py> <tag> ["reviewed sig;sig"]  (DRY=1: builds + stage A only)      watch: tail -F O/release/runs/qualify_<tag>.log
set -u
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
FILE="$1"; TAG="$2"; REV="${3:-}"; LOG=runs/qualify_${TAG}.log
[ -f "$FILE" ] || { echo "no such file: $FILE"; exit 2; }
[ -e "$LOG" ] && { echo "tag $TAG used already ($LOG)"; exit 2; }
echo $$ > runs/qualify_${TAG}.pid
LEGS="$(cat "$(ls calibration/bank_stage_c_paced_*.txt | sort | tail -n 1)")"
window_open() { [ "$(grep -E '^\[lead\] .{0,40}QUIET WINDOW (OPEN|CLOSED)' ../TEAM_LOG.md | tail -n 1 | grep -Eo 'QUIET WINDOW (OPEN|CLOSED)' | head -n 1)" = "QUIET WINDOW OPEN" ]; }
step() { name=$1; pat=$2; shift; shift
  if window_open; then echo "$(date -u +%H:%M:%SZ) quiet window open: $name not started, stopping" >> $LOG; exit 0; fi
  echo "$(date -u +%H:%M:%SZ) start $name (load $(uptime | sed 's/.*load averages: //'))" >> $LOG
  "$@" > runs/qualify_${TAG}_${name}.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if window_open; then
      echo "$(date -u +%H:%M:%SZ) quiet window opened: stopping $name" >> $LOG
      for p in $(pgrep -f "^[^ ]*python[^ ]* -B [^ ]*(boundary_run|native_probe).py .*$pat"); do kill -TERM -- -"$(ps -o pgid= -p $p | tr -d ' ')" 2>/dev/null; done
      pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; exit 0
    fi
    sleep 5
  done
  echo "$(date -u +%H:%M:%SZ) $name ended: $(tail -n 1 runs/qualify_${TAG}_${name}.log | cut -c1-90)" >> $LOG
}
echo "$(date -u +%H:%M:%SZ) qualify_pair $TAG: file $FILE sha $(shasum -a 256 "$FILE" | cut -c1-12); parent legs $LEGS" > $LOG
B=variants/${TAG}_bank/candidate.py; S=variants/${TAG}_safe/candidate.py
$PY -B build_variant.py "$FILE" ${TAG}_bank l3_bind128 >> $LOG 2>&1 || exit 1
SAFE_FRAGS="${SAFE_FRAGS:-l3_bind128 lvfs_limit_off sym_two_strips}"; echo "safe fragments: $SAFE_FRAGS" >> $LOG
$PY -B build_variant.py "$FILE" ${TAG}_safe $SAFE_FRAGS >> $LOG 2>&1 || exit 1
step a_bank "${TAG}_bank" $PY -B stage_a_package.py "$B" a_${TAG}_bank --reviewed "$REV"
step a_safe "${TAG}_safe" $PY -B stage_a_package.py "$S" a_${TAG}_safe --reviewed "$REV"
[ -n "${DRY:-}" ] && { echo "$(date -u +%H:%M:%SZ) DRY=1: builds and stage A only, QUALIFY DONE" >> $LOG; exit 0; }
for who in ${ONLY:-bank safe}; do
  F=variants/${TAG}_$who/candidate.py
  step c_$who "runs/c_${TAG}_$who/" $PY -B stage_c_boundary.py "$F" c_${TAG}_$who --short-setup --legs C --parent-from "$LEGS"
  step d_$who "${TAG}_$who" $PY -B stage_d_serving.py "$F" d_${TAG}_$who --short-setup
  step slow_$who "runs/${TAG}_${who}_slow/" $PY -B stage_c_boundary.py "$F" ${TAG}_${who}_slow --setup-slowdown 1.3 --legs C --parent-from "$LEGS"
  echo "$(date -u +%H:%M:%SZ) $(echo $who | tr a-z A-Z) VERDICT" >> $LOG
  $PY -B designation_judge.py ${TAG}_$who --pair runs/c_${TAG}_$who --slow runs/${TAG}_${who}_slow --worker runs/d_${TAG}_$who >> $LOG 2>&1
done
echo "$(date -u +%H:%M:%SZ) QUALIFY DONE" >> $LOG
