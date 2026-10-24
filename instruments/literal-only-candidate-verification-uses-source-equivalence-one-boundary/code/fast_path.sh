#!/bin/bash
# TAG opus-release: FAST PATH for a file that differs from an already QUALIFIED file only in literals (the trainer's dial
# candidates); fast_path.py holds the argument and the gates. Order: (1) the literal proof (2 seconds; when code differs it stops
# here: full chain); (2) stage A of the new file (contract, archive to fire, einsum signatures against the qualified file: a
# literal-only file has exactly its signatures); (3) one worker session of each file under the platform clock on networks 35 and
# 100 (the qualified file's session is reused when a stored one holds the full census) and the verdict. About 2 minutes per
# session on a quiet box, 12-15 at load 20-30; counts are exact at any load, so no quiet window is needed. Nothing is uploaded.
# Refuses to start while the lead's quiet window is open (the newest line of O/TEAM_LOG.md that begins with
# '[lead] <time> QUIET WINDOW OPEN' or '... CLOSED') unless the fourth argument is in-window (the lead's own run inside it).
# Give the last file that went through the FULL chain as the qualified file (today S38 = sha 7c1a67f4, tag s38).
# usage: fast_path.sh <qualified.py> <new.py> <tag> [in-window] [--fresh] [--native <qualified rows.json> <new rows.json>]
#   -> runs/a_<tag>/submission.tar.gz (the archive), runs/f_<tag>/fast_path.json, log runs/fast_<tag>.log
#   exit 0 PASS (projections inherited), 3 SAME COST (first file has no stage C and D record), 4 REPEAT (counts the same, a
#   loaded box spoiled the instruction counts), 1 FAIL (full chain), 2 not started
set -u
R="$(cd "$(dirname "$0")" && pwd)"; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
[ $# -ge 3 ] || { awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2; }
QUAL="$1"; NEW="$2"; TAG="$3"; shift 3
INSIDE=no; if [ "${1:-}" = in-window ]; then INSIDE=yes; shift; fi
[ -f "$QUAL" ] && [ -f "$NEW" ] || { echo "no such file: $QUAL or $NEW"; exit 2; }
mkdir -p "$R/runs"; LOG="$R/runs/fast_${TAG}.log"
[ -e "$LOG" ] && { echo "tag $TAG already used ($LOG)"; exit 2; }
echo "$(date -u +%FT%TZ) fast path $TAG qualified=$QUAL new=$NEW" | tee -a "$LOG"
"$PY" -B "$R/literal_diff.py" "$QUAL" "$NEW" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "[fast path] FAIL: not literal-only: no session run -> full chain (stages A, C, D); NOT uploaded" | tee -a "$LOG"; exit 1; }
# only a [lead] line that BEGINS with the phrase (after its time stamp) counts, and its first occurrence: the lead's OPEN line
# of 07:25Z also contains the word CLOSED further on
LAST="$(grep -E '^\[lead\] .{0,40}QUIET WINDOW (OPEN|CLOSED)' "$R/../TEAM_LOG.md" | tail -n 1 | grep -Eo 'QUIET WINDOW (OPEN|CLOSED)' | head -n 1)"
if [ "$LAST" = "QUIET WINDOW OPEN" ] && [ "$INSIDE" = no ]; then
  echo "[fast path] NOT STARTED: the lead's quiet window is open (the literal proof above stands; rerun with a new tag after QUIET WINDOW CLOSED)" | tee -a "$LOG"; exit 2
fi
"$PY" -B "$R/stage_a_package.py" "$NEW" "a_${TAG}" --proven "$QUAL" 2>&1 | tee -a "$LOG"
grep -q '^\[A package\] PASS' "$LOG" || { echo "[fast path] FAIL: stage A of the new file did not pass; NOT uploaded" | tee -a "$LOG"; exit 1; }
"$PY" -B "$R/fast_path.py" "$QUAL" "$NEW" "f_${TAG}" --listed "$@" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
