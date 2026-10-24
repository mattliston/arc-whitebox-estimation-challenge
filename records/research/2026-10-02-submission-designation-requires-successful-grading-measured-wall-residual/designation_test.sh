#!/bin/bash
# TAG opus-release: THE DESIGNATION TEST of one archive-ready file (rule accepted by the lead, TEAM_LOG 2026-10-02 03:50Z / 04:0xZ).
# Run it only in a quiet half hour the lead has announced; it starts three heavy steps one after the other and nothing else:
#   1. the pair in ONE window under the platform clock: S35, file, file, S35 (2 networks each);
#   2. one session of the file with its setup at the pace of a worker 1.3x slower;
#   3. the 20-network worker of the file under the platform clock, with native parity on networks 35 and 100.
# Then designation_judge.py prints one line per criterion and one verdict line. Nothing is uploaded.
# Pass = at most 43,300 requests per network; at most +2% requests under the slow setup; every key bound at a normal worker's
# pace with a binding phase of at most 18,100 requests; projected mean wall at most 80 s; projected mean residual at most
# 0.305 s; handles under 70,000; group RSS under 7.5 GB; all outputs finite; bytes and bill equal to native on 35 and 100.
# usage: designation_test.sh <file.py> <tag> ["reviewed einsum sig;sig"]     (about 25 minutes on a quiet box)
set -u
R="$(cd "$(dirname "$0")" && pwd)"; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
FILE="$1"; TAG="$2"; LOG="$R/runs/${TAG}_designation.log"
echo "$(date -u +%FT%TZ) designation test $TAG file=$FILE sha256=$(shasum -a 256 "$FILE" | cut -c1-12)" | tee -a "$LOG"
"$PY" -B "$R/stage_a_package.py" "$FILE" "${TAG}_a" --reviewed "${3:-}" 2>&1 | tail -n 1 | tee -a "$LOG"
"$PY" -B "$R/setup_margin.py" "$TAG=$FILE" 2>&1 | tee -a "$LOG"
"$PY" -B "$R/stage_c_boundary.py" "$FILE" "${TAG}_pair" --short-setup --legs PCCP 2>&1 | tee -a "$LOG"
"$PY" -B "$R/stage_c_boundary.py" "$FILE" "${TAG}_slow" --setup-slowdown 1.3 --legs C --parent-from "$R/runs/${TAG}_pair" 2>&1 | tee -a "$LOG"
"$PY" -B "$R/stage_d_serving.py" "$FILE" "${TAG}_worker" --short-setup 2>&1 | tee -a "$LOG"
"$PY" -B "$R/designation_judge.py" "$TAG" 2>&1 | tee -a "$LOG"
