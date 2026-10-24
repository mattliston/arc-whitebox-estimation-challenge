#!/bin/bash
# TAG opus-release fastgate v1 (2026-10-12 17:55 UTC; lead 17:1xZ APPROVED with conditions): a TWIN gated by RE-ASSESSING the engineer's
# sha-matched count sessions (same harness, deterministic counts: S50 reproduced to the request and the flop) instead of re-measuring them,
# then the FINAL twin's own native ten networks stay ON the fire path; the lead fires only after the CONFIRM line.
#  (0) fragment check: one Rl3_bind128, one Rprices_zero, no Rlvfs_limit_off;
#  (1) fastgate_pin_v1.py --check: the file's sha in each session's manifest and launcher, the pinned harness (boundary_run.py & co), the
#      gate's flags; ANY failure, or no BASEQ -> FALLBACK to the full twin_pipeline_v4.sh with one line saying why (not in rehearsals);
#  (2) gate: REASSESS_N/_S[/_39] FRAGS=none MODE=primary s47_safe_v8.sh -> '<tag> SAFE GATE PASS (... FAST ...)' (+ FIRE COMMAND) or DO NOT FIRE;
#  (3) the FINAL twin's native ten networks (engineer/quick.py, list a, guard v2) -> runs/quick_<tag>_a.json;
#  (4) runs/compare_twin_native_v3.py against BASEQ (the native session of the safe file of the SAME physics, e.g. the engineer's census
#      quick_keys_<..>s_a.json) -> 'TWIN <tag> CONFIRMED (...) ... GO + FIRE COMMAND' or 'TWIN <tag> DO NOT FIRE (...)', residual flag included.
# Before (1) and before (3) it waits while loop v2 runs a round (s47_safe_v7.sh / twin_pipeline_v4.sh alive: S<N> gates first) or the lead's
# QUIET WINDOW is open; (3) waits for gate.sh. TL=<file> writes every line there instead of TEAM_LOG (rehearsals).
# usage: BASEQ=<safe-physics quick json> [SAFESRC=<safe .py, for the fallback>] [PROVEN=..] [TWIN_DESC=..] [TL=..]
#        bash fastgate_v1.sh <twin .py> <tag> <fire tag> <normal session dir> <slow session dir> [<35,100,39 session dir>]
#        watch: tail -F O/release/runs/fastgate_<tag>.log
set -u
cd "$(dirname "$0")"; R=$PWD; O=$(dirname "$R"); P2=$(cd ../.. && pwd); PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
F=$1; TAG=$2; FT=$3; N=$4; S=$5; N39=${6:-}; LOG=runs/fastgate_${TAG}.log; TL=${TL:-$O/TEAM_LOG.md}
[ -f "$F" ] || { echo "no such file $F"; exit 2; }
[ -e "$LOG" ] && { echo "tag $TAG used already"; exit 2; }
if [ -z "${PROVEN:-}" ]; then   # the last FIRED safe file (as release_loop_v2): newest O/fire/* with a receipt and no 'twin' in its tag
  PROVEN=$($PY - <<'EOF'
import json; from pathlib import Path
O = Path('..').resolve(); best = None
for a in (O/'fire').glob('*/attempt.json'):
    d = json.loads(a.read_text())
    if 'twin' not in d.get('tag', '') and d.get('archive') and (a.parent/'receipt.json').exists() and (best is None or d.get('utc', '') > best[0]):
        best = (d.get('utc', ''), Path(d['archive']).parent/'payload'/'estimator.py')
print(best[1])
EOF
)
fi
say() { echo "$(date -u +%H:%M:%SZ) $*" >> $LOG; }
window_open() { [ "$(grep -E '^\[lead\] .{0,40}QUIET WINDOW (OPEN|CLOSED)' ../TEAM_LOG.md | tail -n 1 | grep -Eo 'QUIET WINDOW (OPEN|CLOSED)' | head -n 1)" = "QUIET WINDOW OPEN" ]; }
yield_to_loop() { while pgrep -f "s47_safe_v7.sh|twin_pipeline_v4.sh" > /dev/null || window_open; do say "waiting: loop v2 round running or quiet window open"; sleep 60; done; }
c() { grep -c "^# BEGIN OPUS-RELEASE FRAGMENT $1 " "$F"; }
fallback() {
  say "FALLBACK: $1"
  if [ "$TL" != "$O/TEAM_LOG.md" ]; then say "rehearsal: the full pipeline is NOT run"; exit 5; fi
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) FASTGATE $TAG FELL BACK TO THE FULL TWIN PIPELINE: $1" >> "$TL"
  BASEQ=${BASEQ:-} BASESRC=${SAFESRC:-} PROVEN=$PROVEN TWIN_DESC="${TWIN_DESC:-twin $TAG}" bash twin_pipeline_v4.sh "$F" $TAG $FT; exit $?
}
say "fastgate $TAG: $F sha $(shasum -a 256 "$F" | cut -c1-12); normal $N; slow $S; n39 ${N39:-none}; base ${BASEQ:-none}; proven $PROVEN"
if [ "$(c l3_bind128)" != 1 ] || [ "$(c prices_zero)" != 1 ] || [ "$(c lvfs_limit_off)" != 0 ]; then
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) TWIN $TAG DO NOT FIRE: STOPPED AT THE FRAGMENT CHECK (l3_bind128 x$(c l3_bind128), prices_zero x$(c prices_zero), lvfs_limit_off x$(c lvfs_limit_off); a twin needs exactly one Rl3_bind128 and one Rprices_zero and no Rlvfs_limit_off)" >> "$TL"; exit 3
fi
yield_to_loop
PINL=$($PY -B fastgate_pin_v1.py --check fastgate_pin_v1.json "$F" "$N" "$S" ${N39:+--n39 "$N39"} 2>&1) || fallback "$(echo "$PINL" | cut -c1-500)"
say "$PINL"
[ -n "${BASEQ:-}" ] && [ -f "$BASEQ" ] || fallback "no native session of the safe file of the same physics (BASEQ)"
T0=$(date +%s)
REASSESS_N="$N" REASSESS_S="$S" REASSESS_39="$N39" FRAGS=none MODE=primary PROVEN=$PROVEN TL="$TL" \
  REV='alsqj,lsjk,lk->aqsk;a,sq,k->aqsk;kn,kc->nc;als,lsjk,lk->asjk;a,sj,k->asjk' \
  bash s47_safe_v8.sh "$F" $TAG $FT "${TWIN_DESC:-twin $TAG (fast-gated)} (file $(shasum -a 256 "$F" | cut -c1-8))"
say "gate done in $(( $(date +%s) - T0 )) s"
grep -q "$TAG SAFE GATE PASS" "$TL" || { say "gate did not pass: no native run"; exit 0; }
yield_to_loop
bash ../gate.sh 4 "fastgate $TAG" >> $LOG 2>&1
say "native ten start (load $(uptime | sed 's/.*load averages: //'))"
(cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 \
  $PY -B $R/guard_footprint_v2.py --record $R/runs/quick_${TAG}_a_guard.json --seconds 3000 --rss-backstop 7500000000 \
  $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $P2/opus_reexam_20261001_v1/engineer/quick.py \
  $R/variants/${TAG}_safe/candidate.py 651,378,424,328,287,641,548,557,316,390 $R/runs/quick_${TAG}_a.json > $R/runs/quick_${TAG}_a.log 2>&1)
say "native ten ended (exit $?)"
TL="$TL" $PY -B runs/compare_twin_native_v3.py $TAG "$BASEQ" >> $LOG 2>&1 || \
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) TWIN $TAG DO NOT FIRE: the confirmation compare failed (O/release/$LOG)" >> "$TL"
say "FASTGATE DONE in $(( $(date +%s) - T0 )) s"
