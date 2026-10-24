#!/bin/bash
# v5 (2026-10-12 23:08 UTC; lead 22:57Z, s51k2 = a byte-identical bill stack): when runs/fastgate_mode_<tag>.json exists ({byteid_base, byteid_tag})
# the confirmation is runs/compare_twin_byteid_v1.py: the FINAL twin's native ten must equal that file's native ten byte for byte (pred sha,
# final MSE, all 16 layers) on every network, else DO NOT FIRE; no BASEQ needed, and a pin failure posts DO NOT FIRE YET instead of the full
# twin pipeline. Without a mode file it is v4. v4 = fastgate_v4.sh.
# v4 (2026-10-12 18:08 UTC; lead 17:28Z: round twins also get FASTGATE lines): the confirmation is compare_twin_native_v5.py (a ROUND tag
# s<N>tw keeps the line shape the trainer's add-ons key on: 'TWIN s<N>tw (<sha8>), ... VERDICT: CONFIRMED ...'); it waits for twin_pipeline_v5.sh
# too (loop v3's fallback), and its own fallback is twin_pipeline_v5.sh (7200 s guards). While runs/EASE_UP exists (lead 17:5xZ) the two
# native halves run in sequence, one process at a time. Nothing else changed. v3 = fastgate_v3.sh.
# v3 (2026-10-12 17:50 UTC; lead 17:4xZ (c) APPROVED): the native ten networks run as 2 release processes (651,378,424,328,287 and
# 641,548,557,316,390; gate.sh before each, the second 60 s after the first), each under guard v2 at 7200 s (a hang guard: on today's
# power-capped box a network takes about 6 min and quick.py writes its JSON only at the end), then merged into runs/quick_<tag>_a.json
# (+ _pred_<rank>.npy) for compare v4. Nothing else changed. v2 = fastgate_v2.sh.
# v2 (2026-10-12 17:40 UTC): the confirmation uses runs/compare_twin_native_v4.py: the native run judges the PHYSICS only; the bill comes from
# the deterministic count sessions against the last fired twin ('BILL UP' flag), because quick.py's ops / flops follow the box's load through
# the files' 3 s prebind wall guard (measured: same file, +0.56% flops on a loaded box). Nothing else changed. v1 = fastgate_v1.sh.
# TAG opus-release fastgate v1 (2026-10-12 17:55 UTC; lead 17:1xZ APPROVED with conditions): a TWIN gated by RE-ASSESSING the engineer's
# sha-matched count sessions (same harness, deterministic counts: S50 reproduced to the request and the flop) instead of re-measuring them,
# then the FINAL twin's own native ten networks stay ON the fire path; the lead fires only after the CONFIRM line.
#  (0) fragment check: one Rl3_bind128, one Rprices_zero, no Rlvfs_limit_off;
#  (1) fastgate_pin_v1.py --check: the file's sha in each session's manifest and launcher, the pinned harness (boundary_run.py & co), the
#      gate's flags; ANY failure, or no BASEQ -> FALLBACK to the full twin_pipeline_v4.sh with one line saying why (not in rehearsals);
#  (2) gate: REASSESS_N/_S[/_39] FRAGS=none MODE=primary s47_safe_v8.sh -> '<tag> SAFE GATE PASS (... FAST ...)' (+ FIRE COMMAND) or DO NOT FIRE;
#  (3) the FINAL twin's native ten networks (engineer/quick.py, list a, guard v2) -> runs/quick_<tag>_a.json;
#  (4) runs/compare_twin_native_v5.py against BASEQ (the native session of the safe file of the SAME physics, e.g. the engineer's census
#      quick_keys_<..>s_a.json) -> 'TWIN <tag> CONFIRMED (...) ... GO + FIRE COMMAND' or 'TWIN <tag> DO NOT FIRE (...)', residual flag included.
# Before (1) and before (3) it waits while loop v2 runs a round (s47_safe_v7.sh / twin_pipeline_v4.sh alive: S<N> gates first) or the lead's
# QUIET WINDOW is open; (3) waits for gate.sh. TL=<file> writes every line there instead of TEAM_LOG (rehearsals).
# usage: BASEQ=<safe-physics quick json> [SAFESRC=<safe .py, for the fallback>] [PROVEN=..] [TWIN_DESC=..] [TL=..]
#        bash fastgate_v5.sh <twin .py> <tag> <fire tag> <normal session dir> <slow session dir> [<35,100,39 session dir>]
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
yield_to_loop() { while pgrep -f "s47_safe_v7.sh|twin_pipeline_v[45].sh" > /dev/null || window_open; do say "waiting: loop v2 round running or quiet window open"; sleep 60; done; }
c() { grep -c "^# BEGIN OPUS-RELEASE FRAGMENT $1 " "$F"; }
MODEF=runs/fastgate_mode_${TAG}.json; BYTEID_BASE=""; BYTEID_TAG=""
if [ -f "$MODEF" ]; then   # byte-identical mode (lead 22:57Z): {"byteid_base": "<quick json of the file it must equal>", "byteid_tag": "<its tag>"}
  BYTEID_BASE=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['byteid_base'])" $MODEF); BYTEID_TAG=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['byteid_tag'])" $MODEF)
fi
fallback() {
  say "FALLBACK: $1"
  if [ -n "$BYTEID_BASE" ]; then   # the full twin pipeline judges against a safe base with the native bill: not the byte-identical test
    echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) TWIN $TAG DO NOT FIRE YET: the byte-identical fast gate cannot run ($1); release runs it by hand" >> "$TL"; exit 6
  fi
  if [ "$TL" != "$O/TEAM_LOG.md" ]; then say "rehearsal: the full pipeline is NOT run"; exit 5; fi
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) FASTGATE $TAG FELL BACK TO THE FULL TWIN PIPELINE: $1" >> "$TL"
  BASEQ=${BASEQ:-} BASESRC=${SAFESRC:-} PROVEN=$PROVEN TWIN_DESC="${TWIN_DESC:-twin $TAG}" bash twin_pipeline_v5.sh "$F" $TAG $FT; exit $?
}
say "fastgate $TAG: $F sha $(shasum -a 256 "$F" | cut -c1-12); normal $N; slow $S; n39 ${N39:-none}; base ${BASEQ:-none}; proven $PROVEN"
if [ "$(c l3_bind128)" != 1 ] || [ "$(c prices_zero)" != 1 ] || [ "$(c lvfs_limit_off)" != 0 ]; then
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) TWIN $TAG DO NOT FIRE: STOPPED AT THE FRAGMENT CHECK (l3_bind128 x$(c l3_bind128), prices_zero x$(c prices_zero), lvfs_limit_off x$(c lvfs_limit_off); a twin needs exactly one Rl3_bind128 and one Rprices_zero and no Rlvfs_limit_off)" >> "$TL"; exit 3
fi
yield_to_loop
PINL=$($PY -B fastgate_pin_v1.py --check fastgate_pin_v1.json "$F" "$N" "$S" ${N39:+--n39 "$N39"} 2>&1) || fallback "$(echo "$PINL" | cut -c1-500)"
say "$PINL"
[ -n "$BYTEID_BASE" ] || { [ -n "${BASEQ:-}" ] && [ -f "$BASEQ" ]; } || fallback "no native session of the safe file of the same physics (BASEQ)"
[ -z "$BYTEID_BASE" ] || [ -f "$BYTEID_BASE" ] || fallback "the byte-identical base $BYTEID_BASE is missing"
if [ -n "$BYTEID_BASE" ]; then say "mode: BYTE-IDENTICAL against $BYTEID_TAG ($BYTEID_BASE)"; else say "mode: normal (physics against BASEQ)"; fi
T0=$(date +%s)
REASSESS_N="$N" REASSESS_S="$S" REASSESS_39="$N39" FRAGS=none MODE=primary PROVEN=$PROVEN TL="$TL" \
  REV='alsqj,lsjk,lk->aqsk;a,sq,k->aqsk;kn,kc->nc;als,lsjk,lk->asjk;a,sj,k->asjk' \
  bash s47_safe_v8.sh "$F" $TAG $FT "${TWIN_DESC:-twin $TAG (fast-gated)} (file $(shasum -a 256 "$F" | cut -c1-8))"
say "gate done in $(( $(date +%s) - T0 )) s"
grep -q "$TAG SAFE GATE PASS" "$TL" || { say "gate did not pass: no native run"; exit 0; }
yield_to_loop
half() {   # <name> <ranks>: one native process on half of list a
  (cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 \
    $PY -B $R/guard_footprint_v2.py --record $R/runs/quick_${TAG}_$1_guard.json --seconds 7200 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $P2/opus_reexam_20261001_v1/engineer/quick.py \
    $R/variants/${TAG}_safe/candidate.py $2 $R/runs/quick_${TAG}_$1.json > $R/runs/quick_${TAG}_$1.log 2>&1)
  say "native half $1 ended (exit $?)"
}
bash ../gate.sh 4 "fastgate $TAG a1" >> $LOG 2>&1
say "native ten start as 2 processes (load $(uptime | sed 's/.*load averages: //'))"
if [ -e runs/EASE_UP ]; then   # lead 17:5xZ EASE UP: one native process at a time, the halves in sequence
  half a1 651,378,424,328,287; half a2 641,548,557,316,390
else
  half a1 651,378,424,328,287 &
  sleep 60; bash ../gate.sh 4 "fastgate $TAG a2" >> $LOG 2>&1
  half a2 641,548,557,316,390 &
  wait
fi
$PY - $R/runs/quick_${TAG}_a1.json $R/runs/quick_${TAG}_a2.json $R/runs/quick_${TAG}_a.json >> $LOG 2>&1 <<'PYEOF'
import json, shutil, sys
from pathlib import Path
a, b, out = map(Path, sys.argv[1:4])
da, db = json.loads(a.read_text()), json.loads(b.read_text())
assert da['candidate'] == db['candidate'] and not out.exists()
for src, d in ((a, da), (b, db)):
    for r in d['rows']:
        shutil.copyfile(src.with_name(src.stem + f"_pred_{r['rank']}.npy"), out.with_name(out.stem + f"_pred_{r['rank']}.npy"))
out.write_text(json.dumps(dict(candidate=da['candidate'], rows=da['rows'] + db['rows'])))
print(f"merged {len(da['rows'])} + {len(db['rows'])} rows into {out.name}")
PYEOF
say "native ten merged (exit $?)"
if [ -n "$BYTEID_BASE" ]; then CMP=(runs/compare_twin_byteid_v1.py $TAG "$BYTEID_BASE" "$BYTEID_TAG"); else CMP=(runs/compare_twin_native_v5.py $TAG "$BASEQ"); fi
TL="$TL" $PY -B "${CMP[@]}" >> $LOG 2>&1 || \
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) TWIN $TAG DO NOT FIRE: the confirmation compare failed (O/release/$LOG)" >> "$TL"
say "FASTGATE DONE in $(( $(date +%s) - T0 )) s"
