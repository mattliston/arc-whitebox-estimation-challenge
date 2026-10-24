#!/bin/bash
# v7 (release, 2026-10-11 21:10 UTC): MODE=primary judges and gates a TWIN (--primary: both counts priced by loss_model_v3 first-priced at
# 7.35 us, limit 20%); the posted P(loss) uses the same first-priced model at 7.35 us (and 1.5% slower); RANKS39=1 adds, AFTER the fire
# line, one candidate-only session on networks 35, 100, 39 and a line with network 39 against the 42,700 warm cap (informational, lead 21:0xZ).
# v6 (release, 2026-10-11 19:25 UTC): v5 reused F (the candidate path) for the fragment list, so with FRAGS=none stage A got no candidate
# argument (s48t: "the following arguments are required: tag"); the fragment list is FR now. Nothing else changed.
# TAG opus-lead S47 SAFE gates v5 (2026-10-11 19:25 UTC) = s47_safe_v4.sh with ONE change: the fragments appended by build_variant come from
# FRAGS (default "l3_bind128 lvfs_limit_off prices_zero", as v4). For a source that ALREADY carries the release fragments (the S48 files are
# built on the fired S47e, which has them) use FRAGS=none: the variant is then the source byte for byte. Reason: v4 on s48/m_sd_rc appended
# the three fragments a second time; the doubled l3_bind128 broke the setup binding (setup 592 requests, first network 49,873).
# TAG opus-release S47 SAFE gates v4 (2026-10-11 18:20 UTC: FIRST_PRICED=1 = the lead's S48 override: the judge and release_gate_v4
# price the first-network count (loss_model_v3, first networks separately) instead of capping it at 43,800; the warm cap stays. v3: default proven source = the fired S47e #335349 file; P(loss) from loss_model_v3's
# 2,500 network factors at the S47 hour (7.31 us a request) and 1.5% slower, plus the 120 s wall risk by the projected mean wall + 2 s (the S47
# runs graded 1.7-1.8 s above their projections); DRY=1 = build + stage A only. v2 = s47_safe_v2.sh, v1 = s47_safe.sh; lead 06:50Z: one safe file, no twin; S46 safe #335259 stays the designated insurance).
# For one engineer file: SAFE = file + Rl3_bind128 + Rlvfs_limit_off + Rprices_zero; stage A against the fired S46 safe source (a new einsum
# signature not in the reviewed list stops here with a TEAM_LOG line); stage C (platform setup clock, constants charged, parent = the S42
# legs); the 1.3x slow-setup session; designation_judge.py; release_gate_v3.py; then ONE TEAM_LOG line: the counts, P(loss) from
# loss_model_v2's network model, and the exact fire_once.py command when the gate passes, else the gate's errors. FALLBACK: when the first
# file's stage C first-network count is above 43,800, the same steps run on the fallback file (lead: S47n fe7f2ab6, about -500 requests).
# One process at a time, nice 10, one thread; stops within 5 s of a QUIET WINDOW OPEN line of the lead.
# usage: bash s47_safe.sh <engineer file> <tag> <fire tag> '<fire description>' [<fallback file> <fallback tag> <fallback fire tag> '<desc>']
#        watch: tail -F O/release/runs/s47safe_<tag>.log
set -u
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; O=$(dirname "$R")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REV="${REV:-alsqj,lsjk,lk->aqsk;a,sq,k->aqsk;kn,kc->nc}"; PROVEN="${PROVEN:-variants/s47eg_safe/candidate.py}"   # v3 default: the fired S47e #335349 source (01103249)
window_open() { [ "$(grep -E '^\[lead\] .{0,40}QUIET WINDOW (OPEN|CLOSED)' ../TEAM_LOG.md | tail -n 1 | grep -Eo 'QUIET WINDOW (OPEN|CLOSED)' | head -n 1)" = "QUIET WINDOW OPEN" ]; }
run() { log=$1; name=$2; pat=$3; shift 3
  if window_open; then echo "$(date -u +%H:%M:%SZ) quiet window open: $name not started, stopping" >> $log; exit 0; fi
  echo "$(date -u +%H:%M:%SZ) start $name (load $(uptime | sed 's/.*load averages: //'))" >> $log
  nice -n 10 "$@" > ${log%.log}_$name.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if window_open; then
      echo "$(date -u +%H:%M:%SZ) quiet window opened: stopping $name" >> $log
      for p in $(pgrep -f "^[^ ]*python[^ ]* -B [^ ]*boundary_run.py .*$pat"); do kill -TERM -- -"$(ps -o pgid= -p $p | tr -d ' ')" 2>/dev/null; done
      pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; exit 0
    fi
    sleep 5
  done
  wait $pid; rc=$?
  echo "$(date -u +%H:%M:%SZ) $name ended (exit $rc): $(tail -n 1 ${log%.log}_$name.log | cut -c1-110)" >> $log
}
gate_one() { SRC=$1; TAG=$2; FTAG=$3; DESC=$4; LOG=runs/s47safe_${TAG}.log; T=${TAG}_safe; F=variants/$T/candidate.py
  [ -e "$LOG" ] && { echo "tag $TAG used already ($LOG)"; exit 2; }
  echo "$(date -u +%H:%M:%SZ) s47_safe $TAG: $SRC sha $(shasum -a 256 "$SRC" | cut -c1-12)" > $LOG
  FR="${FRAGS:-l3_bind128 lvfs_limit_off prices_zero}"; [ "$FR" = none ] && FR=""; $PY -B build_variant.py "$SRC" $T $FR >> $LOG 2>&1 || exit 1
  run $LOG a "$T" $PY -B stage_a_package.py $F a_$T --proven $PROVEN --reviewed "$REV"
  if ! grep -q "\[A package\] PASS" ${LOG%.log}_a.log; then
    echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) $TAG SAFE STOPPED AT STAGE A, review needed: $(tail -n 1 ${LOG%.log}_a.log | cut -c1-700) (O/release/$LOG)" >> ../TEAM_LOG.md
    exit 3
  fi
  [ -n "${DRY:-}" ] && { echo "$(date -u +%H:%M:%SZ) DRY=1: build + stage A only, SAFE DONE" >> $LOG; exit 0; }
  run $LOG c "runs/c_$T/" $PY -B stage_c_boundary.py $F c_$T --short-setup --legs C --parent-from runs/c_S42_bank_legs_heavy
  run $LOG slow "runs/${T}_slow/" $PY -B stage_c_boundary.py $F ${T}_slow --setup-slowdown 1.3 --legs C --parent-from runs/c_S42_bank_legs_heavy
  FP=""; [ -n "${FIRST_PRICED:-}" ] && FP="--first-priced"; [ "${MODE:-}" = primary ] && FP="--primary"
  $PY -B designation_judge.py $T --pair runs/c_$T --slow runs/${T}_slow --worker runs/none $FP > runs/judge_$T.txt 2>&1
  cat runs/judge_$T.txt >> $LOG
  $PY -B release_gate_v4.py $T --stage-a runs/a_$T --pair runs/c_$T --slow runs/${T}_slow --judge runs/judge_$T.txt --allow-missing-worker $FP > runs/gate_$T.json 2>&1
  rc=$?
  FIRST=$($PY -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['requests'])" runs/c_$T/C0/rows.jsonl 2>/dev/null || echo 0)
  WARM=$($PY -c "import json,sys; print(json.loads(open(sys.argv[1]).readlines()[1])['requests'])" runs/c_$T/C0/rows.jsonl 2>/dev/null || echo 0)
  PL=$($PY - "$FIRST" "$WARM" "$T" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import genpareto
first, warm, T = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
x = np.array(json.loads(Path('runs/net_factors_20261011b.json').read_text())['ours'])
proj = json.loads(Path(f'runs/c_{T}/stage_c.json').read_text()).get('projection') or {}
wall = proj.get('mean_wall_s', 0.0) + 2.0
w100 = 0.0 if wall < 85 else 2/359 if wall < 90 else 7/304 if wall < 95 else 9/209 if wall < 100 else 17/187
out = []
for hour, per in (('7.35 us a request', 7.35e-6), ('1.5% slower', 7.46e-6)):
    mw = warm*per; cells = []
    for u in (1.10, 1.12):
        exc = x[x > u] - u; c, _, s = genpareto.fit(exc, floc=0); S = lambda t: len(exc)/len(x)*genpareto.sf(t - u, c, 0, s)
        pf, pw = S(0.4/(mw*first/warm)), S(0.4/mw)
        p100 = 1 - (1 - pf)**6*(1 - pw)**94*(1 - w100); p300 = 1 - (1 - pf)**18*(1 - pw)**282*(1 - w100)**3
        cells.append(f"{100*p100:.1f}% ({100*p300:.1f}% per 300)")
    out.append(f"{hour}: first cap/median {0.4/(mw*first/warm):.3f}, warm {0.4/mw:.3f}: {cells[0]} central, {cells[1]} fat")
print('; '.join(out) + f" (wall part {100*w100:.2f}% per 100 at about {wall:.0f} s mean)")
PYEOF
)
  arch=$(shasum -a 256 runs/a_$T/submission.tar.gz | cut -c1-64); cand=$(shasum -a 256 $F | cut -c1-64)
  nums=$(grep -E '^  (ok  |FAIL) (requests, first|requests, warm|P\(loss\)|projected platform setup|projected worst-network|requests under a setup|binding)' runs/judge_$T.txt | sed 's/^  //' | tr -s ' ' | cut -c1-110 | paste -sd ';' -)
  if [ $rc -eq 0 ]; then
    echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) $TAG SAFE GATE PASS (release_gate_v4${FP:+ with the first-network PRICED override}; stage D not run): candidate sha256 $cand, archive O/release/runs/a_$T/submission.tar.gz sha256 $arch; stage C $FIRST / $WARM requests (first / warm); P(loss) per 100 networks: $PL; judge: $nums. FIRE COMMAND (lead, after stage 2): cd $O && $PY -B fire/fire_once.py $FTAG $R/runs/a_$T/submission.tar.gz '$DESC'" >> ../TEAM_LOG.md
  else
    errs=$($PY -c "import json,sys; print('; '.join(json.load(open(sys.argv[1])).get('errors', ['gate output unreadable']))[:600])" runs/gate_$T.json 2>/dev/null)
    echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) $TAG SAFE: DO NOT FIRE, release_gate_v4 blocked: $errs; stage C $FIRST / $WARM requests (O/release/runs/gate_$T.json)" >> ../TEAM_LOG.md
  fi
  if [ -n "${RANKS39:-}" ]; then
    run $LOG n39 "runs/c_${T}_n39/" $PY -B stage_c_boundary.py $F c_${T}_n39 --short-setup --ranks 35,100,39 --legs C
    N39=$($PY -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; print(' / '.join('%d: %s' % (r['rank'], format(r['requests'], ',')) for r in rows))" runs/c_${T}_n39/C0/rows.jsonl 2>/dev/null || echo missing)
    W39=$($PY -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; print([r['requests'] for r in rows if r['rank'] == 39][0])" runs/c_${T}_n39/C0/rows.jsonl 2>/dev/null || echo 0)
    echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) $TAG NETWORK 39 (informational, the gate stays on network 100 + the priced loss model): one candidate-only session at the platform setup pace on networks 35 (first), 100, 39: $N39 requests; network 39 against the 42,700 warm cap: $([ "$W39" -le 42700 ] 2>/dev/null && echo under || echo OVER) by $((42700 - W39)) (O/release/runs/c_${T}_n39)." >> ../TEAM_LOG.md
  fi
  echo "$(date -u +%H:%M:%SZ) SAFE DONE" >> $LOG
  LAST_FIRST=$FIRST
}
echo $$ > runs/s47safe.pid
LAST_FIRST=0
gate_one "$1" "$2" "$3" "$4"
if [ $# -ge 8 ] && [ "$LAST_FIRST" -gt 43800 ]; then
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) $2: first-network count $LAST_FIRST is above 43,800, so the fallback $6 runs now (lead 06:50Z)." >> ../TEAM_LOG.md
  gate_one "$5" "$6" "$7" "$8"
fi
