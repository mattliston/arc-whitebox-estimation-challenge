#!/bin/bash
# TAG opus-release S47 pair, version 1 (2026-10-11, lead's task (a): a fire within 60 minutes of a built candidate; only what protects
# the score). From the engineer's two built files:
#   SAFE = <safe file> + Rl3_bind128 + Rlvfs_limit_off + Rprices_zero      (the S44-S46 recipe)
#   TWIN = <twin file> as built by the engineer (+ the fragments in TWIN_FRAGS, default none: his OVR@ block is appended after release
#          fragments, so Rprices_zero would be undone on his kernel-price base)
# Steps, one process at a time, nice 10, one thread: builds; stage A of both against the LAST FIRED sources (S46 safe / twin), so only
# genuinely new einsum signatures are listed: if any is not in the reviewed list, the script writes them to TEAM_LOG and STOPS for the
# lead's review (rerun with the 4th argument); then per file: stage C (platform setup clock, constants charged, parent = the S42
# legs), the 1.3x slow-setup session, designation_judge.py (twin in primary mode: P(loss) <= 20% instead of the request caps), and
# release_gate_v3.py (hashes, archive, every stage A / C gate, fresh judge; this machine's swap / paging only warn); a gate exit 0
# writes ONE fire line to TEAM_LOG, otherwise a DO NOT FIRE line with the gate's errors. Stage D is not run (no designation work).
# Expected: about 35 min on a quiet machine, 60 at load 25. Stops within 5 s of a QUIET WINDOW OPEN line of the lead.
# usage: [TWIN_FRAGS='...'] [DRY=1] bash s47_pair.sh <engineer safe file> <engineer twin file> <tag> ['reviewed sig;sig']
#        DRY=1: builds + stage A only.   watch: tail -F O/release/runs/s47pair_<tag>.log
set -u
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SAFE_SRC="$1"; TWIN_SRC="$2"; TAG="$3"; REV="${4:-alsqj,lsjk,lk->aqsk;a,sq,k->aqsk;kn,kc->nc}"; LOG=runs/s47pair_${TAG}.log
PROVEN_SAFE=variants/s46_63f89e8d_safe_pz/candidate.py; PROVEN_TWIN=variants/s46e_f89b8951_primary/candidate.py   # fired as #335259 / #335261
for f in "$SAFE_SRC" "$TWIN_SRC"; do [ -f "$f" ] || { echo "no such file: $f"; exit 2; }; done
[ -e "$LOG" ] && { echo "tag $TAG used already ($LOG)"; exit 2; }
echo $$ > runs/s47pair_${TAG}.pid
window_open() { [ "$(grep -E '^\[lead\] .{0,40}QUIET WINDOW (OPEN|CLOSED)' ../TEAM_LOG.md | tail -n 1 | grep -Eo 'QUIET WINDOW (OPEN|CLOSED)' | head -n 1)" = "QUIET WINDOW OPEN" ]; }
step() { name=$1; pat=$2; shift; shift
  if window_open; then echo "$(date -u +%H:%M:%SZ) quiet window open: $name not started, stopping" >> $LOG; exit 0; fi
  echo "$(date -u +%H:%M:%SZ) start $name (load $(uptime | sed 's/.*load averages: //'))" >> $LOG
  nice -n 10 "$@" > runs/s47pair_${TAG}_${name}.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if window_open; then
      echo "$(date -u +%H:%M:%SZ) quiet window opened: stopping $name" >> $LOG
      for p in $(pgrep -f "^[^ ]*python[^ ]* -B [^ ]*(boundary_run|native_probe).py .*$pat"); do kill -TERM -- -"$(ps -o pgid= -p $p | tr -d ' ')" 2>/dev/null; done
      pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; exit 0
    fi
    sleep 5
  done
  wait $pid; rc=$?
  echo "$(date -u +%H:%M:%SZ) $name ended (exit $rc): $(tail -n 1 runs/s47pair_${TAG}_${name}.log | cut -c1-110)" >> $LOG
  return $rc
}
echo "$(date -u +%H:%M:%SZ) s47_pair $TAG: safe $SAFE_SRC sha $(shasum -a 256 "$SAFE_SRC" | cut -c1-12); twin $TWIN_SRC sha $(shasum -a 256 "$TWIN_SRC" | cut -c1-12); twin fragments: ${TWIN_FRAGS:-none}" > $LOG
$PY -B build_variant.py "$SAFE_SRC" ${TAG}_safe l3_bind128 lvfs_limit_off prices_zero >> $LOG 2>&1 || exit 1
if [ -n "${TWIN_FRAGS:-}" ]; then $PY -B build_variant.py "$TWIN_SRC" ${TAG}_twin $TWIN_FRAGS >> $LOG 2>&1 || exit 1
else mkdir variants/${TAG}_twin && cp "$TWIN_SRC" variants/${TAG}_twin/candidate.py && echo "{\"source\": \"$TWIN_SRC\", \"as_is\": true}" > variants/${TAG}_twin/build.json; fi
new=""
for who in safe twin; do
  proven=$PROVEN_SAFE; [ $who = twin ] && proven=$PROVEN_TWIN
  step a_$who "${TAG}_$who" $PY -B stage_a_package.py variants/${TAG}_$who/candidate.py a_${TAG}_$who --proven $proven --reviewed "$REV"
  grep -q "\[A package\] PASS" runs/s47pair_${TAG}_a_$who.log || new="$new $who: $(tail -n 1 runs/s47pair_${TAG}_a_$who.log | grep -o "new vs proven source: .*" | cut -c1-600)"
done
if [ -n "$new" ]; then
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) S47 PAIR $TAG STOPPED AT STAGE A, review needed:$new (O/release/runs/s47pair_${TAG}.log); rerun: bash O/release/s47_pair.sh <safe> <twin> ${TAG}b '<reviewed sig;sig>'" >> ../TEAM_LOG.md
  echo "$(date -u +%H:%M:%SZ) STOPPED at stage A" >> $LOG; exit 3
fi
[ -n "${DRY:-}" ] && { echo "$(date -u +%H:%M:%SZ) DRY=1: builds and stage A only, PAIR DONE" >> $LOG; exit 0; }
for who in safe twin; do
  F=variants/${TAG}_$who/candidate.py; T=${TAG}_$who; mode=""; [ $who = twin ] && mode="--primary"
  step c_$who "runs/c_$T/" $PY -B stage_c_boundary.py $F c_$T --short-setup --legs C --parent-from runs/c_S42_bank_legs_heavy
  step slow_$who "runs/${T}_slow/" $PY -B stage_c_boundary.py $F ${T}_slow --setup-slowdown 1.3 --legs C --parent-from runs/c_S42_bank_legs_heavy
  $PY -B designation_judge.py $T --pair runs/c_$T --slow runs/${T}_slow --worker runs/none $mode > runs/judge_$T.txt 2>&1
  cat runs/judge_$T.txt >> $LOG
  $PY -B release_gate_v3.py $T --stage-a runs/a_$T --pair runs/c_$T --slow runs/${T}_slow --judge runs/judge_$T.txt --allow-missing-worker $mode > runs/gate_$T.json 2>&1
  rc=$?
  arch=$(shasum -a 256 runs/a_$T/submission.tar.gz | cut -c1-64); cand=$(shasum -a 256 $F | cut -c1-64)
  nums=$(grep -E '^  ok   (requests, first|requests, warm|P\(loss\)|projected platform setup|projected worst-network|requests under a setup)' runs/judge_$T.txt | sed 's/^  ok   //' | tr -s ' ' | cut -c1-110 | paste -sd ';' -)
  if [ $rc -eq 0 ]; then
    echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) FIRE LINE ${TAG} $(echo $who | tr a-z A-Z): archive O/release/runs/a_$T/submission.tar.gz, archive sha256 $arch, candidate sha256 $cand; release_gate_v3 PASS (stage D not run); judge: $nums" >> ../TEAM_LOG.md
  else
    errs=$($PY -c "import json,sys; print('; '.join(json.load(open(sys.argv[1])).get('errors', ['gate output unreadable']))[:500])" runs/gate_$T.json 2>/dev/null)
    echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${TAG} $(echo $who | tr a-z A-Z): DO NOT FIRE, release_gate_v3 blocked: $errs (O/release/runs/gate_$T.json)" >> ../TEAM_LOG.md
  fi
done
echo "$(date -u +%H:%M:%SZ) PAIR DONE" >> $LOG
