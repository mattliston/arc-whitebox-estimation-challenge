#!/bin/bash
# TAG opus-release twin pipeline (2026-10-12 00:45 UTC; lead: the S48 twin tonight with its own prebind list). For the engineer's twin file:
# (1) fragment check: exactly one Rl3_bind128 and one Rprices_zero, no Rlvfs_limit_off (the doubling bit us once), else stop with a line;
# (2) gate: s47_safe_v7.sh MODE=primary RANKS39=1 FRAGS=none against the fired S48 source, F4C v3 signatures reviewed (writes its own
#     gate line; NOT the go); (3) ONE native process: the engineer's quick.py on list a (ten networks) into O/release/runs; (4) the comparison
#     against his native S48 session on the same networks, one TEAM_LOG line ending in VERDICT ... GO + the fire command, or DO NOT FIRE.
# v3 (2026-10-12 07:15 UTC): BASESRC=<the safe file of the SAME physics>: when BASEQ does not exist yet, its native base session is run
# first (one process, before the twin's own native run) into BASEQ (default runs/quick_<tag>_base_a.json). v2 = twin_pipeline_v2.sh.
# v2 (2026-10-12 04:50 UTC): BASEQ=<quick json of the safe file of the SAME physics> (default the S48 session) and PROVEN env; v1 = twin_pipeline.sh.
# usage: [BASEQ=..] [PROVEN=..] bash twin_pipeline_v3.sh <twin file .py> <tag> <fire tag>      watch: tail -F O/release/runs/s47safe_<tag>.log O/release/runs/twinpipe_<tag>.log
set -u
cd "$(dirname "$0")"; R=$PWD; P2=$(cd ../.. && pwd); PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
F=$1; TAG=$2; FT=$3; LOG=runs/twinpipe_${TAG}.log
[ -f "$F" ] || { echo "no such file $F"; exit 2; }
[ -e "$LOG" ] && { echo "tag $TAG used already"; exit 2; }
c() { grep -c "^# BEGIN OPUS-RELEASE FRAGMENT $1 " "$F"; }
echo "$(date -u +%H:%M:%SZ) twin $F sha $(shasum -a 256 "$F" | cut -c1-12): l3_bind128 x$(c l3_bind128), prices_zero x$(c prices_zero), lvfs_limit_off x$(c lvfs_limit_off), lvfuse_sized32 x$(c lvfuse_sized32), tsr_off x$(c tsr_off); a62 items: $(grep -o '^# BEGIN OPUS-ENGINEER A62 FRAGMENT [a-z0-9_]*' "$F" | awk '{print $NF}' | paste -sd ',' -)" > $LOG
if [ "$(c l3_bind128)" != 1 ] || [ "$(c prices_zero)" != 1 ] || [ "$(c lvfs_limit_off)" != 0 ]; then
  echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) TWIN $TAG STOPPED AT THE FRAGMENT CHECK: $(tail -n 1 $LOG | cut -c1-400) (needs exactly one Rl3_bind128 and one Rprices_zero and no Rlvfs_limit_off)" >> ../TEAM_LOG.md; exit 3
fi
FRAGS=none MODE=primary RANKS39=1 PROVEN=${PROVEN:-../engineer/a62_modules/s48/m_sd_rc/candidate.py} REV='alsqj,lsjk,lk->aqsk;a,sq,k->aqsk;kn,kc->nc;als,lsjk,lk->asjk;a,sj,k->asjk' \
  bash s47_safe_v7.sh "$F" $TAG $FT "${TWIN_DESC:-twin (aggressive half of the pair, own prebind list): no fused trees + F4Cv3/K4Lv7 + Rl3_bind128 + Rprices_zero} (file $(shasum -a 256 "$F" | cut -c1-8))"
grep -q "SAFE DONE" runs/s47safe_${TAG}.log || { echo "$(date -u +%H:%M:%SZ) gate run did not finish: stop" >> $LOG; exit 0; }
grep -q "$TAG SAFE GATE PASS" ../TEAM_LOG.md || { echo "$(date -u +%H:%M:%SZ) gate blocked: no native check" >> $LOG; exit 0; }
BASEQ=${BASEQ:-${BASESRC:+runs/quick_${TAG}_base_a.json}}
if [ -n "${BASESRC:-}" ] && [ ! -f "$BASEQ" ]; then
  BASESRC_ABS=$(cd "$(dirname "$BASESRC")" && pwd)/$(basename "$BASESRC")
  echo "$(date -u +%H:%M:%SZ) base session start: $BASESRC (sha $(shasum -a 256 "$BASESRC" | cut -c1-12)) -> $BASEQ" >> $LOG
  (cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 \
    $PY -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py --record $R/${BASEQ%.json}_guard.json --seconds 3000 \
    --rss-limit 7500000000 $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $P2/opus_reexam_20261001_v1/engineer/quick.py \
    $BASESRC_ABS 651,378,424,328,287,641,548,557,316,390 $R/$BASEQ > $R/${BASEQ%.json}.log 2>&1)
  echo "$(date -u +%H:%M:%SZ) base session ended (exit $?)" >> $LOG
fi
echo "$(date -u +%H:%M:%SZ) quick start (load $(uptime | sed 's/.*load averages: //'))" >> $LOG
cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 \
  $PY -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py --record $R/runs/quick_${TAG}_a_guard.json --seconds 3000 \
  --rss-limit 7500000000 $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $P2/opus_reexam_20261001_v1/engineer/quick.py \
  $R/variants/${TAG}_safe/candidate.py 651,378,424,328,287,641,548,557,316,390 $R/runs/quick_${TAG}_a.json > $R/runs/quick_${TAG}_a.log 2>&1
echo "$(date -u +%H:%M:%SZ) quick ended (exit $?)" >> $R/$LOG
cd $R && $PY -B runs/compare_twin_native_v2.py $TAG ${BASEQ:-../engineer/logs/quick_keys_s48rc_a.json} >> $LOG 2>&1
echo "$(date -u +%H:%M:%SZ) TWIN PIPELINE DONE" >> $LOG
