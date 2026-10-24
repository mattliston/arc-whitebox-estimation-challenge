#!/bin/bash
# TAG opus_engineer a62 S48 chain v6. At 17:39:40Z the lifecycle guards stopped five of my runs together (system swap grew by more than
# 256 MiB with too many processes on a loaded box): the sealed S47e reference arm (2 of 32 done, archived, nothing of S48 seen), the MS
# certificate, the column stack's count session and the two MS parity runs (all in _stale/swap_kill_1739). From here at most TWO
# processes of mine at once, and only the sealed pair runs two. Waits for the pid given as $1 (the MS + star count session), then:
# (1) the column stack's platform-pace counts again (ps_s48_m_sd_rc2), alone; (2) the MS certificate again on 7100136 (s48/m_off: the MS
# fragment then Nd's literals: must give the fired sha 265d607d), alone; (3) waits for a62_modules/S48_CHOICE.txt (written by hand from
# the counts: MS + star + column if it fits 43,800 / 42,700, else MS + star); (4) the single sealed look: S47e and the choice on
# 7100328-359, two processes; (5) MS parity runs on public 651 / 378 (quick.py, fired and s48/m), one at a time.
# Watch: tail -F a62_modules/chain_s48_v6.out
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules
while kill -0 $1 2>/dev/null; do sleep 10; done
echo "$(date -u +%H:%M:%SZ) MS + star counts done"
run() {   # <candidate> <seeds> <name>
  [ -e "$A/gate_rdz_$3.log" ] && { echo "exists: $3"; return; }
  bash $O/gate.sh 7 engineer || exit 3
  cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    /Users/matt/arc-white-box-estimation/.venv/bin/python -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py \
    --record $A/gate_rdz_$3_guard.json --seconds 14400 --rss-limit 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py "$1" none none $2 $A/gate_rdz_$3.json \
    > $A/gate_rdz_$3.log 2>&1
  echo "$(date -u +%H:%M:%SZ) exit=$? $3"
}
bash $A/run_boundary_counts.sh $A/s48/m_sd_rc/candidate.py ps_s48_m_sd_rc2
run $A/s48/m_off/candidate.py 7100136 s48_m_off_cert2
until [ -s $A/S48_CHOICE.txt ]; do sleep 15; done
CHOICE=$(head -1 $A/S48_CHOICE.txt)
NAME=$(basename $(dirname $CHOICE))
echo "$(date -u +%H:%M:%SZ) choice $CHOICE ($NAME)"
SEALED=$(seq -f %.0f 7100328 7100359 | paste -sd, -)
run $E/stack/rl/s47rsd3x_safe/candidate.py $SEALED sealed_s47e2 &
sleep 60
run $CHOICE $SEALED sealed_$NAME &
wait
echo "$(date -u +%H:%M:%SZ) sealed look done"
PER_NET=600 bash $E/run_quick.sh $E/stack/rl/s47rsd3x_safe/candidate.py 651,378 s47e_fired_651_378b
PER_NET=600 bash $E/run_quick.sh $A/s48/m/candidate.py 651,378 s48m_651_378b
echo "$(date -u +%H:%M:%SZ) chain done"
