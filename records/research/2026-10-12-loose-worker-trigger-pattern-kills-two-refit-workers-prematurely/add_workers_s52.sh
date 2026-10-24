#!/bin/bash
# TAG opus-trainer RAD S52 (lead 07:1xZ): add S52 workers 2 / 3 (4 threads, 8M-capable launchers) only when release's lines carry S51's gate verdicts
# (lane-anchored: 'TWIN s51tw (' with 'VERDICT:', AND 's51s<k> SAFE GATE PASS' or 's51s<k> SAFE: DO NOT FIRE') or 3 h after S52's start at the latest,
# then memory_pressure free >= 60% for 5 consecutive minutes; logs one TEAM_LOG line. Ends without launching if S52's run has a STOP file.
# usage (started by chain_s52_v2.sh): bash cert/RAD/S52/add_workers_s52.sh <S52 start epoch>
cd "$(dirname "$0")/../../.."
O=..; RUN=run_RAD_S52_gen; LATEST=$(( $1 + 10800 ))
twin() { grep '^\[release\] ' $O/TEAM_LOG.md | grep -F 'TWIN s51tw (' | grep -qF 'VERDICT:'; }
safe() { grep '^\[release\] ' $O/TEAM_LOG.md | grep -qE 's51s[0-9]+ SAFE (GATE PASS|: DO NOT FIRE)'; }
until { twin && safe; } || [ $(date +%s) -ge $LATEST ]; do sleep 60; [ -e $RUN/STOP ] && exit 0; done
ok=0; until [ $ok -ge 5 ]; do mp=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%'); [ "$mp" -ge 60 ] && ok=$((ok + 1)) || ok=0; sleep 60; [ -e $RUN/STOP ] && exit 0; done
for i in 2 3; do bash $O/gate.sh 6 trainer_S52_w$i > /dev/null && (nohup bash cert/RAD/launch_gen_v8_t4.sh $i S52 > /dev/null 2>&1 &); sleep 20; done
(bash cert/RAD/swap_guard_v3.sh $RUN 2 3 >> cert/RAD/S52/swap_guard_v3.log 2>&1 &)
echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) S52 at 4 workers (release's S51 gate lines seen, or 3 h after S52's start; memory_pressure free >= 60% for 5 min): workers 2 / 3 added at 4 threads, swap guard v3 on them." >> $O/TEAM_LOG.md
