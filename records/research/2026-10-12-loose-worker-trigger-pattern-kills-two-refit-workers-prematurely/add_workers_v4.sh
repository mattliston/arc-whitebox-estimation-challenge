#!/bin/bash
# TAG opus-trainer RAD S51 v4 (lead 06:03Z: v3 misfired on loose patterns): add S51 workers 2 / 3 (4 threads) only when release's lines carry the EXACT
# strings 'TWIN s50tw (' + 'VERDICT:' (compare_twin_native_v2.py) AND 's50s3 SAFE GATE PASS' or 's50s3 SAFE: DO NOT FIRE', or at 08:00Z at the latest;
# then memory_pressure free >= 60% for 5 consecutive minutes; logs one TEAM_LOG line. Ends without launching if S51's run has a STOP file.
cd "$(dirname "$0")/../../.."
O=..; RUN=run_RAD_S51_gen
LATEST=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-12T08:00:00Z" +%s)
twin() { grep '^\[release\] ' $O/TEAM_LOG.md | grep -F 'TWIN s50tw (' | grep -qF 'VERDICT:'; }
safe3() { grep '^\[release\] ' $O/TEAM_LOG.md | grep -qF -e 's50s3 SAFE GATE PASS' -e 's50s3 SAFE: DO NOT FIRE'; }
until { twin && safe3; } || [ $(date +%s) -ge $LATEST ]; do sleep 60; [ -e $RUN/STOP ] && exit 0; done
ok=0; until [ $ok -ge 5 ]; do mp=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%'); [ "$mp" -ge 60 ] && ok=$((ok + 1)) || ok=0; sleep 60; [ -e $RUN/STOP ] && exit 0; done
for i in 2 3; do bash $O/gate.sh 6 trainer_S51_w$i > /dev/null && (nohup bash cert/RAD/launch_gen_v7_t4.sh $i S51 > /dev/null 2>&1 &); sleep 20; done
(bash cert/RAD/swap_guard_v3.sh $RUN 2 3 >> cert/RAD/S51/swap_guard_v3.log 2>&1 &)
echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) S51 at 4 workers (exact release lines 'TWIN s50tw ( ... VERDICT:' + 's50s3 SAFE ...' seen, or 08:00Z; memory_pressure free >= 60% for 5 min): workers 2 / 3 added at 4 threads, swap guard v3 on them." >> $O/TEAM_LOG.md
