#!/bin/bash
# TAG opus-trainer RAD S51 v3 (lead 05:5xZ: release's 05:56Z BOX FREE was premature): add S51 workers 2 / 3 (4 threads) only when TEAM_LOG has BOTH
# release's S50 TWIN verdict line AND release's s50s3 gate line (or at 08:00Z at the latest), and system memory_pressure free >= 60% over 5 consecutive
# minutes; logs one TEAM_LOG line. Ends without launching if S51's run has a STOP file.
cd "$(dirname "$0")/../../.."
O=..; RUN=run_RAD_S51_gen
LATEST=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-12T08:00:00Z" +%s)
twin() { grep -iE "^\[release\] 2026-10-1[2-6]T[0-9:]*Z.*(S50.*TWIN|s50t[0-9]).*(VERDICT|FIRE|PASS|FAIL|BLOCKED)" $O/TEAM_LOG.md | grep -qv "S50 ORDER"; }
safe3() { grep -iE "^\[release\] 2026-10-1[2-6]T[0-9:]*Z.*s50s3.*(VERDICT|FIRE|PASS|FAIL|BLOCKED|gate)" $O/TEAM_LOG.md > /dev/null; }
until { twin && safe3; } || [ $(date +%s) -ge $LATEST ]; do sleep 60; [ -e $RUN/STOP ] && exit 0; done
ok=0; until [ $ok -ge 5 ]; do mp=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%'); [ "$mp" -ge 60 ] && ok=$((ok + 1)) || ok=0; sleep 60; [ -e $RUN/STOP ] && exit 0; done
for i in 2 3; do bash $O/gate.sh 6 trainer_S51_w$i > /dev/null && (nohup bash cert/RAD/launch_gen_v7_t4.sh $i S51 > /dev/null 2>&1 &); sleep 20; done
(bash cert/RAD/swap_guard_v3.sh $RUN 2 3 >> cert/RAD/S51/swap_guard_v3.log 2>&1 &)
echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) S51 at 4 workers (release's S50 twin verdict + s50s3 gate lines seen, or 08:00Z; memory_pressure free >= 60% for 5 min): workers 2 / 3 added at 4 threads, swap guard v3 on them." >> $O/TEAM_LOG.md
