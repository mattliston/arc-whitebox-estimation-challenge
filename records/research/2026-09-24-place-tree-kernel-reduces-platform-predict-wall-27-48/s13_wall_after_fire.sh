#!/usr/bin/env bash
# TAG fable-engineer2: S13's measured expected worst FOR THE RECORD (lead 2026-09-24: run after the S13 fire, low priority; base for the
# engineer's S14 depth re-optimization). Waits for the lead's "FIRED S13" line (or an S13 failure line, or 3 h), then runs the user-CPU pair
# S13 (fable_engineer_v1/candS12ipt3_mem.py 2a5dd324) vs the fired S11 through the porter's runner (30 s lock poll: no queue jumping),
# projects the expected worst with telemetry/project_wall_v2.py and appends the result to fable_refit_v1/TEAM_LOG.md.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
end=$(( $(date +%s) + 3*3600 ))
until grep -q "FIRED S13\|S13.*FAIL" $TL || [ $(date +%s) -ge $end ]; do sleep 30; done
echo "$(date +%T) trigger: $(grep -o 'FIRED S13[^.]*\|S13[^.]*FAIL[^.]*' $TL | tail -1 | cut -c1-80)"
bash $P2/fable_porter_v1/run_usercpu_pair.sh e2_S13 $P2/fable_engineer_v1/candS12ipt3_mem.py e2_S11ref_S13 $P2/fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py
line=$(cd $E/telemetry && /Users/matt/anaconda3/bin/python project_wall_v2.py --pair=S13:optime_e2_S13.json:optime_e2_S11ref_S13.json 2>&1 | grep "^S13 " | tr -s ' ')
echo "$(date +%T) $line"
[ -n "$line" ] && echo "$(date '+%Y-%m-%d %H:%M') [engineer2] S13 WALL MEASURED (for the record; base for S14's depth re-optimization): candS12ipt3_mem.py 2a5dd324 vs fired S11, porter's pair runner, 6 nets concurrent, 1 thread; project_wall_v2 on S11 (97.6 s, kernel 84.91 s), user ratio 0.57/0.75/1.0 = pess/central/opt: $line" >> $TL
