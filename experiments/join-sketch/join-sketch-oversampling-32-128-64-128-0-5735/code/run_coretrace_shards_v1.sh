#!/bin/bash
set -eu
K=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
cd "$K"
trap 'rm -f "$K/queue_v14.txt.reserve_coretrace"' EXIT
while :; do
  lanes=$(ps -eo command | grep -c '[r]un_current32_.*_fable_v1.py' || true)
  extra=0
  [ ! -e queue_v14.txt.reserve_copyall ] || extra=1
  free_pct=$(memory_pressure | tail -1 | sed -n 's/.*: \([0-9][0-9]*\)%.*/\1/p')
  if [ "$((lanes + extra))" -le 4 ] && [ -n "$free_pct" ] && [ "$free_pct" -ge 45 ]; then break; fi
  sleep 5
done
settings=$(cat agent_astra_joint_v1/coretrace_screen_settings_v1.json)
for seed in 7100076 7100077 7100078 7100079; do
  date -u "+%Y-%m-%dT%H:%M:%SZ coretrace native fresh screen $seed"
  "$PY" -B ../astra_crunch_physics_v1/guard_memory15_sci_v2.py \
    --record "coretrace_${seed}_GUARD_v1.json" --seconds 600 \
    --footprint-limit 7900000000 --rss-backstop 9500000000 \
    "$PY" -B eval_fresh_arm_v8.py agent_astra_joint_v1/candidate_coretrace_screen_sci_v1.py \
    "$seed" 1 "$settings" > "eval_coretrace_${seed}_v1.log" 2>&1
  grep -q '^EVAL_DONE$' "eval_coretrace_${seed}_v1.log"
done
"$PY" - <<'PY'
from pathlib import Path
parts=[Path(f'eval_coretrace_{s}_v1.log').read_text()
       for s in range(7100076,7100080)]
assert all('\nEVAL_DONE\n' in s and 'Traceback' not in s for s in parts)
Path('eval_coretrace_fresh_v1.log').write_text('\n'.join(parts))
PY
date -u '+%Y-%m-%dT%H:%M:%SZ coretrace fresh screen complete'
