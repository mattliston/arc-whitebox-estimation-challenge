#!/bin/bash
# TAG opus_engineer a56 (2026-10-11, lead 03:5xZ request census): one boundary worker session WITH --sites (every request's call site
# and operation; release lane's boundary_run.py + boundary_client.py used read-only, output in THIS lane), under the engineer's gate and
# the lifecycle guard. Requests are exact; timings are inflated by the site recorder (diagnostic only).
# usage: run_sites.sh <candidate.py (absolute)> <ranks, first = cold> <name (new)>  -> a56_req/runs/<name>/rows.jsonl (by_site, by_func)
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; R=$O/release
cand=$1; ranks=$2; name=$3; out=$E/a56_req/runs/$name
[ -e "$out" ] && { echo "name exists: $name"; exit 2; }
bash $E/ack_lead.sh check || { echo "NOT STARTED $name (unread lead line)"; exit 3; }
while pgrep -f "engineer/quick.py|engineer/census/census2.py|release/boundary_run[.]py" > /dev/null || ! bash $O/idle_gate.sh 30 > /dev/null; do sleep 15; done
bash $O/gate.sh 7 engineer || exit 3
mkdir -p "$out"
cd $R && /Users/matt/arc-white-box-estimation/.venv/bin/python -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py \
  --record $out/guard.json --seconds 2400 --rss-limit 7500000000 \
  /Users/matt/arc-white-box-estimation/.venv/bin/python -B $R/boundary_run.py "$cand" "$ranks" "$out" --sites > $out/run.log 2>&1
echo "exit=$? name=$name"
/Users/matt/arc-white-box-estimation/.venv/bin/python -c "
import json, sys
for l in open('$out/rows.jsonl'):
    r = json.loads(l); print(r['rank'], 'requests', r['requests'], 'flops', r['flops'])" 2>&1 | tail -3
