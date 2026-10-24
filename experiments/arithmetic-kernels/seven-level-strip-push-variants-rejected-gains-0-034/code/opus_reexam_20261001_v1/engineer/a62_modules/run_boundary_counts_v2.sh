#!/bin/bash
# TAG opus_engineer a62: platform-pace REQUEST COUNTS only (release's boundary_run.py --short-setup: the setup clock is emulated from
# requests, so the counts are exact and load-independent; the timings of a loaded box are NOT valid and are not used). Same command as
# a56_req/run_boundary.sh minus its idle-CPU wait (the box is loaded by other lanes' torch workers). Networks 35 (cold) / 100 (warm).
# usage: run_boundary_counts_v2.sh <candidate.py (absolute)> <name (new)> [<networks, default 35,100>]   (v2 = v1 with the network list as an argument)
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; R=$O/release
cand=$1; name=$2; out=$E/a56_req/runs/$name
[ -e "$out" ] && { echo "name exists: $name"; exit 2; }
bash $O/gate.sh 7 engineer || exit 3
mkdir -p "$out"
cd $R && /Users/matt/arc-white-box-estimation/.venv/bin/python -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py \
  --record $out/guard.json --seconds 3600 --rss-limit 7500000000 \
  /Users/matt/arc-white-box-estimation/.venv/bin/python -B $R/boundary_run.py "$cand" ${3:-35,100} "$out" --short-setup > $out/run.log 2>&1
echo "exit=$? name=$name"
/Users/matt/arc-white-box-estimation/.venv/bin/python -c "
import json
for l in open('$out/rows.jsonl'):
    r = json.loads(l); print(r['rank'], 'requests', r['requests'], 'flops', r['flops'])" 2>&1 | tail -3
