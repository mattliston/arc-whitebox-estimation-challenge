#!/bin/bash
# usage: score_fresh_candidate_v1.sh <name>   (runs on the laptop; scores both fresh cohorts on the VM paired vs 49bd)
N=$1; O=pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1; F=$O/lead/runs/fable_oct21_final_v1
VM=/private/tmp/claude-501/-Users-matt-arc-white-box-estimation/fa350431-f721-4021-9dc7-6a57a67ff7ed/scratchpad/vm.sh
$VM ssh "cd /home/arcworker/arc-workspace && for c in original8 extension4; do /home/arcworker/arc-envs/judge/bin/python -B $F/score_fresh_paired_v1.py --root /home/arcworker/arc-workspace --plan $F/stage_$N/\${c}_PLAN_v1.json --cand-run /home/arcworker/arc-runs/fresh12_\${c}_${N}_confirmation_v1 --cand-key $N --ref-run /home/arcworker/arc-runs/fresh12_\${c}_first128_paid_nocensus_confirmation_v1 --ref-key first128_paid_nocensus --out /home/arcworker/arc-runs/FRESH_\${c}_${N}_vs_49bd_v1.json 2>&1 | tail -n 1 | cut -c1-200; done; /home/arcworker/arc-envs/judge/bin/python -c \"
import json,statistics,math
rows=[]
for c in ('original8','extension4'): rows+=json.load(open('/home/arcworker/arc-runs/FRESH_%s_${N}_vs_49bd_v1.json'%c))['rows']
v=[r['adjusted_pct'] for r in rows]; print('WHOLE12 ${N} adjusted mean %.4f%% se %.4f lower %d/12 bill %.4f'%(statistics.mean(v),statistics.stdev(v)/math.sqrt(12),sum(x<0 for x in v),statistics.mean(r['bill_pct'] for r in rows))); print([round(x,3) for x in v])\""
