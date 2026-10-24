#!/bin/bash
# TAG fable-engineer2 (E2-G): the full engineer2 fire gate for one candidate against a reference build, run back to back, each
# step waiting until no quiet window (fable_engineer_v1/quiet_pairs_v2.sh) is running:
#   1. memory + handles: run_mem_fine.sh over 6 predictions in one process (r35, r100, r37, r39, r35, r100): exact live-array peak
#      per prediction, peak site (op, client call stack, 25 largest buffers), handles, server peak RSS;
#   2. paired-concurrent server CPU vs the reference (pair_cpu.sh, 2 rounds, r35 cold then r100).
# usage: gate_fire.sh <cand.py abs> <ref.py abs> <tag>     -> gate_<tag>.log (summary) + the usual per-step files
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1
CAND=$1; REF=$2; TAG=$3; OUT=$E/gate_$TAG.log; cd $E
waitq() { while pgrep -f "[q]uiet_pairs_v2" >/dev/null; do sleep 10; done; }
echo "gate $TAG cand $CAND ($(shasum -a 256 $CAND | cut -c1-16)) ref $REF ($(shasum -a 256 $REF | cut -c1-16)) $(date '+%F %T')" | tee $OUT
waitq; echo "$(date +%T) memory: run_mem_fine.sh 6 predictions, load $(uptime | sed 's/.*load averages: //')" | tee -a $OUT
bash $E/run_mem_fine.sh $CAND 35,100,37,39,35,100 gm$TAG > $E/gm$TAG.out 2>&1
grep "server peak RSS\|live-array peak" $E/gm$TAG.out | tee -a $OUT
python3 - "$E/memwin_gm$TAG.jsonl" <<'PY' | tee -a $OUT
import json, sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    print(f"  {d['window']:8s} peak {d['peak_live_MB']:8.1f} MB  live after {d['live_at_end_MB']:8.1f} MB  handles {d['handles']:6d}  at {d['peak_op']} {d['peak_site'].split('|')[-1].split('<')[0]}")
PY
waitq; echo "$(date +%T) paired server CPU vs ref (pair_cpu.sh, 2 rounds)" | tee -a $OUT
bash $E/pair_cpu.sh g$TAG 2 $REF $CAND > $E/pair_cpu_g$TAG.out 2>&1
grep -v "^pair_cpu\|PAIR_CPU_DONE" $E/pair_cpu_g$TAG.log | tee -a $OUT
echo "GATE_DONE $(date +%T)" | tee -a $OUT
