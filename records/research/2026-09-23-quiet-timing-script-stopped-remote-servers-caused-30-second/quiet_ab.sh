#!/bin/bash
# TAG fable-engineer2 (E2-Q): quiet-window wall A/B at the production boundary. For each candidate: ONE fresh numpy-free client +
# OpenBLAS server pair (1 thread each), predictions r35 (cold) then r100 in the same process. Reports per net: requests, hidden free
# trips, round trips, server CPU s (local compute), client CPU s, residual s (participant Python, as flopscope bills it), wall s,
# plus the receipts-style compute estimate (wall - 0.30 ms x requests). Alternates candidates; ROUNDS rounds (default 1).
# usage: [NETS=35,100] quiet_ab.sh [ROUNDS] [candidate paths...]   (defaults: nets 35,100; the bank and chain/c4_final_phys.py)
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1
ROUNDS=${1:-1}; shift
CANDS=("$@"); [ ${#CANDS[@]} -eq 0 ] && CANDS=("$E/cand_bank.py" "$E/chain/c4_final_phys.py")
cd $E; STAMP=$(date +%H%M%S); OUT=$E/quiet_ab_$STAMP.log
echo "quiet A/B $STAMP rounds=$ROUNDS candidates=${CANDS[*]}" | tee $OUT
for r in $(seq 1 $ROUNDS); do
  for c in "${CANDS[@]}"; do
    tag=q$(basename $c .py | tr -cd 'A-Za-z0-9_' | cut -c1-24)_r$r
    echo "$(date +%H:%M:%S) load $(uptime | sed 's/.*load averages: //') start $tag" | tee -a $OUT
    ./run_census.sh $c ${NETS:-35,100} $tag 2>&1 | grep "^$tag:" | /Users/matt/anaconda3/bin/python -c "
import sys, json
for line in sys.stdin:
    tag, rest = line.split(': ', 1); d = json.loads(rest)
    print(f\"{tag:34s} r{d['rank']:<4d} req {d['requests']:6d} free {d['free_trips']:6d} rt {d['round_trips']:6d} | server_cpu {d['server_cpu_s']:6.1f}s client_cpu {d['client_cpu_s']:5.1f}s residual {d['residual_s']:.3f}s wall {d['wall_s']:6.1f}s compute_est {d['wall_s'] - 3e-4 * d['requests']:6.1f}s util {d['util']}\")
" | tee -a $OUT
  done
done
echo "QUIET_AB_DONE $OUT" | tee -a $OUT
