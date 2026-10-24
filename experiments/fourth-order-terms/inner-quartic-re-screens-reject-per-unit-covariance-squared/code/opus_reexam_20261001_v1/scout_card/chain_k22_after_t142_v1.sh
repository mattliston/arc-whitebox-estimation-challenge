#!/bin/bash
# TAG scout-card (2026-10-13 11:0xZ, lead 10:3xZ: my re-screens run on my 2 slots once the physicist's T142 frees them): waits for the line
# 'T142 ALL DONE' in O/physicist/t116_chain.log (read only) and no worker on physicist/t142_k4l_scan, then runs run_k22 (worker_sc11 = otr_m_sc7) on 2
# workers. Abort: touch O/scout_card/NO_K22.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
S=$O/scout_card
until grep -q 'T142 ALL DONE' $O/physicist/t116_chain.log 2>/dev/null; do [ -f $S/NO_K22 ] && exit 0; sleep 30; done
while pgrep -f "worker_sc8.py $O/physicist/t142_k4l_scan" > /dev/null; do sleep 10; done
[ -f $S/NO_K22 ] && exit 0
echo "$(date -u +%FT%TZ) T142 done; starting run_k22 (2 workers, worker_sc11)"
cd $S && bash k1_chain_v13.sh run_k22 2 14400 > run_k22/chain.out 2>&1
