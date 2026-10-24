#!/bin/bash
# TAG opus-physicist T131 (lead 16:5xZ GATE 1b, after T129): star rank 8 at layers 11-14 only on the S50 frozen point (physicist/t131_worker_ss8late.py):
# control first (T131_SSK=-1 at 11-14 on 7100136-137: must equal T129's base within 1e-4 per layer, else stop), then the arm on 7100136-151; 2 workers at nice
# 10 x 2 threads through the machine gate; base = T129's base runs (same request, same worker). Waits for T129's workers to exit. Stop: STOP_T131.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
P=$O/physicist; LOG=$P/t116_chain.log
while pgrep -f "t129_worker_floors_v12.py" > /dev/null; do sleep 30; done
cd $O/trainer
arm() {   # <run> <K>
  [ -f $P/STOP_T131 ] && { echo "$(date -u +%H:%M:%SZ) T131 STOP before $1" >> $LOG; exit 0; }
  echo "$(date -u +%H:%M:%SZ) T131 $1 start (K $2 at 11-14)" >> $LOG
  for W in 0 1; do
    bash $O/gate.sh 4 physicist_$1_w$W > /dev/null
    env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 \
      TRAINER_BIRTH_ATTACH=0 TRAINER_RSS_LIMIT=7500000000 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
      TRAINER_GENERATED_ALLOW_FILE=../physicist/$1/allowed_seeds.txt T131_SS8=11-14 T131_SSK=$2 nice -n 10 /Users/matt/anaconda3/bin/python -B \
      ../physicist/t131_worker_ss8late.py ../physicist/$1 $W >> ../physicist/$1/worker_w$W.log 2>&1 &
    sleep 5
  done
  wait
  echo "$(date -u +%H:%M:%SZ) T131 $1 done ($(ls ../physicist/$1/results | grep -c 'json$') results, $(ls ../physicist/$1/results | grep -c 'error$') errors)" >> $LOG
}
arm t131_ctl -1
(cd $P && /Users/matt/anaconda3/bin/python -B - <<'PY'
import json, sys
from pathlib import Path
worst = 0.0; n = 0
for s in (7100136, 7100137):
    a = Path('t131_ctl/results/t131_ctl_%d.json' % s); b = Path('t129_floor_raise/results/t129_%d_base.json' % s)
    if not (a.exists() and b.exists()):
        continue
    x, y = json.loads(a.read_text())['mse_layers'], json.loads(b.read_text())['mse_layers']
    worst = max(worst, max(abs(p / q - 1) for p, q in zip(x, y))); n += 1
print('T131 CONTROL (K -1 at 11-14 vs T129 base): networks %d, worst per-layer relative difference %.3e' % (n, worst))
sys.exit(0 if n == 2 and worst <= 1e-4 else 1)
PY
) >> $LOG 2>&1 || { echo "$(date -u +%H:%M:%SZ) T131 STOPPED: control failed" >> $LOG; exit 1; }
arm t131_ss8late 8
