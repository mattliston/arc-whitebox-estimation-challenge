#!/bin/zsh
# TAG fable-physicist (EW-1): four no-refit Edgeworth birth arms on VALID16, two at a time (3 threads each, nice 10).
cd "$(dirname "$0")"
export MallocLargeCache=0 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/Users/matt/anaconda3/bin/python3
echo "$(date '+%F %T') start" >> arms.log
nice -n 10 $PY ew_eval.py 1 1 valid:16 3 ew_o1_s1 > ew_o1_s1.log 2>&1 &
nice -n 10 $PY ew_eval.py 0.5 1 valid:16 3 ew_o1_s05 > ew_o1_s05.log 2>&1 &
wait
echo "$(date '+%F %T') order-1 arms done" >> arms.log
nice -n 10 $PY ew_eval.py 1 2 valid:16 3 ew_o2_s1 > ew_o2_s1.log 2>&1 &
nice -n 10 $PY ew_eval.py 0.5 2 valid:16 3 ew_o2_s05 > ew_o2_s05.log 2>&1 &
wait
echo "$(date '+%F %T') all arms done" >> arms.log
