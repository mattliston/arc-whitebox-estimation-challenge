#!/usr/bin/env bash
# TAG fable-physicist WEIGHT-AVERAGING probe (lead GO 09-25 ~02:40): no-refit VALID16 on the S24-structure torch card (s6bcard_loop5j;
# fine_ckpt=False + pair_series loop form), CPU float64. Arms: A = avg(R3,R4) params + avg dials; B = avg params + R3 dials;
# C = 0.3 R3 + 0.7 R4 params and dials; D = R4 params + R4 dials (reference). Baseline S24 (R3) = c0scan/c0scan_S24 (same card/driver).
cd "$(dirname "$0")/.."
E="nice -n 10 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MallocLargeCache=0 python3 -u refit_mps_v3j2.py --device cpu --eval-only --threads 3 --card s6bcard_loop5j --outdir wavg"
run() { bash ../../fable_porter_v1/machine_gate.sh 7 "wavg $1"; $E --tag wavg_$1 --resume $2 --dials $3 2>&1 | grep --line-buffered '"eval"\|Traceback\|Error'; }
( run A wavg/params_avg_R3R4_w50.pt wavg/dials_avg_R3R4_w50.json; run C wavg/params_avg_R3R4_w30.pt wavg/dials_avg_R3R4_w30.json ) > wavg/streamA.log 2>&1 &
( run B wavg/params_avg_R3R4_w50.pt s6b_loopR3_joint_s23/dials_epoch0.json; run D s6b_loopR4_joint_s23/params_epoch0.pt s6b_loopR4_joint_s23/dials_epoch0.json ) > wavg/streamB.log 2>&1 &
wait; echo "wavg done $(date +%T)"
