#!/bin/bash
# [opus-storeval] store-rank validation lane, admission-gated (memory_pressure free >= 55%, lead 18:2xZ), skips done: lane_store_v1.sh <arm> <rank>...
#   arm ctl -> cand_s2tr_kinkoracle_v1.py mode ctl (= S2tr predictions); arm <tag> -> cand_decomp_<tag>_v1.py mode dump (= that candidate's predictions)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1; K=$O/lead/runs/opus_kinkoracle_v1
NPY=$O/../astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python; VPY=/Users/matt/arc-white-box-estimation/.venv/bin/python
for v in OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS VECLIB_NUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS; do export $v=1; done
export PB_ALL=1 PYTHONWARNINGS=ignore
arm=$1; shift
for r in "$@"; do
  if [ "$arm" = ctl ]; then res=$K/res_ctl_$r.json; judge=$K/oracle_judge_v1.py; cand=$K/cand_s2tr_kinkoracle_v1.py; mode=ctl
  else res=$K/res_d_${arm}_$r.json; judge=$K/oracle_judge_v2.py; cand=$K/cand_decomp_${arm}_v1.py; mode=dump; fi
  [ -f $res ] && continue
  while true; do free=$(memory_pressure | awk '/free percentage/ {gsub("%","",$NF); print $NF}'); [ "${free:-0}" -ge 55 ] && break; echo "$(date -u +%H:%M:%SZ) wait $arm $r free=$free" >> $K/admission.log; sleep 20; done
  g=$K/guard_s_${arm}_$r.json; [ -f $g ] && mv $g $g.stale_$(date -u +%H%M%S)
  nice -n 10 $VPY -B $O/release/guard_footprint_v3.py --record $g --seconds 900 --footprint-limit 7500000000 --rss-backstop 10000000000 \
    $NPY -B -W ignore $judge $cand $r $mode $res > $K/log_s_${arm}_$r.txt 2>&1
  echo "$(date -u +%H:%M:%SZ) store $arm $r exit=$?" >> $K/lanes.log
done
echo "$(date -u +%H:%M:%SZ) STORE_DONE $arm" >> $K/lanes.log
