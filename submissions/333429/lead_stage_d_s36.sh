#!/bin/bash
# TAG opus-lead (2026-10-02 01:55 UTC): stage D of the S36 file, launched by the lead through the release lane's own tool because the
# bank has been waiting on this one run; chain13b's shell was stopped (its running session was left to finish). One process at a time:
# waits for any running boundary / serving process of the release lane first.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; E=$PWD/../engineer/stack
while pgrep -f "stage_c_boundary.py|stage_d_serving.py" > /dev/null; do sleep 10; done
date -u +"start %H:%M:%SZ" > runs/lead_d_s36.log
$PY -B stage_d_serving.py $E/candidate_T_C_B_F_D_Hfull_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py d_stackHP_lvfs32_tsr_paced --short-setup >> runs/lead_d_s36.log 2>&1
echo "exit=$? $(date -u +%H:%M:%SZ)" >> runs/lead_d_s36.log
