#!/bin/bash
# TAG opus-lead (2026-10-02 02:15 UTC): qualification of the S37 file (S36 file + F4 strength 0.6 + K4L age 2, c 1; engineer's sha 2a25a074)
# through the release lane's own tools, one process at a time, right after the lead's stage D of S36 (runs/lead_d_s36.log) ends.
# A (packaging; the three new K4L einsum signatures were read by the release lane in a_stack_T_C_B_F_D_F4v2_K4L_v3: different operands,
# no repeated operand; here the same three with the engineer's index letters), C (two candidate sessions against the stored S35 legs of
# runs/c_stackH_lvfs32_tsr_paced2, platform clock), D (20-network worker, platform clock).
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; E=$PWD/../engineer/stack
S37=$E/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py
until grep -q "^exit=" runs/lead_d_s36.log 2>/dev/null; do sleep 15; done
date -u +"A start %H:%M:%SZ" > runs/lead_s37_chain.log
$PY -B stage_a_package.py $S37 a_s37_F4t06_K4L --reviewed "amk,kj,mj->aj;ij,ij->i;qk,kj,kj->qj" > runs/lead_s37_a.log 2>&1; echo "A exit=$? $(date -u +%H:%M:%SZ)" >> runs/lead_s37_chain.log
grep -q "PASS" runs/lead_s37_a.log || { echo "A did not pass: chain stops" >> runs/lead_s37_chain.log; exit 1; }
$PY -B stage_c_boundary.py $S37 c_s37_paced --short-setup --legs CC --parent-from runs/c_stackH_lvfs32_tsr_paced2 > runs/lead_s37_c.log 2>&1; echo "C exit=$? $(date -u +%H:%M:%SZ)" >> runs/lead_s37_chain.log
$PY -B stage_d_serving.py $S37 d_s37_paced --short-setup > runs/lead_s37_d.log 2>&1; echo "D exit=$? $(date -u +%H:%M:%SZ)" >> runs/lead_s37_chain.log
echo "CHAIN DONE $(date -u +%H:%M:%SZ)" >> runs/lead_s37_chain.log
