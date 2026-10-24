#!/bin/bash
# RE-FIRE (only if #337619 fails on SETUP_TIMEOUT and Matt says so): same gated archive 55a93ef002c2, fires ONCE.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/submission.tar.gz
test "$(shasum -a 256 $A | cut -c1-12)" = "55a93ef002c2" || { echo "archive sha mismatch"; exit 1; }
date -u
/Users/matt/arc-white-box-estimation/.venv/bin/python -B fire/fire_once.py fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1 $A "fable RE-FIRE of #337619 after SETUP_TIMEOUT: hg+fh readout + Smirnov cap4 + size-8e9 drop, XSG off" 2>&1 | tail -n 3 | cut -c1-300
date -u
