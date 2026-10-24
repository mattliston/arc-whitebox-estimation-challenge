#!/bin/bash
# ONE command, fires ONCE: extgate_s15_cap4 (gate readout refit x1.5 + Smirnov cap 4). Expected ~1.4039e-9, pooled -0.378% on 76 nets.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
test "$(shasum -a 256 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/fable_extgate_s15_cap4_b5a2d638_v1/submission.tar.gz | cut -c1-12)" = "e096a47b1d4f" || { echo "archive sha mismatch"; exit 1; }
date -u
/Users/matt/arc-white-box-estimation/.venv/bin/python -B fire/fire_once.py fable_extgate_s15_cap4_b5a2d638_v1 /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/fable_extgate_s15_cap4_b5a2d638_v1/submission.tar.gz "fable gate readout refit x1.5 + Smirnov read cap 4 on #337506 source; pooled -0.38% paired (Mac32/VM32/fresh12)" 2>&1 | tail -n 2 | cut -c1-300
date -u
