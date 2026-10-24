#!/bin/bash
# TAG opus_engineer a62 (2026-10-11 20:3xZ, lead 20:2xZ queue item 3: aggressive 7-level strip twin pricing): platform-pace request counts
# and bills (networks 35 / 100) of S48 (m_sd_rc = HZ7R, 3% floor, full packets only) with the strip lever swapped, ONE run at a time:
# HZ7 (every packet incl. the partial last one, 3% floor), HZ7g0.0 (every packet, any gain), HZ7Rg0.0 (full packets, any gain).
# Reference: a56_req/runs/ps_s48_m_sd_rc2 (44,222 / 41,971 requests; 251.5696 / 251.4165 G).
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules
for n in s48hz7_sd_rc s48hz7g0_sd_rc s48hz7rg0_sd_rc; do
  echo "== $n $(date -u +%H:%M:%SZ)"
  until bash $A/run_boundary_counts.sh $A/s49/$n/candidate.py ps_$n; do
    [ -e $A/../a56_req/runs/ps_$n ] && break
    sleep 60
  done
done
echo "chain done $(date -u +%H:%M:%SZ)"
