#!/bin/bash
# TAG analyst A13 production: fresh training networks seed = 7,100,000 + k with 64M-sample Monte Carlo truth on the GPU (MPS),
# one network after another from k = $1 (default 0), finished ones skipped. ONE GPU job; it yields to the physicist while the file
# O/gpu_request exists (checked between batches of 32768 samples, i.e. every 0.25 s). About 1.1 GB of GPU memory, 0.1 GB RSS, nice 10.
# Log: fresh/prod.log (one JSON line per network), state: fresh/STATE (rewritten about once a minute).
# Clean stop: touch fresh/STOP (read before each network). Hard stop: kill this script and its python child; finished truths stay.
# If the python process dies (GPU error), it is restarted after 60 s and continues with the first unfinished network.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
F=$O/analyst/fresh; K=${1:-0}
cd $F || exit 1
bash $O/gate.sh 4 analyst
while true; do
  [ -e $F/STOP ] && { echo "$(date -u +%FT%TZ) STOP file present, not starting"; break; }
  echo "$(date -u +%FT%TZ) starting mc_truth.py prod $K"
  nice -n 10 /Users/matt/anaconda3/bin/python $F/mc_truth.py prod $K >> $F/prod.log 2>&1
  code=$?
  echo "$(date -u +%FT%TZ) mc_truth.py exited with code $code"
  [ $code -eq 0 ] && break
  sleep 60
done
