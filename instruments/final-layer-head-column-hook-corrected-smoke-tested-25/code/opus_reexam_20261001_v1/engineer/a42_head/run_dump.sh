#!/bin/bash
# TAG opus_engineer a42: one guarded single-thread native run of dump_columns.py (the same gate, guard and interpreter as run_quick.sh;
# ack_lead.sh check first). usage: run_dump.sh <candidate.py> <rank> <name>  -> a42_head/dumps/<name>.npz (+ .log)
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer
cand=$1; rank=$2; name=$3
mkdir -p $E/a42_head/dumps
[ -e "$E/a42_head/dumps/$name.npz" ] && { echo "name exists: $name"; exit 2; }
bash $E/ack_lead.sh check || { echo "NOT STARTED $name"; exit 3; }
while pgrep -f "engineer/quick.py" > /dev/null || ! bash $O/idle_gate.sh 30 > /dev/null; do sleep 15; done
bash $O/gate.sh 7 engineer || exit 3
cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/matt/arc-white-box-estimation/.venv/bin/python -B $P2/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py \
  --record $E/a42_head/dumps/${name}_guard.json --seconds 600 --rss-limit 7500000000 \
  $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $E/a42_head/dump_columns.py "$cand" "$rank" $E/a42_head/dumps/$name.npz \
  > $E/a42_head/dumps/$name.log 2>&1
echo "exit=$? name=$name"; tail -n 2 $E/a42_head/dumps/$name.log | cut -c1-600
