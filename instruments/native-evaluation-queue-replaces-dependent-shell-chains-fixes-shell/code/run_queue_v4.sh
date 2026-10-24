#!/bin/bash
# TAG fable-lead 2026-10-19: N-lane queue runner for the science judge under the 15%-free guard chain (run_current32_m15_fable_v1).
# Keeps up to N judges running from a list of arms; each arm = candidate_<tag>_sci_v1.py -> science32_<tag>_m15; when an arm ends it
# prints its judge line and the paired comparison vs <ref>. usage: run_queue_v3.sh <N> <ref run dir> <tag> [<tag> ...]   (marker QUEUE_V4_DONE)
# (no set -u: macOS bash 3.2 treats an empty array expansion "${running[@]}" as an unbound variable; v2 died at line 31)
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
K=$O/lead/runs/fable_oct19_kc_v1; S=$O/lead/runs/astra_oct18_small_stack_qualification_v1; H=$O/lead/runs/fable_oct19_half_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
N=$1; REF=$2; shift 2
stamp() { date -u +%H:%M:%SZ; }
done_or_dead() { [ -f "$K/science32_$1_m15/RESULT32_v1.json" ] || grep -q Traceback "$K/science32_$1_m15.log" 2>/dev/null; }
report() { echo "$(stamp) $1_DONE tracebacks $(grep -c Traceback $K/science32_$1_m15.log 2>/dev/null)"; grep -A6 Traceback "$K/science32_$1_m15.log" 2>/dev/null | grep -v '^  File' | head -4 | cut -c1-200; grep 'MEAN of' "$K/science32_$1_m15.log" 2>/dev/null | tail -1 | cut -c1-160; [ -f "$K/science32_$1_m15/RESULT32_v1.json" ] && $PY -B $H/compare32_v1.py "$REF" "$K/science32_$1_m15" 2>&1 | tail -3 | cut -c1-200; }
running=()
pending=("$@")
while [ ${#pending[@]} -gt 0 ] || [ ${#running[@]} -gt 0 ]; do
  still=()
  for t in "${running[@]}"; do if done_or_dead $t; then report $t; else still+=("$t"); fi; done
  running=("${still[@]}")
  while [ ${#running[@]} -lt $N ] && [ ${#pending[@]} -gt 0 ]; do
    t=${pending[0]}; pending=("${pending[@]:1}")
    C=$K/candidate_${t}_sci_v1.py
    if [ ! -f "$C" ]; then echo "$(stamp) missing candidate for $t, skipped"; continue; fi
    if [ -e "$K/science32_${t}_m15" ]; then echo "$(stamp) $t: output dir exists, skipped"; continue; fi
    SHA=$(shasum -a 256 "$C" | cut -d' ' -f1)
    nohup $PY -B $S/run_current32_m15b_fable_v1.py "$C" "$SHA" "$K/science32_${t}_m15" > "$K/science32_${t}_m15.log" 2>&1 &
    running+=("$t"); echo "$(stamp) launched $t ${SHA:0:8} (lanes ${#running[@]}/$N; memory $(memory_pressure | tail -1 | grep -o '[0-9]*%'))"
    sleep 20
  done
  sleep 45
done
echo "QUEUE_V4_DONE $(stamp)"
