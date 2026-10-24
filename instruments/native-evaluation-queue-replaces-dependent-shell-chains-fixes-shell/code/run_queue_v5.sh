#!/bin/bash
# TAG fable-lead 2026-10-19: FILE-DRIVEN queue runner for the science judge (m15b). Differences from v4: (1) the queue is a FILE of tags
# (one per line, '#' comments) re-read before every launch, so it can be re-prioritised live; (2) the lane count is the number of
# judge processes on the box (any runner), so the four-lane law holds across runners and old drivers can be killed without touching
# their lanes; (3) exits when no eligible tag remains and none of its own lanes runs. usage: run_queue_v5.sh <N lanes total> <ref run dir> <queue file>
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
K=$O/lead/runs/fable_oct19_kc_v1; S=$O/lead/runs/astra_oct18_small_stack_qualification_v1; H=$O/lead/runs/fable_oct19_half_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
N=$1; REF=$2; Q=$3
JUDGE="$S/run_current32_m15b_fable_v1.py"
if [ -f "${Q}.runner" ]; then read -r JUDGE < "${Q}.runner"; fi
[ -f "$JUDGE" ] || exit 2
stamp() { date -u +%H:%M:%SZ; }
target_lanes() {
  local total="$N" reserved=0 file
  if [ -f "${Q}.capacity" ]; then read -r total < "${Q}.capacity"; fi
  case "$total" in 1|2|3|4) ;; *) echo "invalid compute capacity" >&2; return 2;; esac
  [ ! -e "${Q}.single_lane" ] || reserved=$((reserved + 1))
  for file in "${Q}".reserve_*; do [ ! -f "$file" ] || reserved=$((reserved + 1)); done
  total=$((total - reserved))
  [ "$total" -ge 1 ] || total=1
  echo "$total"
}
lanes() { ps -eo command | grep -c '[r]un_current32_.*_fable_v1.py'; }
done_or_dead() { [ -f "$K/science32_$1_m15/RESULT32_v1.json" ] || grep -q Traceback "$K/science32_$1_m15.log" 2>/dev/null; }
report() { echo "$(stamp) $1_DONE tracebacks $(grep -c Traceback $K/science32_$1_m15.log 2>/dev/null)"; grep -A6 Traceback "$K/science32_$1_m15.log" 2>/dev/null | grep -v '^  File' | head -4 | cut -c1-200; grep 'MEAN of' "$K/science32_$1_m15.log" 2>/dev/null | tail -1 | cut -c1-160; [ -f "$K/science32_$1_m15/RESULT32_v1.json" ] && $PY -B $H/compare32_v1.py "$REF" "$K/science32_$1_m15" 2>&1 | tail -3 | cut -c1-200; }
next_tag() { grep -v '^#' "$Q" | sed 's/[[:space:]]*$//' | grep -v '^$' | while read -r t; do [ -f "$K/candidate_${t}_sci_v1.py" ] && [ ! -e "$K/science32_${t}_m15" ] && { echo "$t"; break; }; done; }
running=()
while :; do
  still=()
  for t in "${running[@]}"; do if done_or_dead $t; then report $t; else still+=("$t"); fi; done
  running=("${still[@]}")
  while [ "$(lanes)" -lt "$(target_lanes)" ]; do
    t=$(next_tag); [ -z "$t" ] && break
    C=$K/candidate_${t}_sci_v1.py; SHA=$(shasum -a 256 "$C" | cut -d' ' -f1)
    nohup $PY -B "$JUDGE" "$C" "$SHA" "$K/science32_${t}_m15" > "$K/science32_${t}_m15.log" 2>&1 &
    running+=("$t"); echo "$(stamp) launched $t ${SHA:0:8} (lanes $(lanes)/$(target_lanes); memory $(memory_pressure | tail -1 | grep -o '[0-9]*%'))"
    sleep 25
  done
  [ -z "$(next_tag)" ] && [ ${#running[@]} -eq 0 ] && break
  sleep 45
done
echo "QUEUE_V5_DONE $(stamp)"
