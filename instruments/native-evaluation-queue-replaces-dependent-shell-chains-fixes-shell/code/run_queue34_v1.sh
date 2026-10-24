#!/bin/bash
# TAG fable-lead 2026-10-19: queue3 (dead-unit floor 0.0015 + L13 rank 352, re-run tag r13_352b) then queue4 (L15 row-cut z 3.0 / 3.25)
# on all 32 store ranks under the science judge (rss backstop 7.9 GB), two judges at a time (memory law). Replaces the zsh launcher
# whose `set -- $spec` never split (zsh does not word-split unquoted variables): nothing ran at 08:02Z. bash on purpose.
# usage: nohup bash run_queue34_v1.sh > run_queue34_v1.log 2>&1 &   (markers QUEUE3_DONE / QUEUE4_DONE)
set -u
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
K=$O/lead/runs/fable_oct19_kc_v1; D=$O/lead/runs/fable_oct19_dials_v1; S=$O/lead/runs/astra_oct18_small_stack_qualification_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { date -u +%H:%M:%SZ; }
launch() { # tag candidate
  local t=$1 c=$2 sha
  sha=$(shasum -a 256 "$c" | cut -d' ' -f1)
  echo "$(stamp) launch $t ${sha:0:8} -> science32_${t}_r79"
  nohup $PY -B $S/run_current32_rss79_fable_v1.py "$c" "$sha" "$K/science32_${t}_r79" > "$K/science32_${t}_r79.log" 2>&1 &
}
done_or_dead() { [ -f "$K/science32_$1_r79/RESULT32_v1.json" ] || grep -q Traceback "$K/science32_$1_r79.log" 2>/dev/null; }
summary() {
  $PY -c "import json,sys; r=json.load(open(sys.argv[1])); print(' '.join('%s adj %+.3f%% raw %+.3f%% bill %+.3f%%' % (k, r[k]['adjusted_pct'], r[k].get('raw_pct', float('nan')), r[k].get('bill_pct', float('nan'))) for k in ('heldout25','all32') if k in r))" "$K/science32_$1_r79/RESULT32_v1.json" 2>&1 | tail -1
}
wait_pair() {
  sleep 60
  until done_or_dead $1 && done_or_dead $2; do sleep 60; done
  for t in $1 $2; do echo "$(stamp) $t tracebacks $(grep -c Traceback $K/science32_${t}_r79.log) :: $(summary $t)"; done
}
echo "$(stamp) queue3: dcv0015 + r13_352b; memory $(memory_pressure | tail -1)"
launch dcv0015 $D/candidate_dcv0015_sci_v1.py
launch r13_352b $K/candidate_r13_352_sci_v1.py
wait_pair dcv0015 r13_352b
echo "QUEUE3_DONE $(stamp)"
echo "$(stamp) queue4: l15rc30 + l15rc325; memory $(memory_pressure | tail -1)"
launch l15rc30 $K/candidate_l15rc30_sci_v1.py
launch l15rc325 $K/candidate_l15rc325_sci_v1.py
wait_pair l15rc30 l15rc325
echo "QUEUE4_DONE $(stamp)"
