#!/bin/bash
# TAG scout-card (2026-10-13 03:2xZ, lead 03:1xZ: the gbsm05L5_9 extension split across both workers): replaces auto_k14_v1.sh after its band
# tables (gbsm05L12_14 +0.072%, gbsp1L10_11 +0.051%: no extension; gbsm05L5_9 -0.421%: extended). Posts, as they land:
# 1) gk+gbsm05L5_9 vs gk at n = 24 (7100584-607) once the 16 extension evals exist, wherever they ran (k72f on worker 2 = run_k7, 7100600-607;
#    k5zy on worker 1 = run_k6, 7100592-599), plus the 16 alone;
# 2) gk+gbsm10L5_9 (k72g, 7100584-591) vs gk and vs gk+gbsm05L5_9.
# Markers in O/scout_card/posted_k14 (shared with v1).
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
S=$O/scout_card; PY="/usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/matt/anaconda3/bin/python"
post() { echo "[scout-card] $(date -u +%FT%TZ) $1" >> $O/TEAM_LOG.md; }
M=$S/posted_k14; mkdir -p $M
cnt() { ls $S/run_k6/results $S/run_k7/results 2>/dev/null | grep -cF "_$1.json"; }
R=$S/run_k1,$S/run_k6,$S/run_k7
while [ ! -f $S/NO_AUTOPOST ]; do
  if [ ! -f $M/gk_gbsm05L5_9.n24 ] && [ $(cnt gk+gbsm05L5_9) -ge 24 ]; then
    post "AUTO TABLE K14 n = 24 (gbs -0.5 at L5-9 only, on top of gk, paired vs gk, fresh A41; the 16 extension evals split over both workers): $($PY $S/k_line_v1.py $R gk+gbsm05L5_9 gk 7100584 7100607) || 7100592-607 alone: $($PY $S/k_line_v1.py $R gk+gbsm05L5_9 gk 7100592 7100607)"
    touch $M/gk_gbsm05L5_9.n24; fi
  if [ ! -f $M/gk_gbsm05L5_9.foldone ] && [ $(cnt gk+gbsm10L5_9) -ge 8 ]; then
    post "AUTO TABLE K14 -1.0 (gbs -1.0 at L5-9, on top of gk, fresh A41 7100584-591): vs gk: $($PY $S/k_line_v1.py $R gk+gbsm10L5_9 gk 7100584 7100591) || vs gbs -0.5 at L5-9: $($PY $S/k_line_v1.py $R gk+gbsm10L5_9 gk+gbsm05L5_9 7100584 7100591)"
    touch $M/gk_gbsm05L5_9.foldone; fi
  [ -f $M/gk_gbsm05L5_9.n24 ] && [ -f $M/gk_gbsm05L5_9.foldone ] && exit 0
  sleep 30
done
