#!/bin/bash
# TAG scout-card (2026-10-13 06:2xZ, lead: L15 extensions at the S51 + gk point): posts, as worker 2's run_k7 lands them,
# 1) K13 (k4lf34h) AT THE S51 POINT, n = 8 (7100584-591): 584 / 585 from k72hd, 586-591 from k80, paired vs the S51 + gk controls (584 / 585 my own,
#    586-591 = the physicist's T134 learned-ON controls, copied with their ids renamed to O/scout_card/run_s51ref; my worker reproduced T134 bit for
#    bit on 584 / 585), next to the S50-point numbers on the same 8;
# 2) the S51 strength scan (k81: 0.4 / 0.6 at L15) vs k4lf34h at S51 (the increment) and vs the S51 controls; a <= -0.10% (noise-free) result is
#    POSTED for the lead (its 16-network extension needs S51 references on 592-607 too: 48 evals), not queued.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
S=$O/scout_card; PY="/usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/matt/anaconda3/bin/python"
post() { echo "[scout-card] $(date -u +%FT%TZ) $1" >> $O/TEAM_LOG.md; }
cnt() { ls $S/run_k7/results 2>/dev/null | grep -cE "^sc_[^_]+_r710058[4-9]_$1\.json$|^sc_[^_]+_r710059[01]_$1\.json$"; }
R=$S/run_s51ref,$S/run_k7
until [ $(cnt s51k4lf34h) -ge 8 ]; do [ -f $S/NO_AUTOPOST ] && exit 0; sleep 30; done
post "AUTO TABLE K13 AT THE S51 + gk POINT (k4lf34h; the controls are the physicist's T134 learned-ON S51 + gk11 evals, which my worker reproduced bit for bit on 584 / 585): $($PY $S/k_line_v1.py $R s51k4lf34h s51ctl 7100584 7100591) || the same 8 at the S50 point: $($PY $S/k_line_v1.py $S/run_k1,$S/run_k6 gk+k4lf34h gk 7100584 7100591)"
until [ $(cnt s51k4lf34h4) -ge 8 ] && [ $(cnt s51k4lf34h6) -ge 8 ]; do [ -f $S/NO_AUTOPOST ] && exit 0; sleep 30; done
for a in s51k4lf34h4 s51k4lf34h6; do
  post "AUTO TABLE K13 STRENGTH AT S51 ($a = strength 0.${a: -1} at L15 vs 0.5; a screen, n = 8): vs k4lf34h at S51: $($PY $S/k_line_v1.py $R $a s51k4lf34h 7100584 7100591) || vs the S51 controls: $($PY $S/k_line_v1.py $R $a s51ctl 7100584 7100591)"
done
