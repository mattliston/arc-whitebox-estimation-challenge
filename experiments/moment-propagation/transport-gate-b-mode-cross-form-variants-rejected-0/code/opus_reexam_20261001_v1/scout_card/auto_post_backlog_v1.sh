#!/bin/bash
# TAG scout-card (2026-10-12, lead: backlog v1 behind K6 / K7, same protocol): per arm, when its 8 networks (7100584-591) land, post the table on
# top of gk (paired vs gk; raw and noise-free) and, if its mean log <= -0.15%, queue 16 more (7100592-607) in the same run dir right behind (its
# own extension tag) and post the n = 24 table when they land. Also the run_k12 control (gk on worker_sc9 = otr_m_sc5 with F4C age 3 off) vs sc1 gk;
# if it is not bit for bit, the f4c3 arm is posted against that control instead of gk. Markers in O/scout_card/posted_backlog.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
S=$O/scout_card; PY="/usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/matt/anaconda3/bin/python"
post() { echo "[scout-card] $(date -u +%FT%TZ) $1" >> $O/TEAM_LOG.md; }
M=$S/posted_backlog; mkdir -p $M
ENTRIES="run_k7|gk+kb10|make_k11_v1.py|k78x|K21 recentering by the head's d, b <- b - 0.10 (m - host)
run_k7|gk+kb25|make_k11_v1.py|k78x|K21 recentering, kb 0.25
run_k7|gk+bfx15|make_k11_v1.py|k79x|gain b-fix gamma 0.1 -> 0.15
run_k7|gk+bfx20|make_k11_v1.py|k79x|gain b-fix gamma 0.1 -> 0.20
run_k12|gk+f4c3|make_k12_v1.py|k121x|F4C age 3 at the age-2 strengths (f4a3 = f4a2, x2a3 = x2a2)"
cnt() { ls $S/$1/results 2>/dev/null | grep -cF "_$2.json"; }
while [ ! -f $S/NO_AUTOPOST ]; do
  if [ ! -f $M/ctl12 ] && [ -f $S/run_k12/results/sc_k120_r7100584_gk.json ]; then
    v=$($PY $S/k_ctl_pair_v1.py $S/run_k12/results/sc_k120_r7100584_gk.json $S/run_k1/results/sc_k1_r7100584_gk.json)
    post "AUTO CONTROL K12 (gk on worker_sc9 = otr_m_sc5, F4C age 3 OFF (f4a3 -0.6, x1a3 = x2a3 = 0), vs gk on worker_sc1, 7100584): $v"
    case "$v" in BIT*) echo gk > $M/ref12 ;; *) echo ctl12 > $M/ref12 ;; esac
    touch $M/ctl12; fi
  ndone=0
  while IFS='|' read -r run arm gen ext label; do
    k=$(echo "${run}_${arm}" | tr '+' '_')
    ref=gk; refruns="$S/run_k1,$S/$run"
    if [ "$run" = run_k12 ] && [ -f $M/ref12 ] && [ "$(cat $M/ref12)" = ctl12 ]; then ref=gk; refruns="$S/$run"; fi
    if [ ! -f $M/$k.n8 ] && [ $(cnt $run $arm) -ge 8 ]; then
      line=$($PY $S/k_line_v1.py $refruns $arm $ref 7100584 7100591)
      post "AUTO TABLE BACKLOG ($label; on top of gk, paired vs gk, fresh A41 7100584-591): $line"
      ml=$(echo "$line" | sed -n 's/.*mean log \([-+0-9.]*\)%.*/\1/p' | head -1)
      if awk -v m="$ml" 'BEGIN {exit !(m <= -0.15)}'; then
        (cd $S && $PY $gen $run $ext 7100592 16 $arm > /dev/null) && post "AUTO BACKLOG: $arm n = 8 mean log ${ml}% <= -0.15%: queued on 7100592-607 in $run (tag $ext)."
        touch $M/$k.ext; fi
      touch $M/$k.n8; fi
    if [ -f $M/$k.ext ] && [ ! -f $M/$k.n24 ] && [ $(cnt $run $arm) -ge 24 ]; then
      post "AUTO TABLE BACKLOG n = 24 ($label; on top of gk, paired vs gk, 7100584-607): $($PY $S/k_line_v1.py $refruns $arm $ref 7100584 7100607)"; touch $M/$k.n24; fi
    if [ -f $M/$k.n8 ] && { [ ! -f $M/$k.ext ] || [ -f $M/$k.n24 ]; }; then ndone=$((ndone + 1)); fi
  done <<< "$ENTRIES"
  [ $ndone -ge 5 ] && exit 0
  sleep 30
done
