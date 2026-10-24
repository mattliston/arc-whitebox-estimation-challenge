#!/bin/bash
# TAG scout-card (2026-10-13 08:4xZ, lead 08:3xZ idle-worker screens; pre-declared in TEAM_LOG before any result): run_k20 (worker_sc10, the TRUE S51 +
# gk point = the physicist's T134 requests, every arm ON TOP of k4lf34h = the fired card). Per arm (tag k20, 7100584-591), when its 8 land: post vs
# k4lf34h at S51 (the increment) and vs the T134 S51 + gk controls. Rule (pre-declared): noise-free mean log vs k4lf34h <= -0.15% -> the 24-network
# check runs automatically: the arm on 7100592-607 plus, once for all arms, the S51 references there (controls = T134's request with id / rank
# replaced, and k4lf34h), tag k20x; then the n = 24 table and the out-of-selection 16 with a sign check.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
S=$O/scout_card; PY="/usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/matt/anaconda3/bin/python"
post() { echo "[scout-card] $(date -u +%FT%TZ) $1" >> $O/TEAM_LOG.md; }
meanlog() { echo "$1" | sed -n 's/.*mean log \([-+0-9.]*\)%.*/\1/p' | head -1; }
M=$S/posted_k20; mkdir -p $M
cnt() { ls $S/run_k20/results 2>/dev/null | grep -cE "^sc_k20x?_r[0-9]+_$1\.json$"; }
R=$S/run_s51ref,$S/run_k7,$S/run_k20
while [ ! -f $S/NO_AUTOPOST ]; do
  n_done=0
  for a in s51e2m05 s51e34m05 s51a4b1214; do
    if [ ! -f $M/$a ] && [ $(cnt $a) -ge 8 ]; then
      inc=$($PY $S/k_line_v1.py $R $a s51k4lf34h 7100584 7100591); ml=$(meanlog "$inc")
      post "AUTO TABLE K20 SCREEN ($a on top of k4lf34h, the S51 + gk point, n = 8; pre-declared 08:4xZ): vs k4lf34h at S51: $inc || vs the T134 S51 + gk controls: $($PY $S/k_line_v1.py $R $a s51ctl 7100584 7100591)"
      if awk -v m="$ml" 'BEGIN {exit !(m <= -0.15)}'; then
        mkdir -p $S/run_k20/stage
        if [ ! -f $M/refs ]; then (cd $S && $PY make_k18_v3.py run_k20/stage k20x 7100592 16 s51ctl s51k4lf34h > /dev/null) && touch $M/refs; fi
        (cd $S && $PY make_k18_v3.py run_k20/stage k20x 7100592 16 $a > /dev/null)
        for f in $S/run_k20/stage/queue/*.json; do mv "$f" $S/run_k20/queue/; done
        echo "$ml" > $M/$a.ext
        post "AUTO K20: $a noise-free ${ml}% vs k4lf34h <= -0.15%: its 24-network check is queued (7100592-607 at S51, with the S51 references there; tag k20x)."
      fi
      touch $M/$a; fi
    if [ -f $M/$a.ext ] && [ ! -f $M/$a.n24 ] && [ $(cnt $a) -ge 24 ] && [ $(cnt s51k4lf34h) -ge 16 ] && [ $(cnt s51ctl) -ge 16 ]; then
      out=$($PY $S/k_line_v1.py $R $a s51k4lf34h 7100592 7100607); m16=$(meanlog "$out"); m8=$(cat $M/$a.ext)
      sign=$(awk -v a="$m8" -v b="$m16" 'BEGIN {print (a * b > 0) ? "AGREES in sign with the screen" : "does NOT agree in sign with the screen"}')
      post "AUTO TABLE K20 n = 24 ($a vs k4lf34h at S51, 7100584-607): $($PY $S/k_line_v1.py $R $a s51k4lf34h 7100584 7100607) || out-of-selection 16: $out -> $sign || k4lf34h itself vs the S51 controls on 592-607: $($PY $S/k_line_v1.py $R s51k4lf34h s51ctl 7100592 7100607)"
      touch $M/$a.n24; fi
    if [ -f $M/$a ] && { [ ! -f $M/$a.ext ] || [ -f $M/$a.n24 ]; }; then n_done=$((n_done + 1)); fi
  done
  [ $n_done -ge 3 ] && exit 0
  sleep 30
done
