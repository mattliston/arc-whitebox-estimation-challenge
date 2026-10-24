#!/bin/bash
# TAG scout-card (2026-10-13 04:2xZ, lead 04:1xZ: K13 follow-ups on worker 1 after its G-net half; every 8-network result is a SCREEN only, nothing goes
# to a stack before n = 24 and the out-of-selection 16 must agree in sign): per arm (run_k6 tag k60db, 7100584-591), when its 8 land: post vs
# gk+k4lf34h (the accepted K13, paired on the same networks: the increment) and vs gk. Noise-free mean log vs k4lf34h <= -0.10% -> 16 more
# (7100592-607, run_k6 tag k60dc, right behind), then the n = 24 table and the 16 alone with a sign check against the n = 8 screen.
# Arms: k4l1415h (K4L ages 3-4 at 0.5 on L14-15), k4lf3h (age 3 only at L15), k4lf34ha3e (k4lf34h + age 3 at 0.7 on L1-9). Markers: posted_k13f.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
S=$O/scout_card; PY="/usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /Users/matt/anaconda3/bin/python"
post() { echo "[scout-card] $(date -u +%FT%TZ) $1" >> $O/TEAM_LOG.md; }
M=$S/posted_k13f; mkdir -p $M
cnt() { ls $S/run_k6/results 2>/dev/null | grep -cF "_$1.json"; }
meanlog() { echo "$1" | sed -n 's/.*mean log \([-+0-9.]*\)%.*/\1/p' | head -1; }          # greedy: the noise-free number
R=$S/run_k1,$S/run_k6
while [ ! -f $S/NO_AUTOPOST ]; do
  n_done=0
  for a in gk+k4l1415h gk+k4lf3h gk+k4lf34ha3e; do
    k=$(echo $a | tr '+' '_')
    if [ ! -f $M/$k ] && [ $(cnt $a) -ge 8 ]; then
      inc=$($PY $S/k_line_v1.py $R $a gk+k4lf34h 7100584 7100591); ml=$(meanlog "$inc")
      post "AUTO TABLE K13 FOLLOW-UP SCREEN (n = 8, 7100584-591; a screen only): vs the accepted k4lf34h: $inc || vs gk: $($PY $S/k_line_v1.py $R $a gk 7100584 7100591)"
      if awk -v m="$ml" 'BEGIN {exit !(m <= -0.10)}'; then
        (cd $S && $PY make_k3_v8.py run_k6 k60dc 7100592 16 $a > /dev/null) && post "AUTO K13 FOLLOW-UP: $a noise-free ${ml}% vs k4lf34h <= -0.10%: 16 more on 7100592-607 (run_k6, tag k60dc)."; echo "$ml" > $M/$k.ext; fi
      touch $M/$k; fi
    if [ -f $M/$k.ext ] && [ ! -f $M/$k.n24 ] && [ $(cnt $a) -ge 24 ]; then
      out=$($PY $S/k_line_v1.py $R $a gk+k4lf34h 7100592 7100607); m16=$(meanlog "$out"); m8=$(cat $M/$k.ext)
      sign=$(awk -v a="$m8" -v b="$m16" 'BEGIN {print (a * b > 0) ? "AGREES in sign with the screen" : "does NOT agree in sign with the screen"}')
      post "AUTO TABLE K13 FOLLOW-UP n = 24 ($a vs the accepted k4lf34h, 7100584-607): $($PY $S/k_line_v1.py $R $a gk+k4lf34h 7100584 7100607) || out-of-selection 16 (7100592-607): $out -> $sign || vs gk n = 24: $($PY $S/k_line_v1.py $R $a gk 7100584 7100607)"; touch $M/$k.n24; fi
    if [ -f $M/$k ] && { [ ! -f $M/$k.ext ] || [ -f $M/$k.n24 ]; }; then n_done=$((n_done + 1)); fi
  done
  [ $n_done -ge 3 ] && exit 0
  sleep 30
done
