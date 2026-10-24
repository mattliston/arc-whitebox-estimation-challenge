#!/bin/bash
# TAG opus-physicist T112 chain v17 (lead 15:2xZ disk) = v16 + each lane's weight cache (generated_cache_v1) made a SYMLINK, before its seed runs, to ONE directory per run: physicist/sealed_cache/<run dir name> (the worker writes the same sha-checked weights there; numerics identical; one directory per round to clear with Matt's OK). v16 = v15 on the release guard v2 (release 08:2xZ: a slow ps no longer retires the group; same limits). v15: = t112_chain_v14.sh with the JUDGE given as an argument and the machine gate
# (O/gate.sh 4: fewer than 6 heavy python processes, free+inactive >= 8 GB, memory_pressure >= 30%) before EACH seed; each seed's baseline + candidate pair
# runs in one judge process (3 threads) under the release memory guard (footprint 7.5 GB, RSS 10 GB, 30 min). Logs the guard wrapper's REAL exit code.
# usage: bash t112_chain_v15.sh <judge .py> <manifest.json> <name> <seed> [<seed> ...]
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
JUDGE=$1; MAN=$2; NAME=$3; shift 3
mkdir -p $O/physicist/t112_guards
for s in "$@"; do
  [ -f $O/physicist/STOP_T112 ] && { echo "$(date -u +%H:%M:%SZ) STOP_T112 present" >> $O/physicist/t112_chain_$NAME.log; break; }
  bash $O/gate.sh 4 physicist_$NAME > /dev/null
  LANE=$(dirname $MAN)/seed$s; SC=$O/physicist/sealed_cache/$(basename $(dirname $MAN)); mkdir -p $SC
  [ -e $LANE/generated_cache_v1 ] || ln -s $SC $LANE/generated_cache_v1
  echo "$(date -u +%H:%M:%SZ) seed $s start" >> $O/physicist/t112_chain_$NAME.log
  nice -n 10 /Users/matt/anaconda3/bin/python -B $O/release/guard_footprint_v2.py --record $O/physicist/t112_guards/seed$s.json --seconds 1800 \
    --footprint-limit 7500000000 --rss-backstop 10000000000 /Users/matt/anaconda3/bin/python -B $JUDGE run-one $MAN $s \
    >> $O/physicist/t112_chain_$NAME.log 2>&1
  rc=$?
  echo "$(date -u +%H:%M:%SZ) seed $s exit $rc" >> $O/physicist/t112_chain_$NAME.log
done
echo "$(date -u +%H:%M:%SZ) CHAINDONE" >> $O/physicist/t112_chain_$NAME.log
