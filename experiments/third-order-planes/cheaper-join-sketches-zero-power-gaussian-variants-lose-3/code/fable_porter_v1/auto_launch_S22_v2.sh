#!/usr/bin/env bash
# TAG fable-porter (PT-39c v2: + machine_gate.sh before the launch, lead 21:5x machine rule; v1 = auto_launch_S22.sh, stopped in its dials wait)
# (lead 21:0x "don't launch S22 until the physicist's refit passes its VALID16 bar"): waits for
#   (1) fable_physicist_v1/refit6b/s6b_loopR1.log to carry {"eval": "valid_epoch0", "mean": X} with X <= 1.657977e-8 (the physicist's
#       pre-registered bar: 0.995 x the run's own valid_before 1.666309e-8) AND s6b_loopR1/params_epoch0.pt to exist; X > bar -> log, stop;
#   (2) covhunter's round-2 dials at $DIALS (flat FIT keys);
# then runs fire_chain_S22.sh <params> <dials> r2 (log fire_chain_S22.out). Refuses if fire_chain_S22.out exists.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
LOG=$P2/fable_physicist_v1/refit6b/s6b_loopR1.log; PT=$P2/fable_physicist_v1/refit6b/s6b_loopR1/params_epoch0.pt; BAR=1.657977e-8
DIALS=${1:-$P2/fable_covhunter_v1/dials_r2_loopR1.json}
until grep -q '"eval": "valid_epoch0"' $LOG 2>/dev/null && [ -s $PT ]; do sleep 30; done
sleep 20
V=$(grep '"eval": "valid_epoch0"' $LOG | tail -1 | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['mean'])")
if ! python3 -c "import sys; sys.exit(0 if float('$V') <= $BAR else 1)"; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S22 NOT launched: physicist loopR1 VALID16 $V > bar $BAR" >> $TL; echo "loopR1 VALID16 $V FAILS bar $BAR"; exit 0; fi
echo "$(date +%T) loopR1 VALID16 $V <= $BAR PASS (params $(shasum -a 256 $PT | cut -c1-16)); waiting for round-2 dials $DIALS"
until [ -s $DIALS ]; do sleep 30; done
sleep 20
python3 -c "import json; d=json.load(open('$DIALS')); assert sorted(d)==sorted(['X1','X0','X2','X3','bscale','s_carry','s_new','c0','c1','k4a','k4b','k4c']), sorted(d)" || { echo "round-2 dials malformed"; exit 1; }
[ -e $H/fire_chain_S22.out ] && { echo "fire_chain_S22.out exists; not relaunching"; exit 1; }
echo "$(date '+%Y-%m-%d %H:%M') porter: S22 launching (loopR1 VALID16 $V PASS; round-2 dials $(basename $DIALS) sha $(shasum -a 256 $DIALS | cut -c1-16))" >> $TL
bash $H/machine_gate.sh 7 S22-launch
cd $H; bash fire_chain_S22.sh $PT $DIALS r2 > fire_chain_S22.out 2>&1
echo "$(date +%T) S22 chain finished: $(tail -1 fire_chain_S22.out)"
