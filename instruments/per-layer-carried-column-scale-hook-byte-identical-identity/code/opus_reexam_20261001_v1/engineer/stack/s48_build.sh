#!/bin/bash
# TAG opus_engineer (2026-10-11 07:4xZ; 08:1xZ: default base s46h7rsd3): S48 in one command from a trainer hand-over: ONE safe file = the
# hand-over's physics on base s46h7rsd3 (S47s's base + the persistent sandwich scratch XRQS, byte-identical, about -500 requests per
# network, + the direct strip loads HZD3, byte-identical, about -0.1% of the bill); BASE=s46h7rs = without HZD3; BASE=s46h7rsd3x adds the
# inner-size covariance sandwich XSDM (rounding-level; only after its checks). Through s43_build.sh
# (KC@ / DLY@ / HCF@ / BAND@ from the hand-over as for S46 / S47). Per-layer dials the trainer's exporter cannot carry (iso / bscale / X1
# varying by layer, and per-layer s_carry since 09:1xZ): give the REAL checkpoint as the third argument (a52_pld/dly_from_pt_v2.py, read
# without torch; DLY@<name>_pt).
# PLIST=<name> (env): the prebind list a13_prebind/prebind_<name>.py in place of hfull_v1. BASE=<base> (env) overrides s46h7rsd3.
# usage: [PLIST=<list>] [BASE=<base>] s48_build.sh <hand-over name under O/trainer/candidates> <head table name> [<real checkpoint .pt>]
set -eu
E=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer
O=$(dirname $E); PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
name=$1; head=$2; pt=${3:-}
dly=""
if [ -n "$pt" ]; then
  if [ ! -e "$E/a52_pld/values/${name}_pt.json" ]; then
    $PY -B $E/a52_pld/dly_from_pt_v2.py "$pt" "${name}_pt" "$O/trainer/candidates/$name/physics_dials.json"   # v2: + per-layer s_carry ('scarry', pld v3)
  fi
  dly="${name}_pt"
fi
base=${BASE:-s46h7rsd3}
echo "== $base${PLIST:+ (prebind list $PLIST)}"
DLY_NAME=$dly PLIST=${PLIST:-hfull_v1} bash $E/stack/s43_build.sh "$name" "$head" "$base" 2>&1 | grep -E "^base|^kappa|^per-layer|VERDICT|^[0-9a-f]{64}|Error|rror" | cut -c1-200
