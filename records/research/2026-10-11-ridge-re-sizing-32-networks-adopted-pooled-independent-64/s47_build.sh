#!/bin/bash
# TAG opus_engineer (2026-10-11, lead 05:52Z): S47 in one command from the trainer's hand-over: the SAFE file S47s = Nd physics + request
# diet v2 + 7-level strip pushes on full packets (base s46h7r) and the fallback S47n = Nd physics + request diet v2 (base s46xq2).
# Both through s43_build.sh (KC@ / DLY@ / HCF@ / BAND@ from the hand-over as for S46). Per-layer dials the trainer's exporter cannot carry
# (iso / bscale / X1 varying by layer): give the REAL checkpoint as the third argument; its values go to a52_pld/values/<name>_pt.json
# (a52_pld/dly_from_pt.py, read without torch) and both files take DLY@<name>_pt.
# usage: s47_build.sh <hand-over name under O/trainer/candidates> <head table name> [<real checkpoint .pt>]   (text only; prints shas)
set -eu
E=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer
O=$(dirname $E); PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
name=$1; head=$2; pt=${3:-}
dly=""
if [ -n "$pt" ]; then
  if [ ! -e "$E/a52_pld/values/${name}_pt.json" ]; then
    $PY -B $E/a52_pld/dly_from_pt.py "$pt" "${name}_pt" "$O/trainer/candidates/$name/physics_dials.json"
  fi
  dly="${name}_pt"
fi
for base in s46h7r s46xq2; do
  echo "== $base"
  DLY_NAME=$dly bash $E/stack/s43_build.sh "$name" "$head" "$base" 2>&1 | grep -E "^base|^kappa|^per-layer|VERDICT|^[0-9a-f]{64}|Error|rror" | cut -c1-200
done
