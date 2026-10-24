#!/bin/bash
# TAG opus_engineer a62 v1b (2026-10-11 20:4xZ) = port_rad_v1.sh with the source found by the build's own base token (BT@k<hex>) and
# the hand-over name; v1 took the newest file by mtime and, when the build refused to overwrite an existing source, picked another
# base's file (rehearsal on jmMS_v2: the HZ7Rg0.0 twin). An existing source is refused unless REUSE=1 (rehearsals only).
# v1 header: RAD-S48's port in one command, text only (nothing native runs).
# From the trainer's jmMS_v2-style hand-over O/trainer/candidates/<name>/ (candidate.py, physics_dials.json, per_layer_dials.json,
# head_weights_31x128.json) and its real checkpoint <pt>:
#   1. head: the hand-over's head table = jmNd's (cmp) -> jmNdh; otherwise converted to a42_head/weights/<name>h.json (+ _f15, + band)
#   2. source: BASE=s46h7rsd3x stack/s48_build.sh <name> <head> <pt>   (law literals, KC, per-layer dials from the checkpoint, head)
#   3. law literals: law_literals_from_pt_v1.py <pt> --check <source> (the source's six pair-law literals = the checkpoint's)
#   4. release form: stack/release_like.py <source> <name>_rl l3_bind128 lvfs_limit_off prices_zero
#   5. star strength: star_beta_from_pt_v1.py <pt> <name> -> fragment_star_beta_<name>.py (beta = 1 + ssb per layer)
#   6. S49: make_s49_v1.py rl/<name>_rl/candidate.py <name> + star_dir_v1 + star_beta_<name> + readcol_v1 (a >= -2.0) + prebind_s48b_v2
#      -> a62_modules/s49/<name>/candidate.py (= S48b's tail on the RAD point)
# Checks printed: the fragment count (5 = release-ready, FRAGS=none), the diff against S48b outside the learned values.
# usage: port_rad_v1.sh <hand-over name under O/trainer/candidates> <checkpoint .pt>
set -eu
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules
E=$(dirname $A); O=$(dirname $E); T=$O/trainer/candidates; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
name=$1; pt=$2
[ -d "$T/$name" ] && [ -f "$pt" ] || { echo "usage: port_rad_v1.sh <hand-over name> <checkpoint .pt>"; exit 2; }
[ ! -e "$A/s49/$name" ] && [ ! -e "$E/stack/rl/${name}_rl" ] || { echo "outputs for $name exist; pick a new name"; exit 2; }
echo "== $(date -u +%H:%M:%SZ) hand-over $T/$name, checkpoint $pt ($(shasum -a 256 "$pt" | cut -c1-8))"
if cmp -s "$T/$name/head_weights_31x128.json" "$T/jmNd/head_weights_31x128.json"; then
  head=jmNdh; echo "head table = jmNd's (byte for byte): jmNdh"
else
  head=${name}h
  [ -e "$E/a42_head/weights/$head.json" ] && { echo "a42_head/weights/$head.json exists; refusing to reuse it"; exit 3; }
  $PY -B $E/a42_head/weights_from_trainer.py --json "$T/$name/head_weights_31x128.json" all "$head" --candidate "$T/$name/candidate.py"
  echo "head table converted: $head"
fi
cd $E
log=$A/port_${name}_build.log
BASE=s46h7rsd3x bash $E/stack/s48_build.sh "$name" "$head" "$pt" | tee $log
k=$(grep -o '^base k[0-9a-f]*' $log | tail -1 | cut -d' ' -f2)
[ -n "$k" ] || { echo "no base token in the build output"; exit 4; }
n=$(ls $E/stack/candidate_*_BT@${k}_*__"$name".py | wc -l)
[ "$n" -eq 1 ] || { echo "expected one source for BT@$k and $name, found $n"; exit 4; }
src=$(ls $E/stack/candidate_*_BT@${k}_*__"$name".py)
if grep -q 'AssertionError' $log; then
  [ "${REUSE:-0}" = 1 ] || { echo "the build refused (an existing source?): see $log; REUSE=1 only for rehearsals"; exit 4; }
  echo "REUSE=1: the existing source for BT@$k is taken"
fi
echo "source $(shasum -a 256 "$src" | cut -c1-8) $src"
$PY -B $A/law_literals_from_pt_v1.py "$pt" --check "$src"
$PY -B $E/stack/release_like.py "$src" "${name}_rl" l3_bind128 lvfs_limit_off prices_zero > /dev/null
echo "release form $(shasum -a 256 $E/stack/rl/${name}_rl/candidate.py | cut -c1-8) stack/rl/${name}_rl/candidate.py"
$PY -B $A/star_beta_from_pt_v1.py "$pt" "$name"
$PY -B $A/make_s49_v1.py $E/stack/rl/${name}_rl/candidate.py "$name" $A/fragment_star_dir_v1.py $A/fragment_star_beta_${name}.py \
  $A/fragment_readcol_v1.py $A/fragment_prebind_s48b_v2.py > /dev/null
f=$A/s49/$name/candidate.py
echo "S49 $(shasum -a 256 $f | cut -c1-8) $f; release fragment lines $(grep -c 'BEGIN OPUS-RELEASE FRAGMENT' $f) (5 = release-ready)"
echo "diff against S48b (8378970c), lines: $(diff $A/s48/s48b/candidate.py $f | grep -c '^[<>]' || true)"
