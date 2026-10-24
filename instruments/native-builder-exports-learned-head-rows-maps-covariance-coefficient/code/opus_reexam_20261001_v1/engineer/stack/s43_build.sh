#!/bin/bash
# TAG opus_engineer: S43 / S44 in one command (lead 10-10 04:05Z, 05:xxZ; trainer 10-10 jmC heads-up). The trainer's hand-over
# O/trainer/candidates/<name>/ (candidate.py with the learned arrays + physics_dials.json: pair-law dials, F4C / K4L strengths, the young
# pairs' plane by cohort age, kappa_C per layer when it has one) on the deeper request diet, with
#   <head>   the head columns: a42_head/weights/<head>.json, or, when that does not exist, the trainer's table
#            O/trainer/candidates/<name>/<head>.json (31 x 128, rows = his HEAD_COLUMNS), converted by a42_head/weights_from_trainer.py
#            (layers 1-14 -> <head>.json; rows 17-19, layer 15 -> <head>_f15.json); default s2 = S42's step-2 column
#   <base>   a41_base/bases/<base>.json; default s43m = BSK at 0.34 + DCV64f0.001p (the padded dead cut at gate floor 1e-3);
#            s43c17 = s43m + column 17 at layer 15 with the physicist's u15
# The build adds, through a generated base k<5 hex> (short: the file name must stay under 244 bytes): KC@<name> when physics_dials.json
# carries kappa_C (a47_kc/kc_from_trainer.py), HCF@<head>_f15 in place of the base's own final-layer columns when <head>_f15.json exists,
# DLY@<name> when physics_dials.json carries per-layer dial values (a52_pld/dly_from_trainer.py: bscale / X1 / iso lists, the age-1 K4L
# strength by layer), and BAND@<head> when the table has the per-band rows 31-35 (a49_band/weights/<head>.json: enc0_b by band 1-9 / 10-14 / 15, columns 6 / 20 at 10-14).
# <final weights>: the final layer's head columns as an extra lever (a42_head/weights/<name>.json with "columns"); none by default.
# DLY_NAME=<values> (env): DLY@<values> in place of DLY@<hand-over name> (a52_pld/values/<values>.json written by hand, e.g. jmIv).
# usage: s43_build.sh <hand-over name> [<head weights> [<base> [<final weights>]]]   (text only: builds, prints the file and its sha)
set -eu
E=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer
O=$(dirname $E); PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
name=$1; head=${2:-s2}; base=${3:-s43m}; final=${4:-}
if [ ! -e "$E/a42_head/weights/$head.json" ]; then
  [ -e "$O/trainer/candidates/$name/$head.json" ] || { echo "no head weights $head (a42_head/weights or the hand-over)"; exit 7; }
  $PY -B $E/a42_head/weights_from_trainer.py --json "$O/trainer/candidates/$name/$head.json" all "$head" --candidate "$O/trainer/candidates/$name/candidate.py"
fi
kc=$($PY -B $E/a47_kc/kc_from_trainer.py "$O/trainer/candidates/$name/physics_dials.json" "$name")
echo "kappa_C: $kc"
dly=$($PY -B $E/a52_pld/dly_from_trainer.py "$O/trainer/candidates/$name/physics_dials.json" "$name")
echo "per-layer dials: $dly"
extra=""
[ "$kc" != "none" ] && extra="KC@$name"
[ "$dly" != "none" ] && extra="$extra DLY@$name"
[ -n "${DLY_NAME:-}" ] && extra="${extra//DLY@$name/} DLY@$DLY_NAME" && echo "per-layer values from a52_pld/values/$DLY_NAME.json (DLY_NAME; lead 21:02Z: jmIv = jmIf + iso at 10-14)"
[ -e "$E/a42_head/weights/${head}_f15.json" ] && extra="$extra HCF@${head}_f15"
[ -e "$E/a49_band/weights/${head}.json" ] && extra="$extra BAND@${head}"
if [ -n "$extra" ]; then
  gen=k$(printf "%s" "$base $name $head $extra" | md5 | cut -c1-5)
  $PY -B -c "
import json, sys
b = json.load(open(sys.argv[1])); extra = sys.argv[3].split()
drop = tuple(x.split('@')[0] + '@' for x in extra)
b['levers'] = [x for x in b['levers'] if not x.startswith(drop)] + extra
b['request'] = b.get('request', '') + ' + ' + ' '.join(extra)
json.dump(b, open(sys.argv[2], 'w'), indent=1)" "$E/a41_base/bases/$base.json" "$E/a41_base/bases/$gen.json" "$extra"
  echo "base $gen = $base + $extra"
  base=$gen
fi
bash $E/stack/law_file_from_trainer.sh "$name" 2 deep "$base" "$head" $final
