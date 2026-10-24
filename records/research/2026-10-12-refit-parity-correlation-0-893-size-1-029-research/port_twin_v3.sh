#!/bin/bash
# TAG opus_engineer a62 port_twin_v3 (file written 2026-10-12T17:20Z) = port_twin_v2.sh + the safe port's x2a parent edit (s49/<name>_x2a,
# make_x2a_v1.py) re-applied to the twin's parent with the same values fragment, like kp / k22. v2 header:
# TAG opus_engineer a62 port_twin_v2 (file written 2026-10-12T16:53Z) = port_twin_v1.sh + stack/s48_build_v4.sh (dly_from_pt_v3) + the safe file's
# PARENT text edits re-applied to the twin's release form: when the safe's S49 parent is s49/<name>_k22 (k22 by layer, port_rad_v1m) or
# s49/<name>_kp (kappa carried), the same make_k22_v1.py / make_kappa_v1.py with the same values fragment edits the twin's parent first.
# v1 header: (file written 2026-10-12 04:33Z; lead: build S50's twin right behind the safe file): the TWIN of a ported RAD point, text only,
# the way release built the S48 twin (variants/s48t13_twin, 23:06Z 10-11): the same hand-over and checkpoint on the "same" diet
# (F4Cv3 / K4Lv7 in place of the safe's F4Cv2 / K4Lv6; same base token, so the strips are the safe's), the release fragments
# l3_bind128 + prices_zero only (no lvfs_limit_off), then the safe S49 file's own tail re-applied item for item from its build.json
# (star, star strength, read column cut, prebind list, k31l by layer, k4, ...), so the twin differs from the safe file only in the diet
# and that one release fragment. Its own prebind list, XSG and certificates come after (census -> prebind_targeted_v2 -> make_xsg_any_v1).
# usage: port_twin_v1.sh <hand-over name> <checkpoint .pt (the safe port's)> <safe S49 name> <twin name>
set -eu
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules
E=$(dirname $A); O=$(dirname $E); T=$O/trainer/candidates; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
name=$1; pt=$2; safe=$3; twin=$4
[ -d "$T/$name" ] && [ -f "$pt" ] && [ -f "$A/s49/$safe/build.json" ] || { echo "usage: port_twin_v1.sh <hand-over> <pt> <safe S49 name> <twin name>"; exit 2; }
[ ! -e "$A/s49/$twin" ] && [ ! -e "$E/stack/rl/${twin}_rl" ] || { echo "outputs for $twin exist; pick a new name"; exit 2; }
echo "== $(date -u +%H:%M:%SZ) twin $twin of $safe: hand-over $T/$name, checkpoint $pt ($(shasum -a 256 "$pt" | cut -c1-8))"
if cmp -s "$T/$name/head_weights_31x128.json" "$T/jmNd/head_weights_31x128.json"; then head=jmNdh; else head=${name}h; fi
[ -e "$E/a42_head/weights/$head.json" ] || { echo "head table $head missing (the safe port converts it)"; exit 3; }
cd $E
log=$A/port_${twin}_build.log
DIET=same BASE=s46h7rsd3x bash $E/stack/s48_build_v4.sh "$name" "$head" "$pt" | tee $log
k=$(grep -o '^base k[0-9a-f]*' $log | tail -1 | cut -d' ' -f2)
src=$(ls $E/stack/candidate_*_F4Cv3*_K4Lv7*_BT@${k}_*__"$name".py)
[ $(echo "$src" | wc -l) -eq 1 ] || { echo "expected one same-diet source for BT@$k and $name"; exit 4; }
echo "source $(shasum -a 256 "$src" | cut -c1-8) $src"
$PY -B $A/law_literals_from_pt_v2.py "$pt" --check "$src" | tee $A/port_${twin}_law.txt | tail -2
grep -q "^CHECK PASS" $A/port_${twin}_law.txt || { echo "law literals differ from the checkpoint: stop"; exit 6; }
$PY -B $E/stack/release_like.py "$src" "${twin}_rl" l3_bind128 prices_zero > /dev/null
echo "release form $(shasum -a 256 $E/stack/rl/${twin}_rl/candidate.py | cut -c1-8) stack/rl/${twin}_rl/candidate.py"
items=$($PY -B -c "
import json, sys
for i in json.load(open(sys.argv[1]))['items']:
    print(i['assign'] if 'assign' in i else i['path'])" $A/s49/$safe/build.json)
tpar=$E/stack/rl/${twin}_rl/candidate.py
for kind in kp k22 x2a; do   # v3: x2a (make_x2a_v1.py, F4c strengths by birth; the twin is F4c v3)   # the safe port made s49/<name>_kp (kappa carried) and / or s49/<name>_k22 (k22 by layer) only when it used them
  [ -f $A/s49/${name}_${kind}/build.json ] || continue
  vals=$($PY -B -c "import json; print(json.load(open('$A/s49/${name}_${kind}/build.json'))['values'])")
  tool=make_kappa_v1.py; [ $kind = k22 ] && tool=make_k22_v1.py; [ $kind = x2a ] && tool=make_x2a_v1.py
  $PY -B $A/$tool "$tpar" "${twin}_${kind}" "$vals" > /dev/null
  tpar=$A/s49/${twin}_${kind}/candidate.py; echo "twin parent edit re-applied: $kind ($tool, $vals)"
done
$PY -B $A/make_s49_v1.py "$tpar" "$twin" $items > /dev/null
f=$A/s49/$twin/candidate.py
echo "TWIN $(shasum -a 256 $f | cut -c1-8) $f; release fragment lines $(grep -c 'BEGIN OPUS-RELEASE FRAGMENT' $f) (4 = the twin's set)"
echo "diff against the safe $safe ($(shasum -a 256 $A/s49/$safe/candidate.py | cut -c1-8)), lines: $(diff $A/s49/$safe/candidate.py $f | grep -c '^[<>]' || true)"
