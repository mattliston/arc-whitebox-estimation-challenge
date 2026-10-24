#!/bin/bash
# TAG opus_engineer: the trainer's hand-over of a pair-law dial set -> the native candidate. Text only: nothing is run natively.
#   1. copies O/trainer/candidates/<name>/candidate.py and physics_dials.json into a20_runm/exports/<name>/ (an existing copy must be the same bytes)
#   2. a38_pairlaw/dials_from_trainer.py -> a38_pairlaw/dials/<name>_r<dfit>.json (fold strength per layer, the factor on the multiple of C, the three
#      source strengths, the young pairs' plane's strength by cohort age; three forms, beta)
#   3. stack/build.py on the card with the lever stack the fourth argument names and the pair-law fragment with those dials;
#      the F4C and K4L strengths are read from the hand-over's builder_levers
#   4. the einsum gate against the fired S40 file
# usage: law_file_from_trainer.sh <name of the directory under O/trainer/candidates> <dfit: 2 = table of multiples | 3 = constant regressor> [same | deep | s41 | plain] [<base> | -] [<head weights> | -] [<final weights>]
#   <head weights>  the trainer's new input column of the learned mean correction: the 128 weights in a42_head/weights/<name>.json
#                   (a42_head/weights_from_trainer.py <hand-over> <name>); lever HC@<name>, pair law v9 only
#   <final weights> the final head's columns (a42_head/weights/<name>.json with "columns"); lever HCF@<name>
#   <base>          the physics base of a refit: the levers listed in a41_base/bases/<base>.json (lever BT@<base>: second-order mean term, birth weight,
#                   transport gate, collective-mode cross form; section 28 of RESULT.md)
#   same (default)  S41's arithmetic bit for bit in fewer requests: the strips' leaf views per tree (HV2), the constants at setup (CS), F4c v3,
#                   K4L v7, edge cut v2, pair law v9
#   deep            the rounding-level diet instead: F4c v2 and K4L v6 (350 requests fewer again, +0.07% of bill), with HV2, CS, edge cut v2, pair law v9
#   s41             the lever stack of the fired S41 source: HV, F4c v1, K4L v5, edge cut v1, pair law v8, no CS
#   plain           s41 without HV (the file of the first hand-over's sessions)
# PLIST=<name> (env, 2026-10-11): the prebind list a13_prebind/prebind_<name>.py in place of hfull_v1 (lever P<name>)
set -eu
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
name=$1; dfit=$2; diet=${3:-same}; base=${4:-}; head=${5:-}; final=${6:-}
[ "$base" = "-" ] && base=
[ "$head" = "-" ] && head=
case "$diet" in
  same)  hv=+HV2; f4=F4Cv3; k4=K4Lv7; ec=EcutSv2; np=NPv9; cs=+CS;;
  deep)  hv=+HV2; f4=F4Cv2; k4=K4Lv6; ec=EcutSv2; np=NPv9; cs=+CS;;
  s41)   hv=+HV; f4=F4C;   k4=K4Lv5; ec=EcutS;   np=NPv8; cs=;;
  plain) hv=;    f4=F4C;   k4=K4Lv5; ec=EcutS;   np=NPv8; cs=;;
  *) echo "third argument: same | deep | s41 | plain"; exit 6;;
esac
short=${name#law_dials_}   # the dials file's name: short, a candidate file name must stay under 255 bytes with its .build.json
src=$O/trainer/candidates/$name
mkdir -p "$E/a20_runm/exports/$name"
for f in candidate.py physics_dials.json; do
  if [ -e "$E/a20_runm/exports/$name/$f" ]; then cmp "$src/$f" "$E/a20_runm/exports/$name/$f"; else cp "$src/$f" "$E/a20_runm/exports/$name/$f"; fi
done
shasum -a 256 "$E/a20_runm/exports/$name/candidate.py" | cut -c1-64
[ -e "$E/a38_pairlaw/dials/${short}_r${dfit}.json" ] || $PY -B "$E/a38_pairlaw/dials_from_trainer.py" "$E/a20_runm/exports/$name/physics_dials.json" "${short}_r${dfit}" "$dfit" | cut -c1-200
levers=$($PY -B -c "
import json, sys
b = json.load(open(sys.argv[1]))['builder_levers']
r, k, s1, s2 = b['rule_strength_per_age'], b['K4L_per_age'], b['exact_S1_per_age'], b['exact_S2_per_age']
k = [x[1] if isinstance(x, list) else x for x in k]   # a per-layer list (index = layer): the lever takes layer 1's value; the other layers go to DLY@ (s43_build.sh)
print('%st%r,%rc%r,%r,%r,%r+%sc%r,%r' % (sys.argv[2], r[0], r[1], float(s1[0]), float(s2[0]), float(s1[1]), float(s2[1]), sys.argv[3], k[0], k[1]))" "$E/a20_runm/exports/$name/physics_dials.json" "$f4" "$k4")
[ -n "${EC_OVERRIDE:-}" ] && ec=$EC_OVERRIDE   # audit: EC_OVERRIDE=none builds the learned edge exactly (no edge cut)
ecpart="+$ec"; [ "$ec" = none ] && ecpart=""
spec="T+C+B+F+D+Hfull$hv+$levers$ecpart+L3+L3o+SS0+$np@${short}_r${dfit}${head:+"+HC@$head"}${base:+"+BT@$base"}+P${PLIST:-hfull_v1}$cs${final:+"+HCF@$final"}+Rlvfuse_sized32+Rtsr_off"
echo "levers: $spec"
cd "$E/stack"
$PY -B build.py "$spec" "$E/a20_runm/exports/$name/candidate.py" "$name" | tail -1 | cut -c1-400
out="$E/stack/candidate_${spec//+/_}__$name.py"
[ $(basename "$out" | wc -c) -lt 244 ] || { echo "file name too long"; exit 5; }
S40="$E/stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py"
$PY -B "$P2/fable_porter_v1/einsum_site_diff.py" "$out" "$S40" | tail -1
shasum -a 256 "$out"
