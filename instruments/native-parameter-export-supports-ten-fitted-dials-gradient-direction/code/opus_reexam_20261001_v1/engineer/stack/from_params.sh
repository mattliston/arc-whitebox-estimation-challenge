#!/bin/bash
# TAG opus_engineer: a run M parameter file -> the native candidate with the S37 lever stack. Text only: nothing is run natively.
#   1. the exported candidate: the trainer's own (O/trainer/candidates/<run>_ema_u<k>/candidate.py, 4th argument) or, without one,
#      stack/export_for_build.py on the parameter file (the same export: rc, edge, bsrc, the _GN_KNOBS values, ds in the head literals)
#   2. the lever string of the otr_m.py dial families that are levers here: F4 and K4L per age, k22 (stack/levers_from_params.py)
#   3. stack/build.py T+C+B+F+D+Hfull+<levers>[+<extra>]+Phfull_v1+Rlvfuse_sized32+Rtsr_off on the exported file (env PRE / POST change the two ends)
#   4. the einsum gate against the S37 file
# After it, by hand and only when the lead asks: a11_bare_views/setup_only.py <file> (light) and one native session (run_quick.sh).
# usage: from_params.sh <params.pt> <tag (new)> [<extra levers joined by +, e.g. L3+L3o, or '' for none>] [<exported candidate.py of the trainer>]
#        -> a20_runm/exports/<tag>/candidate.py and stack/candidate_<levers>__<tag>.py
set -eu
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
E=$P2/opus_reexam_20261001_v1/engineer
PY=/Users/matt/anaconda3/bin/python
params=$1; tag=$2; extra=${3:-}; given=${4:-}
S37=$E/stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py
mkdir -p "$E/a20_runm/exports/$tag"
export OMP_NUM_THREADS=1
if [ -n "$given" ]; then t="$E/a20_runm/exports/$tag/candidate.py"; if [ -e "$t" ]; then cmp "$given" "$t"; else cp "$given" "$t"; fi   # an existing copy must be the same bytes
  shasum -a 256 "$given"; else $PY -B "$E/stack/export_for_build.py" "$params" "$E/a20_runm/exports/$tag/candidate.py"; fi
levers=$($PY -B "$E/stack/levers_from_params.py" "$params")
PRE=${PRE:-T+C+B+F+D+Hfull}; POST=${POST-Phfull_v1+Rlvfuse_sized32+Rtsr_off}   # PRE=T+C+B+F+D POST= gives the screen stack (no strips, no list, no release fragments)
spec="$PRE+$levers${extra:++$extra}${POST:++$POST}"
echo "levers: $spec"
python "$E/stack/build.py" "$spec" "$E/a20_runm/exports/$tag/candidate.py" "$tag"
out="$E/stack/candidate_${spec//+/_}__$tag.py"
python "$P2/fable_porter_v1/einsum_site_diff.py" "$out" "$S37" | tail -1
shasum -a 256 "$out"
