#!/bin/bash
# TAG opus_engineer a27: native parity of the F4c port against the physicist's stored held-out arms (O/physicist/k4l_pred_rank*_fr48.npz).
#   FK  = F4 0.58 + K4L 0.73 (both ages)                                     -> stack T+C+B+F+D+F4v4t0.58,0.58+K4Lv3c0.73,0.73 on RRPI
#   MBK = the mixture + K4L 0.71: rule 0.378, exact S1 0.858 / S2 0.48 in the three-entry form (the rule without its own S1)
#         = F4C with c1 = 0.858 - 0.378 x s_new x s_carry^age (exact: the fit is linear in the S terms)       -> ...+F4Ct0.378c<c1 a1>,0.48,<c1 a2>,0.48+K4Lv4c0.71,0.71
# Two guarded single-thread native sessions, one after the other, on the ranks given (default 474,648: the first two of his list),
# then a27_f4c/parity_fr48.py. usage: run_parity.sh [<ranks>] [<name suffix>] [<torch tag: fr48 (default) or port (networks 35, 651)>]
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
E=$P2/opus_reexam_20261001_v1/engineer
ranks=${1:-474,648}; sfx=${2:-474_648}
cd "$E" || exit 1
C1=$(python3 -c "
s_carry, s_new = 1.0032287195131746, 0.9932375877606482
print(','.join(repr(0.858 - 0.378 * s_new * s_carry ** age) + ',0.48' for age in (1, 2)))")
FK="T+C+B+F+D+F4v4t0.58,0.58+K4Lv3c0.73,0.73"
MBK="T+C+B+F+D+F4Ct0.378c$C1+K4Lv4c0.71,0.71"
for spec in "$FK" "$MBK"; do
  out="$E/stack/candidate_${spec//+/_}.py"
  [ -e "$out" ] || /Users/matt/arc-white-box-estimation/.venv/bin/python -B stack/build.py "$spec" | tail -n 1 | cut -c1-300
done
python3 "$P2/fable_porter_v1/einsum_site_diff.py" "$E/stack/candidate_${MBK//+/_}.py" "$E/stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py" | tail -n 1
PER_NET=420 bash run_quick.sh "$E/stack/candidate_${FK//+/_}.py" "$ranks" "f4c_fk_$sfx" 2>&1 | cut -c1-400
PER_NET=420 bash run_quick.sh "$E/stack/candidate_${MBK//+/_}.py" "$ranks" "f4c_mbk_$sfx" 2>&1 | cut -c1-400
/Users/matt/arc-white-box-estimation/.venv/bin/python -B a27_f4c/parity_fr48.py "f4c_fk_$sfx" "f4c_mbk_$sfx" "${3:-fr48}"
