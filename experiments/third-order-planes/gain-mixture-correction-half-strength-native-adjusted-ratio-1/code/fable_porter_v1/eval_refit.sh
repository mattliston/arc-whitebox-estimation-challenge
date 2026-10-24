#!/usr/bin/env bash
# TAG fable-porter (PT-3 evaluation service): refit params .pt + gamma -> native candidate (build_refit_candidate.py: bank + b fix at
# gamma + refit literals) -> paired VALID40 vs the bank (fable_refit_v1/paired_eval.py, 2 jobs, 1 BLAS thread each, nice 10).
# usage: eval_refit.sh <params.pt> <gamma> <tag>
# outputs (fable_porter_v1/): cand_<tag>.py, paired_valid40_<tag>.{json,log}; one summary JSON line appended to eval_service.log
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"; P2="$(dirname "$HERE")"
NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python
BANK="$P2/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py"
PT="$1"; G="$2"; TAG="$3"
python3 "$HERE/build_refit_candidate.py" "$PT" "$G" "$HERE/cand_$TAG.py"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
nice -n 10 "$NPY" "$P2/fable_refit_v1/paired_eval.py" "$HERE/cand_$TAG.py" "$BANK" valid40 2 "$HERE/paired_valid40_$TAG.json" > "$HERE/paired_valid40_$TAG.log" 2>&1
python3 - "$HERE/paired_valid40_$TAG.json" "$TAG" "$PT" "$G" >> "$HERE/eval_service.log" <<'PY'
import json, sys, time
s = json.load(open(sys.argv[1]))['summary']
print(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M'), tag=sys.argv[2], params=sys.argv[3], gamma=float(sys.argv[4]), n=s['n'],
                      adj_ratio=round(s['adj_ratio'], 5), raw_ratio=round(s['raw_ratio'], 5), adj_wins=s['adj_wins'], raw_wins=s['raw_wins'],
                      util_cand=round(s['util_cand'], 5), util_parent=round(s['util_parent'], 5))))
PY
tail -1 "$HERE/eval_service.log"
