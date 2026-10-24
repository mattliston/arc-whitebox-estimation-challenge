#!/usr/bin/env bash
# TAG fable-porter (PT-4 service): (3,1)-plane formula file + layers -> native candidate cand_t31_<tag>.py (build_t31_variant.py on the pinned
# bank) -> smoke + cost on rank 35 (t31_bill.py: exact flops, formula functions) -> paired VALID40 vs the CACHED bank rows
# (paired_eval_cached.py, 2 jobs x 1 thread, nice 10). Appends one summary JSON line to eval_service.log.
# usage: eval_t31.sh <formula.py> <layers e.g. 2-14> <tag> [ranks subset e.g. 37,39,42]
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"; P2="$(dirname "$HERE")"; NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python
BANK="$P2/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py"
F="$1"; LAYERS="$2"; TAG="$3"; RANKS="${4:-}"
CAND="$HERE/cand_t31_$TAG.py"
python3 "$HERE/build_t31_variant.py" "$BANK" "$F" "$LAYERS" "$CAND"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
nice -n 10 "$NPY" "$HERE/t31_bill.py" "$CAND" 35 > "$HERE/t31_bill_${TAG}_r35.log" 2>&1 || { echo "SMOKE FAILED: see t31_bill_${TAG}_r35.log"; tail -5 "$HERE/t31_bill_${TAG}_r35.log"; exit 1; }
grep '^{' "$HERE/t31_bill_${TAG}_r35.log"
nice -n 10 "$NPY" "$HERE/paired_eval_cached.py" "$CAND" "$HERE/bank_valid40_cache.json" 2 "$HERE/paired_valid40_t31_$TAG.json" $RANKS > "$HERE/paired_valid40_t31_$TAG.log" 2>&1
python3 - "$HERE/paired_valid40_t31_$TAG.json" "$HERE/t31_bill_${TAG}_r35.log" "$TAG" "$F" "$LAYERS" >> "$HERE/eval_service.log" <<'PY'
import json, sys, time, hashlib
P = json.load(open(sys.argv[1])); s = P['summary']
bill = [json.loads(l) for l in open(sys.argv[2]) if l.startswith('{')][-1]
dfl = [P['results'][f'cand:{r}']['adjusted'] / P['results'][f'cand:{r}']['mse'] * 2**41 - P['results'][f'parent:{r}']['flops'] for r in P['ranks'] if 'mse' in P['results'].get(f'cand:{r}', {})]
print(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M'), tag=sys.argv[3], formula=sys.argv[4], formula_sha256=hashlib.sha256(open(sys.argv[4], 'rb').read()).hexdigest()[:16],
                      layers=sys.argv[5], n=s['n'], adj_ratio=round(s['adj_ratio'], 5), raw_ratio=round(s['raw_ratio'], 5), adj_wins=s['adj_wins'], raw_wins=s['raw_wins'],
                      util_cand=round(s['util_cand'], 5), util_parent=round(s['util_parent'], 5), mean_flops_delta_per_net=round(sum(dfl) / len(dfl)),
                      r35_mse=bill['mse'], r35_formula_functions=bill['functions'])))
PY
tail -1 "$HERE/eval_service.log"
