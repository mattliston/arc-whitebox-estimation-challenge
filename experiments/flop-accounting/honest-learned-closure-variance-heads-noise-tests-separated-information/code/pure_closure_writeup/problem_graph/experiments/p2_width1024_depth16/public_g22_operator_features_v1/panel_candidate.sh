#!/usr/bin/env bash
# Score a candidate on the 8-net panel + the 6 virgin nets, two at a time (grader-style calls, independent 4M truths).
# Usage: panel_candidate.sh <candidate.py> <tag>   -> panel_<tag>.jsonl + a summary line
set -u
D="$(cd "$(dirname "$0")" && pwd)"; CAND="$1"; TAG="$2"; OUT="$D/panel_${TAG}.jsonl"; : > "$OUT"
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
NETS=(35 100 150 200 250 300 400 500 3016500 3016501 3016502 3016503 3016504 3016505)
for ((i=0; i<${#NETS[@]}; i+=2)); do
  a=${NETS[$i]}; b=${NETS[$((i+1))]}
  nice -n 10 $PY $D/score_candidate.py "$CAND" $a >> "$OUT" 2>/dev/null &
  nice -n 10 $PY $D/score_candidate.py "$CAND" $b >> "$OUT" 2>/dev/null &
  wait
done
$PY - "$OUT" <<'PYEOF'
import json, sys, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
panel = [r for r in rows if r["rank"] < 3016500]; virgin = [r for r in rows if r["rank"] >= 3016500]
for name, rs in (("panel8", panel), ("virgin6", virgin)):
    if rs:
        raw = np.mean([r["final_mse_nc"] for r in rs]); util = np.mean([r["util"] for r in rs]); adj = np.mean([r["adjusted"] for r in rs])
        print(f"{name}: n={len(rs)} mean raw {raw:.3e} mean util {util:.4f} mean adjusted {adj:.3e} | per-net raw " + " ".join(f"{r['rank']}:{r['final_mse_nc']:.2e}" for r in sorted(rs, key=lambda r: r['rank'])))
PYEOF
