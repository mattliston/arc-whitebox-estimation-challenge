#!/usr/bin/env bash
# Gate a candidate on its paired VALID40 result, then the 14-net panel vs the bank's panel, then fire (pre-approved).
# Usage: gate_and_fire.sh <paired.json> <cand.py> <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; P2="$(dirname "$HERE")"; ROOT="$(cd "$P2/../../../.." && pwd)"
PJ="$1"; CAND="$2"; TAG="$3"; LOG="$HERE/gate_$TAG.log"
python3 - "$PJ" > "$HERE/gate_${TAG}_paired.txt" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))['summary']
ok = s['adj_ratio'] < 0.975 and s['adj_wins'] >= 24
print(json.dumps(dict(pass_paired=ok, adj_ratio=s['adj_ratio'], raw_ratio=s['raw_ratio'], adj_wins=s['adj_wins'], n=s['n'], util_cand=s['util_cand'], util_parent=s['util_parent'])))
PY
cat "$HERE/gate_${TAG}_paired.txt" | tee -a "$LOG"
if ! grep -q '"pass_paired": true' "$HERE/gate_${TAG}_paired.txt"; then echo "PAIRED GATE FAILED for $TAG" | tee -a "$LOG"; exit 0; fi
bash "$P2/public_g22_operator_features_v1/panel_candidate.sh" "$CAND" "$TAG" > "$HERE/panel_$TAG.log" 2>&1
grep -E "panel8|virgin6" "$HERE/panel_$TAG.log" | tee -a "$LOG"
python3 - "$HERE/panel_$TAG.log" "$HERE/panel_bank331824.log" > "$HERE/gate_${TAG}_panel.txt" <<'PY'
import re,sys
def parse(f):
    d={}
    for line in open(f):
        m=re.match(r'(panel8|virgin6): n=(\d+) mean raw ([0-9.e+-]+) mean util ([0-9.]+) mean adjusted ([0-9.e+-]+)', line)
        if m: d[m.group(1)]=dict(n=int(m.group(2)), raw=float(m.group(3)), util=float(m.group(4)), adj=float(m.group(5)))
    return d
c=parse(sys.argv[1]); b=parse(sys.argv[2])
ok = all(k in c and k in b for k in ('panel8','virgin6')) and c['panel8']['adj'] < 0.98*b['panel8']['adj'] and c['virgin6']['adj'] < 0.98*b['virgin6']['adj']
print({'pass_panel': ok, 'cand': c, 'bank': b})
PY
cat "$HERE/gate_${TAG}_panel.txt" | tee -a "$LOG"
if ! grep -q "'pass_panel': True" "$HERE/gate_${TAG}_panel.txt"; then echo "PANEL GATE FAILED for $TAG" | tee -a "$LOG"; exit 0; fi
cd "$ROOT" && bash autoresearch/fire_candidate.sh "$CAND" "$TAG: 331824 + $(cat "$HERE/gate_${TAG}_paired.txt" | cut -c1-120)" "$HERE/fire_$TAG" 2>&1 | tee -a "$LOG"
