#!/bin/bash
# TAG opus-trainer RAD (10-12 13:3xZ; rule pre-declared in TEAM_LOG 13:29:03Z): runs cert/RAD/hold_decide_v1.py ONCE, when S51's downstream is over
# (release's S51 gate lines: 'TWIN s51tw (' with 'VERDICT:', AND 's51s<k> SAFE GATE PASS' / 'SAFE: DO NOT FIRE'), or S51 was NOT accepted with S52
# running for 30 min (no downstream to measure: the rule stays), or at 21:00Z at the latest; the decision is in force for S53 onward.
# usage (from O/trainer): nohup bash cert/RAD/hold_decide_v1.sh > cert/RAD/hold_decide_v1.out 2>&1 &
cd "$(dirname "$0")/../.."
O=..; PY=/Users/matt/anaconda3/bin/python; END=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-12T21:00:00Z" +%s)
done_rel() { grep '^\[release\] ' $O/TEAM_LOG.md | grep -F 'TWIN s51tw (' | grep -qF 'VERDICT:' && grep '^\[release\] ' $O/TEAM_LOG.md | grep -qE 's51s[0-9]+ SAFE (GATE PASS|: DO NOT FIRE)'; }
not_acc() { $PY -c "import json, sys; from pathlib import Path; f = [Path('cert/RAD/S51/%s' % n) for n in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json')]; d = [json.loads(p.read_text()) for p in f if p.exists()]; sys.exit(0 if d and not any(x.get('accepted') is True for x in d) and Path('run_RAD_S51_gen/STOP').exists() else 1)"; }
s52_since() { [ -e cert/RAD/S52/config.json ] && echo $(( $(date +%s) - $(stat -f %m cert/RAD/S52/config.json) )) || echo 0; }
until done_rel || { not_acc && [ $(s52_since) -ge 1800 ]; } || [ $(date +%s) -ge $END ]; do sleep 60; done
$PY -B cert/RAD/hold_decide_v1.py
