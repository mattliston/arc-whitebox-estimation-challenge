#!/bin/bash
# TAG opus-trainer RAD S54 (as S53's hold, lead 07:1xZ: a round starts at 2 workers while the previous round's sealed stage 2 and port run): the
# HOLD of cert/RAD/maintain_workers_v2.sh for S54. Exit 0 (hold over) when S53 was NOT accepted (no accepted CHECK / RESERVE / SECOND verdict
# file), or release's lines carry S53's gate verdicts (lane-anchored fixed strings, by the S51 convention: 'TWIN s53tw (' with 'VERDICT:', AND
# 's53s<k> SAFE GATE PASS' or 's53s<k> SAFE: DO NOT FIRE'), or 3 h after S54's start. usage: bash cert/RAD/S54/hold_s54.sh <S54 start epoch>
cd "$(dirname "$0")/../../.."
O=..; PY=/Users/matt/anaconda3/bin/python
[ $(date +%s) -ge $(( $1 + 10800 )) ] && exit 0
$PY -c "import json, sys; from pathlib import Path; a = [json.loads(Path(p).read_text()).get('accepted') is True for p in ('cert/RAD/S53/CHECK_VERDICT_v1.json', 'cert/RAD/S53/RESERVE_VERDICT_v1.json', 'cert/RAD/S53/SECOND_VERDICT_v1.json') if Path(p).exists()]; sys.exit(1 if any(a) else 0)" && exit 0
grep '^\[release\] ' $O/TEAM_LOG.md | grep -F 'TWIN s53tw (' | grep -qF 'VERDICT:' && grep '^\[release\] ' $O/TEAM_LOG.md | grep -qE 's53s[0-9]+ SAFE (GATE PASS|: DO NOT FIRE)'
