#!/bin/bash
# TAG opus-trainer RAD S53 hold v2 (written 2026-10-12T13:23:23Z; lead 13:3xZ: the 2-worker hold at a round's start costs about 20 worker-hours a round): the
# HOLD of cert/RAD/maintain_workers_v4.sh for S53, with the RULE read from cert/RAD/HOLD_RULE at every check (one word; a decision is a logged
# rewrite of that file, not a code change). Exit 0 = hold over (the maintainer then goes to 6 workers once memory_pressure free >= 60% for 5 min,
# then ramps as usual). Always over: 3 h after S53's start, or S52 NOT accepted (no accepted CHECK / RESERVE / SECOND verdict file).
#   release      (default; = hold_s53.sh): release's S52 gate lines: 'TWIN s52tw (' with 'VERDICT:', AND 's52s<k> SAFE GATE PASS' / 'SAFE: DO NOT FIRE'.
#   ready_table  (lead's 13:3xZ proposal): the engineer's 'S52 FILES READY FOR RELEASE:' (or 'S52 PORT STOPPED') AND the physicist's
#                'S52 STAGE 2 TABLE:' lines, i.e. the port and stage 2 are done; only release's gates still run.
# usage: bash cert/RAD/S53/hold_s53_v2.sh <S53 start epoch>
cd "$(dirname "$0")/../../.."
O=..; PY=/Users/matt/anaconda3/bin/python
[ $(date +%s) -ge $(( $1 + 10800 )) ] && exit 0
$PY -c "import json, sys; from pathlib import Path; a = [json.loads(Path(p).read_text()).get('accepted') is True for p in ('cert/RAD/S52/CHECK_VERDICT_v1.json', 'cert/RAD/S52/RESERVE_VERDICT_v1.json', 'cert/RAD/S52/SECOND_VERDICT_v1.json') if Path(p).exists()]; sys.exit(1 if any(a) else 0)" && exit 0
RULE=$(head -1 cert/RAD/HOLD_RULE 2>/dev/null | tr -d ' \n'); [ -n "$RULE" ] || RULE=release
case "$RULE" in
  ready_table)
    grep '^\[engineer\] ' $O/TEAM_LOG.md | grep -qE 'S52 (FILES READY FOR RELEASE:|PORT STOPPED)' && grep '^\[physicist\] ' $O/TEAM_LOG.md | grep -qF 'S52 STAGE 2 TABLE:' ;;
  *)
    grep '^\[release\] ' $O/TEAM_LOG.md | grep -F 'TWIN s52tw (' | grep -qF 'VERDICT:' && grep '^\[release\] ' $O/TEAM_LOG.md | grep -qE 's52s[0-9]+ SAFE (GATE PASS|: DO NOT FIRE)' ;;
esac
