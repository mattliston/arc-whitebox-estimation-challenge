#!/bin/bash
# Prepare one bracket arm for a possible fire: strip the retired governor (same edit as #337641), wait for the Mac to be free
# (no Mac32 judge, memory free >= 52%), run the unchanged release_min_v1.sh gates, print the condition checks. Does NOT fire.
# Usage: strip_gate_arm_v1.sh <arm name>
N=$1
R=/Users/matt/arc-white-box-estimation; O=$R/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1; F=$O/lead/runs/fable_oct21_final_v1
[ -d $F/stacks/$N ] || { echo "no stack dir $N"; exit 1; }
for ROLE in release science; do SRC=$F/stacks/$N/candidate_${N}_${ROLE}_v1.py; [ -f $SRC ] || { echo "missing $SRC"; exit 1; }
  if grep -q "_active_context\|get_active_budget\|residual_wall_time_s" $SRC; then cp $SRC $F/stacks/$N/candidate_${N}_${ROLE}_v1_before_govstrip.py; python3 - "$SRC" <<'PYEOF'
import ast, sys
p=sys.argv[1]; text=open(p).read(); lines=text.splitlines(keepends=True); tree=ast.parse(text)
targets=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_fast_rescue_live_residual']; assert len(targets)==3
new="def _fast_rescue_live_residual():\n    return None   # retired residual governor removed 2026-10-21: never called (_FAST_RESCUE_FRAC = {} is the last line); no meter access anywhere\n"
for n in sorted(targets, key=lambda n: n.lineno, reverse=True): lines[n.lineno-1:n.end_lineno]=[new]
for i,l in enumerate(lines):
    if l.startswith('#') and ('_active_context' in l or 'get_active_budget' in l or 'residual_wall_time_s' in l): lines[i]='# (comment removed 2026-10-21: described the retired residual governor, which no longer exists)\n'
out=''.join(lines); compile(out, p, 'exec'); assert '_active_context' not in out and 'get_active_budget' not in out and 'residual_wall_time_s' not in out; assert out.rstrip().endswith('_FAST_RESCUE_FRAC = {}')
open(p,'w').write(out); print('stripped', p.split('/')[-1])
PYEOF
  else echo "already stripped: $(basename $SRC)"; fi
done
busy() { pgrep -f "^/Users/matt/arc-white-box-estimation/.venv/bin/python -B .*dispatch_priority32_isolated_v1.py" > /dev/null; }
while busy; do sleep 10; done
for i in $(seq 1 30); do FREE=$(memory_pressure 2>/dev/null | grep -o "free percentage: [0-9]*" | grep -o "[0-9]*$"); [ -n "$FREE" ] && [ "$FREE" -ge 52 ] && break; sleep 5; done
REL=$F/stacks/$N/candidate_${N}_release_v1.py; SHA8=$(shasum -a 256 $REL | cut -c1-8); TAG=fable_${N}_govstrip_${SHA8}_v1; Q=$F/qual_$TAG; REF=$(ls $O/release/runs/fable_extgate_s15_cap3_*_v1/submission.tar.gz | head -1)
echo "=== GATES $N (stripped) tag $TAG $(date -u +%H:%M:%SZ)"; bash $O/lead/runs/fable_oct19_l13rank_v1/release_min_v1.sh "$REL" "$TAG" "$Q" "$REF" > $Q.gates.log 2>&1; grep -E "SETUP5|STAGE A|PREFIRE|GATES|FAIL" $Q.gates.log | cut -c1-160
A=$O/release/runs/$TAG/submission.tar.gz
NF=$(grep -c '^FAIL' $Q/prefire.log 2>/dev/null); GC=$(grep -c "FAIL imports outside the allowed set: \['gc'\]" $Q/prefire.log 2>/dev/null); SIG=$(grep -c "new signatures: 0" $Q/prefire.log 2>/dev/null); SU=$(python3 -c "import json;print(json.load(open('$Q/setup5/SUMMARY_v1.json'))['status'])" 2>/dev/null); XS=$(grep -c "_XSG_ON = set()" $REL); GV=$(grep -c "_active_context\|residual_wall_time_s\|get_active_budget" $REL); CLR=$(grep -c "GATES CLEAR" $Q.gates.log)
echo "checks: FAIL=$NF gc=$GC newsig0=$SIG setup5=$SU xsg_off=$XS governor_refs=$GV clear=$CLR archive=$(test -f $A && shasum -a 256 $A | cut -c1-12)"
if [ "$NF" = "1" ] && [ "$GC" = "1" ] && [ "$SIG" = "1" ] && [ "$SU" = "PASS" ] && [ "$XS" = "1" ] && [ "$GV" = "0" ] && [ "$CLR" = "1" ] && [ -f "$A" ]; then echo "READY: $TAG $A"; else echo "NOT READY"; fi
