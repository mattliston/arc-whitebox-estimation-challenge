"""TAG opus_engineer a62 v2 (file written 2026-10-12T17:39Z; lead 17:3xZ decision 2) = v1 + the CARRY rule and a LEDGER: every delta has "carry": "native"
(re-applied on every round's files) or "card" (skipped on round N once a trainer line '[trainer] <stamp> ABSORBED <id> at S<M>' with M <= N is
in TEAM_LOG, matched at the line start: the port carries it from S<M> on); each build appends {tag, round, applied, skipped} to
a62_modules/stack_ledger_v1.jsonl, and an id the previous build applied that is neither in this list nor absorbed is printed as DROPPED.
usage: stack_deltas_v2.py <deltas.json> <tag> <round N> <safe source> <twin source>. v1 header: (file written 2026-10-12T17:31Z; lead 17:0xZ cadence: a twin fire about every 4 h from verified small deltas): reads a
stack's delta list and checks it against BOTH physics files of the round (safe source and twin source), then prints the builder's inputs.
deltas.json = {"deltas": [<delta>, ...]} in the order to apply, each with an "id" and its "evidence" (where it was verified), and one of
  {"kind": "literal", "name": "_SOME_CONSTANT", "value": "<python expression>"}   re-assigned at module end: valid only for a constant read at
      call time (setup or predict), as the delta was verified (an appended fragment on the round's file); the name must already be assigned
      at module level in both files;
  {"kind": "fragment", "path": "/abs/fragment.py"}   a self-contained block appended as it is (make_s49_v1's BEGIN / END lines and sha);
  {"kind": "edit", "tool": "make_<x>_v1.py", "values": "/abs/values fragment"}   a text edit by one of this lane's tools (it changes code, so
      the build takes the refresh path: census, prebind lists and XSG remade).
Writes a62_modules/fragment_stack_<tag>_lits.py (the literals, each after an assert that the name exists) when there are literals.
Prints 'FRAGMENTS <paths in order>' (the literal fragment last), 'EDITS <tool>:<values> ...' and 'REFRESH yes|no' (yes when any edit, or any
delta says "shapes": "changed").
usage: stack_deltas_v1.py <deltas.json> <tag> <safe source candidate.py> <twin source candidate.py>"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec, tag, N, safe, twin = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
log = (HERE.parents[1]/'TEAM_LOG.md').read_text()
absorbed = {m.group(1): int(m.group(2)) for m in re.finditer(r'^\[trainer\] \S+ ABSORBED (\S+) at S(\d+)', log, re.M)}
deltas, skipped = [], []
for d in json.load(open(spec))['deltas']:
    assert d.get('carry') in ('native', 'card'), '%s: carry must be native or card' % d['id']
    (skipped if d['carry'] == 'card' and absorbed.get(d['id'], 10 ** 9) <= N else deltas).append(d)
texts = {'safe': safe.read_text(), 'twin': twin.read_text()}
frags, edits, lits, refresh, bad = [], [], [], False, []
for d in deltas:
    did, kind = d['id'], d['kind']
    refresh |= d.get('shapes') == 'changed'
    if kind == 'literal':
        compile(d['value'], did, 'eval')
        for which, text in texts.items():
            if not re.search(r'^%s\s*=' % re.escape(d['name']), text, re.M):
                bad.append('%s: %s is not assigned at module level in the %s file' % (did, d['name'], which))
        lits.append(d)
    elif kind == 'fragment':
        p = Path(d['path'])
        if not p.is_file():
            bad.append('%s: no fragment %s' % (did, p))
        frags.append(str(p))
    elif kind == 'edit':
        if not (HERE/d['tool']).is_file() or not Path(d['values']).is_file():
            bad.append('%s: tool %s or values %s missing' % (did, d['tool'], d['values']))
        edits.append('%s:%s' % (d['tool'], d['values']))
        refresh = True
    else:
        bad.append('%s: unknown kind %s' % (did, kind))
if bad:
    print('BAD ' + '; '.join(bad))
    sys.exit(1)
if lits:
    out = HERE/('fragment_stack_%s_lits.py' % tag)
    assert not out.exists(), out
    body = ['# opus-engineer a62 stack %s literals (stack_deltas_v1.py, %s sha256 %s): each name was assigned in the parent; re-assigned here'
            % (tag, spec.name, hashlib.sha256(spec.read_bytes()).hexdigest()[:12])]
    for d in lits:
        body += ['assert %r in globals(), %r' % (d['name'], d['name']), '%s = %s   # %s: %s' % (d['name'], d['value'], d['id'], d.get('evidence', ''))]
    out.write_text('\n'.join(body) + '\n')
    frags.append(str(out))
ledger = HERE/'stack_ledger_v1.jsonl'
last = json.loads(ledger.read_text().splitlines()[-1]) if ledger.exists() and ledger.read_text().strip() else None
ids = [d['id'] for d in deltas]
dropped = [i for i in (last['applied'] if last else []) if i not in ids and i not in [d['id'] for d in skipped] and absorbed.get(i, 10 ** 9) > N]
with open(ledger, 'a') as f:
    f.write(json.dumps(dict(tag=tag, round=N, spec=str(spec), spec_sha=hashlib.sha256(spec.read_bytes()).hexdigest()[:12], applied=ids,
                            skipped=[d['id'] for d in skipped], dropped=dropped, utc=__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()))) + '\n')
print('SKIPPED ' + ' '.join(d['id'] for d in skipped))
print('DROPPED ' + ' '.join(dropped))
print('FRAGMENTS ' + ' '.join(frags))
print('EDITS ' + ' '.join(edits))
print('REFRESH ' + ('yes' if refresh else 'no'))
