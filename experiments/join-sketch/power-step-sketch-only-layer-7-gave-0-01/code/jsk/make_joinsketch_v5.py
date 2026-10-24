"""[opus-lead] JSK v5 = JSK v4 applied ONLY at the join layers listed in the plan (default: L7 only, the flattest spectrum, where the
native L8-L10 gains held on most ranks); the other joins keep today's Nystrom sketch and Nystrom core verbatim. Build <out> from
<parent>, patching BOTH live join bodies. usage: make_joinsketch_v5.py <parent.py> <out.py> [plan json]"""
import json, sys, hashlib
from pathlib import Path
parent, out = Path(sys.argv[1]), Path(sys.argv[2]); assert not out.exists()
plan = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {7: (1024, 2048)}
plan = {int(k): tuple(int(x) for x in v) for k, v in plan.items()}
lines = parent.read_text().split('\n')
V4 = Path(__file__).with_name('make_joinsketch_v4.py').read_text()
def grab(name):
    s = V4.index(name + " = '''") + len(name) + 6; e = V4.index("'''", s); return V4[s:e]
NEW_SKETCH_V4 = grab('NEW_SKETCH'); NEW_CORE_AFTER_FJ_V4 = grab('NEW_CORE_AFTER_FJ'); INHERITED_V4 = grab('INHERITED')

def find_block(start_idx, first, last):
    i = next(k for k in range(start_idx, len(lines)) if lines[k].startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k] == last)
    return i, j

def indent(block, n=4):
    return '\n'.join((' ' * n + l) if l.strip() else l for l in block.split('\n'))

def patch_function(d):
    # 1. sketch: JSK branch for plan layers, today's block otherwise
    i, j = find_block(d, '        if _JOIN_NYSTROM:   # PT-42 Nystrom join', '            del _jn_d')
    assert j - i == 8, (i, j)
    today = '\n'.join(lines[i:j + 1])
    today_inner = today.split('\n', 1)[1]          # drop the 'if _JOIN_NYSTROM:' line
    jsk_inner = NEW_SKETCH_V4.split('\n', 1)[1]
    new = ('        if _JOIN_NYSTROM and layer in _JSK_PLAN:   # [opus-lead] JSK v5: v4 sketch at the planned joins only\n' + jsk_inner
           + '\n        elif _JOIN_NYSTROM:   # PT-42 Nystrom join (today, verbatim)\n' + today_inner)
    lines[i:j + 1] = new.split('\n')
    # 2. today's core block: keep, guarded
    i, j = find_block(d, '        if _need_core:   # opus-engineer a08', '            del R64')
    assert j - i == 4, (i, j)
    lines[i] = lines[i].replace('        if _need_core:', '        if _need_core and layer not in _JSK_PLAN:', 1)
    # today's del line deletes _jn_rows: fine for both branches
    # 3. v4 core after Fj, guarded; 4. inherited term, guarded
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('        Fj = _shared_old384_mm('))
    core = NEW_CORE_AFTER_FJ_V4.replace('        if _need_core:', '        if _need_core and layer in _JSK_PLAN:', 1).replace('\n        del _jn_cols1', '\n        if layer in _JSK_PLAN:\n            del _jn_cols1', 1)
    lines[k + 1:k + 1] = core.split('\n')
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('            overlap = _shared_old384_mm('))
    assert lines[k + 1].startswith('            factors = _slr_join_factors(layer, overlap, state, Fj)'), lines[k + 1]
    inh = INHERITED_V4.replace('            if _need_core:', '            if _need_core and layer in _JSK_PLAN:', 1)
    lines[k + 1:k + 1] = inh.split('\n')

ott = next(k for k in range(len(lines)) if lines[k] == '_OTT_ORIGINAL_PUSH = _shared_old384_push')
owner = max(k for k in range(ott) if lines[k].startswith('def _shared_old384_push('))
impl = [k for k in range(len(lines)) if lines[k].startswith('def _astra_bcc_push_impl(')]
assert len(impl) == 1 and impl[0] > owner
patch_function(impl[0]); patch_function(owner)
text = '\n'.join(lines)
assert text.count('JSK v5: v4 sketch at the planned joins only') == 2 and text.count('PT-42 Nystrom join (today, verbatim)') == 2
text = text.replace('\n_JOIN_NYSTROM = True\n', '\n_JOIN_NYSTROM = True\n_JSK_PLAN = %r   # [opus-lead] JSK v5: join layer -> (kc0 start columns, kc power-step columns); other joins unchanged\n' % (plan,), 1)
assert '_JSK_PLAN = ' in text
out.write_text(text); compile(text, str(out), 'exec')
print('wrote', out, 'sha8', hashlib.sha256(text.encode()).hexdigest()[:8], 'plan', plan)
