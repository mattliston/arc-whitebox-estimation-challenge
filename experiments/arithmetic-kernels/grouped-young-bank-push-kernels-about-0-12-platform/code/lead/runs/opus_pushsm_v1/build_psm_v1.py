"""[opus-pushsm] build: candidate = GOV v4g 805c3598 + fragment_pushsm_v1 (+ _PUSHSM_RULE from RULE_pushsm_v1.json if present);
test copies with overrides. usage: build_psm_v1.py release | build_psm_v1.py test <name> NAME=VALUE ..."""
import hashlib
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
base = (D.parent/'opus_governor_v1/candidate_GOV_v4g.py').read_bytes()
assert hashlib.sha256(base).hexdigest().startswith('805c3598')
frag = (D/'fragment_pushsm_v1.py.txt').read_text()
rule = D/'RULE_pushsm_v1.json'
if rule.exists():
    table = {tuple(map(int, key.split('x'))): bits for key, bits in json.loads(rule.read_text())['rule'].items() if bits}
    assert frag.count('_PUSHSM_RULE = {}\n') == 1
    frag = frag.replace('_PUSHSM_RULE = {}\n', '_PUSHSM_RULE = ' + repr(dict(sorted(table.items()))) + '\n')
src = base.decode() + frag
if sys.argv[1] == 'release':
    out = D/'candidate_PSM_v1.py'
else:
    name = sys.argv[2]
    lines = {'_PUSHSM_FORCE': '_PUSHSM_FORCE = False ', '_PUSHSM_ON': '_PUSHSM_ON = True\n'}
    for kv in sys.argv[3:]:
        k, v = kv.split('=')
        if k in lines:
            old = lines[k]
            assert src.count(old) == 1, old
            src = src.replace(old, f"{k} = {v}" + (' ' if old.endswith(' ') else '\n'))
        else:   # governor thresholds (parity copies)
            import re
            pat = re.compile(r'^' + re.escape(k) + r' = [^\n]*$', re.M)
            assert len(pat.findall(src)) == 1, k
            src = pat.sub(f'{k} = {v}', src)
    out = D/f'test_{name}.py'
out.open('x').write(src)
print(out.name, hashlib.sha256(out.read_bytes()).hexdigest()[:8])
