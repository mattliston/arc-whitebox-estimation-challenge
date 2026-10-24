"""[opus-governor] build: release candidate = S2tr 46e9315d + fragment_governor_v1; test copies = that + constant overrides + trace.
usage: build_v1.py release | build_v1.py test <name> <force|None> <actions comma list> [cap|-] [notrace]"""
import hashlib
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
BASE = D.parent/'opus_s2tr_v1/candidate_S2tr_v1.py'
base = BASE.read_bytes()
assert hashlib.sha256(base).hexdigest().startswith('46e9315d')
gov = (D/'fragment_governor_v1.py.txt').read_text()


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


if sys.argv[1] == 'release':
    out = D/'candidate_GOV_v1.py'
    src = base.decode() + gov
else:
    name, force, actions = sys.argv[2], sys.argv[3], tuple(a for a in sys.argv[4].split(',') if a)
    g = sub(gov, "_GOV_FORCE = None ", f"_GOV_FORCE = {force} ")
    g = sub(g, "_GOV_ACTIONS = ('priced', 'full')", f"_GOV_ACTIONS = {actions!r}")
    if len(sys.argv) > 5 and sys.argv[5] != '-':
        g = sub(g, "_GOV_CAP_S = 0.385", f"_GOV_CAP_S = {sys.argv[5]}")
    trace = sub((D/'fragment_trace_v2.py.txt').read_text(), '__GT_PATH__', str(D/f'trace_{name}.jsonl'))
    if sys.argv[6:] == ['notrace']:
        trace = ''
    out = D/f'test_{name}.py'
    src = base.decode() + g + trace
out.open('x').write(src)
print(out.name, hashlib.sha256(out.read_bytes()).hexdigest()[:8])
