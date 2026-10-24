"""[opus-governor] build v2: release candidate = S2tr 46e9315d + fragment_governor_v2; test copies = that + overrides + trace v3.
usage: build_v2.py release | build_v2.py test <name> <force|None> [cap|-] [notrace]"""
import hashlib
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
base = (D.parent/'opus_s2tr_v1/candidate_S2tr_v1.py').read_bytes()
assert hashlib.sha256(base).hexdigest().startswith('46e9315d')
gov = (D/'fragment_governor_v2.py.txt').read_text()


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


if sys.argv[1] == 'release':
    out = D/'candidate_GOV_v2.py'
    src = base.decode() + gov
else:
    name, force = sys.argv[2], sys.argv[3]
    g = sub(gov, "_GOV_FORCE = None ", f"_GOV_FORCE = {force} ")
    if len(sys.argv) > 4 and sys.argv[4] != '-':
        g = sub(g, "_GOV_CAP_S = 0.385", f"_GOV_CAP_S = {sys.argv[4]}")
    trace = '' if sys.argv[5:] == ['notrace'] else sub((D/'fragment_trace_v3.py.txt').read_text(), '__GT_PATH__',
                                                         str(D/f'trace_{name}.jsonl'))
    out = D/f'test_{name}.py'
    src = base.decode() + g + trace
out.open('x').write(src)
print(out.name, hashlib.sha256(out.read_bytes()).hexdigest()[:8])
