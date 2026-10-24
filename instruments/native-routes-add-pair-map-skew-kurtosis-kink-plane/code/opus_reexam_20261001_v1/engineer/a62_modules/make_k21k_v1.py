"""TAG opus_engineer2 a62 (file written 2026-10-13T00:07:41Z; trainer 23:50Z NEW RAD FAMILIES FOR S54: k21k per band): the K21 scale on the KINK
rows of the GN covariance correction, BY LAYER. Torch (trainer/otr_pl_v2.py cov_pm, layers 2-14): dK21 = iso kink (k3 / v) Cpre + k21n (1 - kink) K21
+ k21k kink K21 with kink = |alpha| < 1.5 and k21k = eps['k21k'][l] (base 0). Native: both forms of _gn_dcov (the G5 one and the older one it
falls back to) build the K21 term from the row factor k21n (1 - kink); edited by TEXT, that factor becomes _k21k_rowf(l, th, kink, _e2_H) = the
parent's own product (same operations, same order) plus k21k kink at the layers listed in _K21K_BY_LAYER ({layer l: k21k}), the same rows
times K21 as torch. Native GN layers = 2-14 (_GN_COV_LAYERS), as torch's cov_pm. A block _K21K_BY_LAYER = {} (= the parent bit for bit: one
dict lookup per call, no array operation); an optional values fragment sets layers.
usage: make_k21k_v1.py <parent.py> <out name> [<fragment_k21kv_<name>.py>]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name = Path(sys.argv[1]).resolve(), sys.argv[2]
values = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
out = HERE/'s49'/name
assert not out.exists(), out
sha = lambda b: hashlib.sha256(b).hexdigest()
text0 = parent.read_text()
lines = text0.split('\n')
OLD, NEW = "_e2_H(th['k21n']) * _e2_H(1 - kink)", "_k21k_rowf(l, th, kink, _e2_H)"
defs = [i for i, s in enumerate(lines) if s.startswith('def _gn_dcov(l, mu, Cpre, K21, k4, aux):')]
assert len(defs) == 2, 'def _gn_dcov(l, ...): %d matches (expected the older form and G5)' % len(defs)
assert not [s for s in lines if re.match(r'_gn_dcov\s*=', s)] and 'global _gn_dcov' not in text0, '_gn_dcov is rebound outside its two defs'
assert text0.count(OLD) == 2, 'the k21n row factor: %d matches' % text0.count(OLD)
edited = []
for d in defs:
    end = next(i for i in range(d + 1, len(lines)) if lines[i] and not lines[i][0].isspace())
    hits = [i for i in range(d, end) if OLD in lines[i]]
    assert len(hits) == 1 and lines[hits[0]].startswith('    dK21 = ') and lines[hits[0]].count(OLD) == 1, (d + 1, hits)
    assert any(lines[i] == '    kink = (_e2_H(fnp.abs(a)) < 1.5).astype(fnp.float64)' for i in range(d, hits[0])), 'kink not formed as expected'
    lines[hits[0]] = lines[hits[0]].replace(OLD, NEW) + '   # opus-engineer2 a62 k21k by layer (make_k21k_v1.py)'
    edited.append(hits[0] + 1)
print('_gn_dcov: the K21 row factor edited at file lines %s' % edited)
helper = '''
# BEGIN OPUS-ENGINEER2 a62 K21K BY LAYER v1 (make_k21k_v1.py)
_K21K_BY_LAYER = {}


def _k21k_rowf(l, th, kink, hold):
    # the K21 row factor of the GN covariance correction: the parent's k21n (1 - kink) (same operations, same order), plus k21k kink at the
    # layers listed (torch otr_pl_v2 cov_pm: + (k21k * kink)[:, None] * K21, base 0)
    rowf = hold(th['k21n']) * hold(1 - kink)
    c = _K21K_BY_LAYER.get(l)
    if c is None:
        return rowf
    return hold(rowf) + hold(c * kink)
# END OPUS-ENGINEER2 a62 K21K BY LAYER v1
'''
text = '\n'.join(lines).rstrip('\n') + '\n' + helper
if values is not None:
    body = values.read_text()
    text += '\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (values.stem, sha(body.encode())[:12], body.strip('\n'), values.stem)
out.mkdir(parents=True)
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), edited_lines=edited,
                                               values=None if values is None else str(values), candidate=str(out/'candidate.py'),
                                               candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8])
