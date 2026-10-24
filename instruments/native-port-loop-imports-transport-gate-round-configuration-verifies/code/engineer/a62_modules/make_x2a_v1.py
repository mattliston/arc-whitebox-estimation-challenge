"""TAG opus_engineer a62 (file written 2026-10-12 17:14Z; lead 16:4xZ / radlab: x2a1 at L7, t -3.27): F4c exact-term strengths BY BIRTH LAYER.
Torch (trainer/otr_m.py with F4C = physicist/f4b.py mode 'mix'): at the end of step l (1..14) the cohort born at j = l - age is shifted with
the strengths cs[age] = (eps['x1a<age>'][l], eps['x2a<age>'][l]), read at the CURRENT layer l. Native (F4c v2 and v3): the strengths of
every age are folded into the newborn cohort's table at its birth j through one constant made once per process by _f4c_const from _F4C_C
(C['mixb'] in v2, C['mix'] in v3). Edited by TEXT: _f4c_const also makes that constant for each birth listed in _F4C_C_BY_BIRTH
({birth j: ((S1, S2) of age 1, (S1, S2) of age 2)} = torch's values at layers j + 1 and j + 2), with the parent's own expressions copied
verbatim, and _f4c_birth uses it at those births (C[<key>_by_birth].get(l, C[<key>]): the parent's constant at every other birth). The
constants are made where the parent makes its own (setup, through the prebind fill), so the predict path has no new operation.
A block _F4C_C_BY_BIRTH = {} (= the parent bit for bit); an optional values fragment sets the births.
usage: make_x2a_v1.py <parent.py> <out name> [<fragment_x2av_<name>.py>]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name = Path(sys.argv[1]).resolve(), sys.argv[2]
values = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
out = HERE/'s49'/name
out.mkdir(parents=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
V2 = "    exact = fnp.einsum('alsqj,lsjk,lk->aqsk', C['mixb'], _e2_H(numer * scale), source, optimize=False)"
V3 = "    exact = _e2_H(fnp.einsum('als,lsjk,lk->asjk', C['mix'], numer, source, optimize=False)) * scale"
lines = parent.read_text().split('\n')
hits = [(i, 'mixb') for i, s in enumerate(lines) if s == V2] + [(i, 'mix') for i, s in enumerate(lines) if s == V3]
assert len(hits) == 1, 'the exact-term einsum of _f4c_birth: %d matches' % len(hits)
e, key = hits[0]
c0 = [i for i, s in enumerate(lines) if s == 'def _f4c_const():']
assert len(c0) == 1 and c0[0] < e
r = next(i for i in range(c0[0], len(lines)) if lines[i] == '    return C')
assert next(s for s in lines[r + 1:] if s.strip()).startswith('def _f4c_birth('), '_f4c_const does not end where expected'
m = [i for i in range(c0[0], r) if lines[i].startswith('    mix = [[[strength[i][0] * -1.0 * sign')]
assert len(m) == 1 and lines[m[0] + 1].startswith('            [strength[i][1] * 2.0 * sign') and lines[m[0] + 1].endswith('for i in range(_F4C_AGE)]')
copy = lines[m[0]:m[0] + 2]
if key == 'mixb':
    b = [i for i in range(c0[0], r) if lines[i].startswith('    mixb = [[[[[mix[i][src][s] * basis[s][q][j]')]
    assert len(b) == 1
    copy += [lines[b[0]]]
assert any("%s=fnp.asarray(%s, dtype=f64)" % (key, key) in s for s in lines[c0[0]:r]), 'the parent makes C[%r] differently' % key
block = (['    # opus-engineer a62 F4c strengths by birth layer (make_x2a_v1.py): the constant above, remade by the same expressions for each listed birth',
          '    by_birth = {}',
          '    for birth, strength in _F4C_C_BY_BIRTH.items():',
          '        if len(strength) != _F4C_AGE:',
          "            raise RuntimeError('F4c: one strength pair per age at each listed birth')"]
         + ['    ' + s for s in copy]
         + ['        by_birth[birth] = fnp.asarray(%s, dtype=f64)' % key,
            "    C['%s_by_birth'] = by_birth" % key])
lines[e] = V2.replace("C['mixb']", "C['mixb_by_birth'].get(l, C['mixb'])") if key == 'mixb' else V3.replace("C['mix']", "C['mix_by_birth'].get(l, C['mix'])")
lines[e] += '   # opus-engineer a62 F4c strengths by birth layer'
lines[r:r] = block
print('F4c %s: einsum line %d, %d lines into _f4c_const before its return (line %d)' % ('v2 (mixb)' if key == 'mixb' else 'v3 (mix)', e + 1, len(block), r + 1))
text = '\n'.join(lines).rstrip('\n') + '\n\n# BEGIN OPUS-ENGINEER a62 F4C BY BIRTH v1 (make_x2a_v1.py)\n_F4C_C_BY_BIRTH = {}\n# END OPUS-ENGINEER a62 F4C BY BIRTH v1\n'
if values is not None:
    body = values.read_text()
    text += '\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (values.stem, sha(body.encode())[:12], body.strip('\n'), values.stem)
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), version=key, einsum_line=e + 1,
                                               values=None if values is None else str(values), candidate=str(out/'candidate.py'),
                                               candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8])
