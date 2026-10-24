"""TAG opus_engineer a62 KAPPA v1 (file written 2026-10-12 03:53Z; S50's 'kappa' family had NO native route: port v1f's family
check flags it). Torch (trainer/otr_pl.py, _patched_step): the card's step computes the birth weight as d (1 + kappa_l radial / 3) with
kappa_l = 1 + eps['kappa'][l]; F4's birth scalar (otr_m.py line 441, An = -alpha / (sigma (1 + radial / 3))) keeps the plain form, and so
does the native one (_f4c rule's A). Native: the step's birth line, edited by TEXT at its LAST occurrence (the active definition; the
earlier copies are dead code), to d (1 + (kappa_l radial) / 3) at the layers in _KAPPA and the unchanged line elsewhere, plus a block
`_KAPPA = {}` (the certificate: = the parent bit for bit). An optional values fragment (kappa_from_pt_v1.py) is appended after it.
usage: make_kappa_v1.py <parent.py> <out name> [<fragment_kappa_<name>.py>]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
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
OLD = 'birth = d * _e2_H(1 + _e2_H(radial / 3))'
NEW = ('birth = d * _e2_H(1 + _e2_H(radial / 3)) if l not in _KAPPA else d * _e2_H(1 + _e2_H(_e2_H(_KAPPA[l] * radial) / 3))'
       '   # opus-engineer a62 kappa: torch d (1 + kappa radial / 3)')
lines = parent.read_text().split('\n')
hits = [i for i, s in enumerate(lines) if s.strip() == OLD]
assert hits, 'birth line not found'
i = hits[-1]
lines[i] = lines[i][:len(lines[i]) - len(lines[i].lstrip())] + NEW
print('birth line: %d occurrence(s), replaced the last at line %d' % (len(hits), i + 1))
text = '\n'.join(lines).rstrip('\n') + '\n\n# BEGIN OPUS-ENGINEER a62 KAPPA v1 (radial birth weight per layer; make_kappa_v1.py)\n_KAPPA = {}\n# END OPUS-ENGINEER a62 KAPPA v1\n'
if values is not None:
    body = values.read_text()
    text += '\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (
        values.stem, sha(body.encode())[:12], body.strip('\n'), values.stem)
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), replaced_line=i + 1,
                                               occurrences=len(hits), values=None if values is None else str(values),
                                               candidate=str(out/'candidate.py'), candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8])
