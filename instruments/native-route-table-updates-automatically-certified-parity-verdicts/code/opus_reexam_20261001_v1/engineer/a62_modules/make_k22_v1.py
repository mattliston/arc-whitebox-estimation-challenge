"""TAG opus_engineer a62 (file written 2026-10-12 16:49Z; lead 16:4xZ / radlab: k22 at L14, t +3.40): k22 BY LAYER. Torch (trainer/otr_m_v5.py pair_in,
layers 1-14): k22 = diagonal_scatter(pin[4] x (1 + eps['k22'][l]), diagonal(pin[4])): the OFF-diagonal part of the isotropic (2,2) plane the
pair map receives is scaled, its diagonal kept. Native: the core loop's line 'k22 = _ps32(radial) * (_ps_vv32 + 2 * _ps32(_e2_CC)) / 3' is
followed by fnp.fill_diagonal(k22, <the unscaled diagonal>), so scaling the scalar radial on that line scales exactly the off-diagonal part,
at zero extra flops. Edited by TEXT at its last occurrence to use _ps32(radial * _K22_BY_LAYER[l]) at the layers listed (the unchanged
expression elsewhere), plus a block _K22_BY_LAYER = {} (= the parent bit for bit); an optional values fragment sets {layer: 1 + shift}.
usage: make_k22_v1.py <parent.py> <out name> [<fragment_k22v_<name>.py>]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
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
OLD = 'k22 = _ps32(radial) * (_ps_vv32 + 2 * _ps32(_e2_CC)) / 3'
NEW = ('k22 = (_ps32(radial) if l not in _K22_BY_LAYER else _ps32(radial * _K22_BY_LAYER[l])) * (_ps_vv32 + 2 * _ps32(_e2_CC)) / 3'
       '   # opus-engineer a62 k22 by layer: torch off-diagonal k22 scale (the diagonal is refilled below)')
lines = parent.read_text().split('\n')
hits = [i for i, s in enumerate(lines) if s.strip() == OLD]
assert hits, 'k22 line not found'
i = hits[-1]
lines[i] = lines[i][:len(lines[i]) - len(lines[i].lstrip())] + NEW
assert lines[i + 1].strip().startswith('fnp.fill_diagonal(k22,'), 'the diagonal refill no longer follows the k22 line'
print('k22 line: %d occurrence(s), edited the last at line %d' % (len(hits), i + 1))
text = '\n'.join(lines).rstrip('\n') + '\n\n# BEGIN OPUS-ENGINEER a62 K22 BY LAYER v1 (make_k22_v1.py)\n_K22_BY_LAYER = {}\n# END OPUS-ENGINEER a62 K22 BY LAYER v1\n'
if values is not None:
    body = values.read_text()
    text += '\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (values.stem, sha(body.encode())[:12], body.strip('\n'), values.stem)
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), edited_line=i + 1, values=None if values is None else str(values),
                                               candidate=str(out/'candidate.py'), candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8])
