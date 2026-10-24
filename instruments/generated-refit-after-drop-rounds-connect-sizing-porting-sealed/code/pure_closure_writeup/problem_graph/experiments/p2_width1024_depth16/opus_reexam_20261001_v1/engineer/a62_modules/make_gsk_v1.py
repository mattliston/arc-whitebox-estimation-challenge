"""TAG opus_engineer a62 (file written 2026-10-12T17:50Z; lead 17:5xZ: scout-card's gk arm, transport gate with the skew + kurtosis correction): the
native gk as a post-build EDIT on a port file (safe F4c v2 or twin F4c v3, source or final): the six edits of stack/build.py's GSK lever,
copied verbatim (the transport gate gq = _gq(g, d, aux, bpre) in place of g in the carried b, the inherited plane and its diagonal, the gated
young columns, the old tier's gated basis, and F4c's step), plus engineer/a41_base/fragment_gsk_v1.py appended with its literals at 0 (= the
parent bit for bit: _gq returns the card's gate object itself), then an optional values fragment (e.g. _GSK = 1.0, _GK4 = 1.0).
Every edit must match exactly once, else nothing is written.
usage: make_gsk_v1.py <parent.py> <out name> [<values fragment>]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name = Path(sys.argv[1]).resolve(), sys.argv[2]
values = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
sha = lambda b: hashlib.sha256(b).hexdigest()
text = parent.read_text()
b_line = '            b = _e2_H(g * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV) if _FULL_LOWER_MOMENTS else _e2_H(_e2_H(g * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV)) * alive'
edits = [
    ("            d = _e2_pair['d'] if _e2_pair is not None and _e2_H(_e2_pair['aux']) is aux else aux['pdf'] / aux['sigma']\n",
     "            d = _e2_pair['d'] if _e2_pair is not None and _e2_H(_e2_pair['aux']) is aux else aux['pdf'] / aux['sigma']\n"
     "            gq = _gq(g, d, aux, bpre)   # opus-engineer gsk: the gate that carries the third-order content, with the mean gate's first-order corrections\n", 1),
    (b_line + "\n", b_line.replace('g * bpre', 'gq * bpre') + "\n", 1),
    ("            _ps_g2 = g ** 2\n"
     "            _ps_inherited_diag = fnp.einsum('i,i,i->i', _ps_g2, g, _plane_cpdiag64, optimize=False)\n"
     "            inherited = fnp.einsum('i,j,ij->ij', _ps32(_ps_g2), _ps32(g), _ps_cp32, optimize=False)\n",
     "            _ps_g2 = gq ** 2\n"
     "            _ps_inherited_diag = fnp.einsum('i,i,i->i', _ps_g2, gq, _plane_cpdiag64, optimize=False)\n"
     "            inherited = fnp.einsum('i,j,ij->ij', _ps32(_ps_g2), _ps32(gq), _ps_cp32, optimize=False)\n", 1),
    ("            _e2_ga = _e2_H(g.astype(fnp.float32)) * alive\n", "            _e2_ga = _e2_H(gq.astype(fnp.float32)) * alive\n", 1),
    ("            _E2_LAYER.update(ga_gate=g, ga_alive=alive, ga=_e2_ga)\n            _shared_old384_commit(l, g, alive, selected, shared_old)\n",
     "            _E2_LAYER.update(ga_gate=gq, ga_alive=alive, ga=_e2_ga)\n            _shared_old384_commit(l, gq, alive, selected, shared_old)\n", 1),
]
if True:   # the exact-term mixture's anchor is made of the gated columns: it takes the gate those columns were gated with
    edits.append(("_f4c_step(l, right if len(lam) else None, g, d, live_indices", "_f4c_step(l, right if len(lam) else None, gq, d, live_indices", 1))
for old, new, n in edits:
    assert text.count(old) == n, ('edit not found exactly %d time(s)' % n, old[:120], text.count(old))
    text = text.replace(old, new)
text = text.rstrip('\n') + '\n\n' + (HERE.parent/'a41_base/fragment_gsk_v1.py').read_text().strip('\n') + '\n'
if values is not None:
    body = values.read_text()
    text += '\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (values.stem, sha(body.encode())[:12], body.strip('\n'), values.stem)
out = HERE/'s49'/name
out.mkdir(parents=True)
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), edits=len(edits), values=None if values is None else str(values),
                                               candidate=str(out/'candidate.py'), candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8], '%d edits' % len(edits))
