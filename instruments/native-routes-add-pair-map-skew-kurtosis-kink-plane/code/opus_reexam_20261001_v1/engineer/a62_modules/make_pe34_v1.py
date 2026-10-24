"""TAG opus_engineer2 a62 (file written 2026-10-13T00:06:41Z; trainer 23:50Z NEW RAD FAMILIES FOR S54: pe3 / pe4 per band): the pair map's local
skew / kurtosis terms BY LAYER. Torch (trainer/otr_m_v6.py, the series wrapper): at layer l the Hermite pair series gets t3 x (1 + eps['pe3'][l])
and t4 x (1 + eps['pe4'][l]); e3 is linear in t3 and e4 in t4, and neither appears anywhere else in the series (T31 is its own input), so e3 and
e4 scale exactly. Native (_nonlinear_pair_batch): st['t3'] = k3 / s^3 and st['t4'] (a copy of the local t4) feed ONLY the FS / GS rows 1-2
([t3 F, t4 F], [t3 G, t4 G], one multiply each), which only the e3 / e4 products T6..T9 read, in the plain series and in the packed 1024 path
(_psp_views / _psp_pair_series read the same FS / GS buffers); the local t4 that builds T31 is not touched. Edited by TEXT: right after the
copy into st['t4'], at the layers listed in _PE34_BY_LAYER ({layer l: (1 + pe3[l], 1 + pe4[l])}; l = _PHYS_LAYER[0], which the live loop sets
just before the pair map = torch's call_layer), st['t3'] and / or st['t4'] are multiplied in place by their factor (a factor of exactly 1.0
is skipped). L0 (the first map is Gaussian: t3 = t4 = 0) and L15 (the final layer runs the scalar map only) have no site, natively as in torch.
A block _PE34_BY_LAYER = {} (= the parent bit for bit: one dict lookup per layer, no array operation); an optional values fragment sets layers.
usage: make_pe34_v1.py <parent.py> <out name> [<fragment_pe34v_<name>.py>]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
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
d = [i for i, s in enumerate(lines) if s.startswith('def _nonlinear_pair_batch(')]
assert len(d) == 1, 'def _nonlinear_pair_batch: %d matches' % len(d)
end = next(i for i in range(d[0] + 1, len(lines)) if lines[i] and not lines[i][0].isspace())
binds = sorted([(i, s.replace(' ', '')) for i, s in enumerate(lines) if re.match(r'nonlinear\s*=', s)] +
               [(i, s) for i, s in enumerate(lines) if s.startswith('def nonlinear(')])
assert binds and binds[-1][1] == 'nonlinear=_nonlinear_pair_batch' and binds[-1][0] > d[0], \
    'the module name nonlinear is not finally _nonlinear_pair_batch: %r' % binds[-3:]
assert '_PHYS_LAYER = [None]' in lines and '            _PHYS_LAYER[0] = l' in lines, 'the live loop does not publish the layer in _PHYS_LAYER'
anchor = "    fnp.copyto(st['t4'], t4)"
hits = [i for i in range(d[0], end) if lines[i] == anchor]
assert len(hits) == 1, 'the copy into st[t4]: %d matches' % len(hits)
a = hits[0]
assert lines[a - 1] == "    t4 = k4 / _e2_H(v ** 2)" and lines[a - 2] == "    _e2_hold.append(fnp.divide(k3, s ** 3, out=st['t3']))", 't3 / t4 not formed as expected'
assert lines[a + 1].startswith("    _e2_hold.append(fnp.multiply(FT, st['T34c'], out=st['FS12'])); _e2_hold.append(fnp.multiply(GT, st['T34c'], out=st['GS12']))"), \
    'the FS / GS rows are not formed right after the copy'
# st['t3'] / st['t4'] are read nowhere else (only their state line and the two writes above), and T34 only through T34c on that line
counts = [text0.count(k) for k in ("st['t3']", "st['t4']", "st['T34']", "st['T34c']", 'T34=fnp.empty', "'T34'")]
assert counts == [2, 2, 2, 3, 1, 2], counts
block = ["    _pe34 = _PE34_BY_LAYER.get(_PHYS_LAYER[0])   # opus-engineer2 a62 pe3 / pe4 by layer (make_pe34_v1.py): torch's e3 x (1 + pe3[l]), e4 x (1 + pe4[l])",
         "    if _pe34 is not None:",
         "        if _pe34[0] != 1.0:",
         "            _e2_hold.append(fnp.multiply(st['t3'], _pe34[0], out=st['t3']))",
         "        if _pe34[1] != 1.0:",
         "            _e2_hold.append(fnp.multiply(st['t4'], _pe34[1], out=st['t4']))"]
lines[a + 1:a + 1] = block
print('pair map: %d lines after the copy into st[t4] (line %d of _nonlinear_pair_batch, file line %d)' % (len(block), a - d[0] + 1, a + 1))
text = '\n'.join(lines).rstrip('\n') + '\n\n# BEGIN OPUS-ENGINEER2 a62 PE3 / PE4 BY LAYER v1 (make_pe34_v1.py)\n_PE34_BY_LAYER = {}\n# END OPUS-ENGINEER2 a62 PE3 / PE4 BY LAYER v1\n'
if values is not None:
    body = values.read_text()
    text += '\n# BEGIN OPUS-ENGINEER A62 FRAGMENT %s (sha256 %s)\n%s\n# END OPUS-ENGINEER A62 FRAGMENT %s\n' % (values.stem, sha(body.encode())[:12], body.strip('\n'), values.stem)
out.mkdir(parents=True)
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), anchor_line=a + 1,
                                               values=None if values is None else str(values), candidate=str(out/'candidate.py'),
                                               candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8])
