"""TAG opus_engineer a62 (2026-10-12 00:0xZ, lead: re-measure natively the levers closed on torch error vs native bill; arm 3): RAYLEIGH-RITZ
at the L10 join on S48b, as priced on 10-10 (TEAM_LOG 10-10 19:31Z; never built then): the join's Nystrom sketch takes _RR_SKETCH[layer]
rows (384) instead of the join's rank (320); after the QR and the Nystrom core (R K^-1 R', the projected Gram the join already makes, here
384 x 384), eigh of the core in float64, the top-rank eigenvectors V rotate the frame (Qn <- Qn V, n x 320) and the core becomes their
eigenvalues; everything after (Fj = Qn' J, the overlap with the old frame, every later layer) runs at rank 320 as before.
Text edits in the ACTIVE old-tier push only (the definition the OTT wrapper wraps). _RR_SKETCH = {} = S48b bit for bit.
usage: make_rr_v1.py <out name> [<python literal for _RR_SKETCH, default {10: 384}>]   -> a62_modules/s49/<out name>/candidate.py"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE/'s48/s48b/candidate.py'
name = sys.argv[1]
sketch = sys.argv[2] if len(sys.argv) > 2 else '{10: 384}'
assert isinstance(eval(sketch), dict)
out = HERE/'s49'/name
out.mkdir(parents=True)
text = SRC.read_text()
assert hashlib.sha256(text.encode()).hexdigest().startswith('8378970c')
anchor = "# Approximate range-finder core, original orthogonal factor projection.\ndef _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):\n"
assert text.count(anchor) == 1
at = text.index(anchor)
end = text.index('\ndef ', at + len(anchor))
body = text[at:end]
EDITS = [
    ("        weight = coefficient * coefficient\n",
     "        weight = coefficient * coefficient\n"
     "        _rr_s = max(rank, int(_RR_SKETCH.get(layer, rank)))   # opus-engineer a62 RR: the sketch's row count (> rank: Rayleigh-Ritz below)\n"),
    ("            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])\n",
     "            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:_rr_s])   # opus-engineer a62 RR: _rr_s rows\n"),
    ("        del K, unused_R, Y, Omega, _jn_rows\n",
     "        if _rr_s > rank:   # opus-engineer a62 RR: Rayleigh-Ritz truncation of the larger sketch to the join's rank\n"
     "            if core is None:\n"
     "                raise RuntimeError('Rayleigh-Ritz needs the Nystrom core')\n"
     "            _rr_w, _rr_v = fnp.linalg.eigh(core.astype(fnp.float64))\n"
     "            _rr_order = fnp.argsort(-_rr_w, kind='stable')[:rank]\n"
     "            _rr_top = _rr_v[:, _rr_order].astype(fnp.float32)\n"
     "            Qn = _shared_old384_mm(Qn, _rr_top)\n"
     "            core = fnp.diag(_rr_w[_rr_order]).astype(fnp.float32)\n"
     "            _RR_STATS['joins'] += 1\n"
     "            del _rr_w, _rr_v, _rr_order, _rr_top\n"
     "        del K, unused_R, Y, Omega, _jn_rows\n"),
]
for old, new in EDITS:
    assert body.count(old) == 1, old
    body = body.replace(old, new)
text = text[:at] + body + text[end:]
text = text.rstrip('\n') + ('\n\n# opus-engineer a62 make_rr_v1: Rayleigh-Ritz at the joins named here (join layer -> sketch rows); {} = S48b bit for bit\n'
                            '_RR_SKETCH = %s\n_RR_STATS = dict(joins=0)\n' % sketch)
(out/'candidate.py').write_text(text)
print(name, hashlib.sha256(text.encode()).hexdigest()[:8])
