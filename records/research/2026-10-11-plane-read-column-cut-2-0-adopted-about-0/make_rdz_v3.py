"""TAG opus_engineer a62 RDZ gate v3 (2026-10-11 23:2xZ, lead: ONE-SIDED read row cut): make_rdz_v1.py's two loop edits on S48b (8378970c)
with the cut rows chosen by ONE tail of the response roster: _RDZ_SIDE '+' = the always-on rows (gate > Phi(_RDZ_Z), a > +z), '-' = the
dead-side rows (gate < Phi(-_RDZ_Z), a < -z). ERROR TEST ONLY (never a release file): the full read still runs, then its plane is zeroed on
those rows (every column; columns and the push untouched), and the cut rows count as omitted rows for the radial source, as in v1.
_RDZ_STATS also counts, per layer l, the response rows (rt<l>) and those beyond +z (rp<l>) and below -z (rm<l>), whatever the side.
_RDZ_SIDE = None: S48b's arithmetic exactly.
usage: make_rdz_v3.py <out name> <side: + | - | none> [<z, default 2.0>]   -> a62_modules/s49/<out name>/candidate.py (refuses to overwrite)"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE/'s48/s48b/candidate.py'
name, side = sys.argv[1], sys.argv[2]
z = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
assert side in ('+', '-', 'none')
out = HERE/'s49'/name
out.mkdir(parents=True)
text = SRC.read_text()
assert hashlib.sha256(text.encode()).hexdigest().startswith('8378970c')
EDITS = [
    ("            omitted = fnp.greater(alive, response_mask)   # E2: == alive & ~response_mask for booleans\n",
     "            omitted = fnp.greater(alive, _rdz_rows_mask(l, response_indices, response_gate, response_mask))   # opus-engineer a62 RDZ v3: the read's cut rows count as omitted\n"),
    ("                _np_forms(l, cp_pre, _plane_cov32)   # opus-engineer NP: the forms the fold must not contain, back on the read's block\n",
     "                _np_forms(l, cp_pre, _plane_cov32)   # opus-engineer NP: the forms the fold must not contain, back on the read's block\n"
     "                _rdz_zero(cp_pre, response_indices, response_gate)   # opus-engineer a62 RDZ v3: the plane on the read's cut rows = 0\n"),
]
for old, new in EDITS:
    assert text.count(old) == 1, old
    text = text.replace(old, new)
text += '''

# BEGIN OPUS-ENGINEER a62 RDZ GATE v3 (error test only; never a release file): see O/engineer/a62_modules/make_rdz_v3.py.
import math as _rdz_math
_RDZ_SIDE = %r
_RDZ_Z = %r
_RDZ_STATS = dict(layers=0, rows_cut=0)


def _rdz_cut(response_indices, gate):
    """The response rows of the chosen tail (positions in response_indices) and the counts of both tails."""
    g = gate[response_indices]
    hi = 0.5 * _rdz_math.erfc(-_RDZ_Z / _rdz_math.sqrt(2.0))       # Phi(+z)
    lo = 0.5 * _rdz_math.erfc(_RDZ_Z / _rdz_math.sqrt(2.0))        # Phi(-z)
    plus, minus = g > hi, g < lo
    where = fnp.nonzero(plus if _RDZ_SIDE == '+' else minus)[0]       # positions in response_indices
    return where, int(fnp.nonzero(plus)[0].shape[0]), int(fnp.nonzero(minus)[0].shape[0])


def _rdz_rows_mask(l, response_indices, gate, response_mask):
    cut, n_plus, n_minus = _rdz_cut(response_indices, gate)
    R = int(response_indices.shape[0])
    for key, value in (('rt%%d' %% l, R), ('rp%%d' %% l, n_plus), ('rm%%d' %% l, n_minus)):
        _RDZ_STATS[key] = _RDZ_STATS.get(key, 0) + value
    if _RDZ_SIDE is None or not int(cut.shape[0]):
        return response_mask
    mask = response_mask.copy()
    fnp.put(mask, response_indices[cut], False)
    return mask


def _rdz_zero(plane, response_indices, gate):
    if _RDZ_SIDE is None:
        return
    cut = _rdz_cut(response_indices, gate)[0]
    k = int(cut.shape[0])
    if not k:
        return
    keep_r = fnp.ones(int(plane.shape[0]), dtype=fnp.float32)
    fnp.put(keep_r, response_indices[cut], 0.0)
    fnp.multiply(plane, keep_r[:, None], out=plane)
    _RDZ_STATS['rows_cut'] += k
    _RDZ_STATS['layers'] += 1
# END OPUS-ENGINEER a62 RDZ GATE v3
''' % (None if side == 'none' else side, z)
(out/'candidate.py').write_text(text)
print(name, hashlib.sha256(text.encode()).hexdigest()[:8])
