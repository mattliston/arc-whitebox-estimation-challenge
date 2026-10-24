"""TAG opus_engineer a62 XSG v2 (2026-10-12 01:1xZ) = v1 with a SITE argument on every call and _XSG_ON a set of sites (or True / False),
to find which site breaks byte-identity (v1: predictions differed at -0.085%% of bill). v1 header: (2026-10-12 01:3xZ, lead: harvest small byte-identical bill cuts): ROW / COLUMN GATHERS BY SCATTER at the
largest gather sites of the ACTIVE code (probe_gathers_v1.py on S48b, network 7100136: fancy gathers are 0.220% of the bill, billed 4 per
output element in float32, while put_along_axis bills 1 per SOURCE element; the scatter-cheaper part of the row gathers is 0.069%, the
column gathers add about 0.03%). A gather of k unique rows (columns) out of m becomes: an inverse index (cached per roster object and size,
so a roster gathered at several sites builds it once), a zero buffer of k + 1 rows (columns), one put_along_axis of the whole source, and
the first k rows (columns): the same float32 values, so predictions are byte-identical (certified by the prediction sha). Taken only when
(4 k - m) x width > 2 m + 6 k (the rule of the file's own _unique_rows_via_scatter); otherwise the plain gather runs.
Edits by LINE NUMBER of S48b (sha 8378970c), each asserted against its expected text; _XSG_ON = False = S48b's arithmetic and bill.
usage: make_xsg_v1.py <out name> [<python literal for _XSG_ON, default True>]   -> a62_modules/s49/<out name>/candidate.py"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE/'s48/s48b/candidate.py'
name = sys.argv[1]
on = sys.argv[2] if len(sys.argv) > 2 else 'True'
out = HERE/'s49'/name
out.mkdir(parents=True)
text = SRC.read_text()
assert hashlib.sha256(text.encode()).hexdigest().startswith('8378970c')
lines = text.split('\n')
EDITS = {   # line number (1-based) -> (old line, new line)
    5154: ("    wp = _e2_H(w[previous, :]).T",
           "    wp = _e2_H(_xsg_rows(5154, w, previous)).T   # opus-engineer a62 XSG: gather by scatter"),
    5159: ("    common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])",
           "    common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,_xsg_rows(5159, base, previous))   # opus-engineer a62 XSG"),
    5161: ("    directions = _e2_H(w[selected, :]).T",
           "    directions = _e2_H(_xsg_rows(5161, w, selected)).T   # opus-engineer a62 XSG"),
    8412: ("    Qp = state['Q'][previous, :] if old_columns else None",
           "    Qp = _xsg_rows(8412, state['Q'], previous) if old_columns else None   # opus-engineer a62 XSG"),
    8480: ("    wx = w32 if rows is None else w32[:, rows]   # PT-15 deadcut: projection rows only; the join above keeps the full W",
           "    wx = w32 if rows is None else _xsg_cols(8480, w32, rows)   # PT-15 deadcut: projection rows only; the join above keeps the full W (opus-engineer a62 XSG)"),
    8481: ("    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm, _e2_H(_e2_H(wx[previous, :]).T), active_basis)",
           "    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm, _e2_H(_e2_H(_xsg_rows(8481, wx, previous)).T), active_basis)   # opus-engineer a62 XSG"),
    9566: ("        pushed = _YNT_PARENT_PUSH(w[:, rows], p, previous, newborn)",
           "        pushed = _YNT_PARENT_PUSH(_xsg_cols(9566, w, rows), p, previous, newborn)   # opus-engineer a62 XSG"),
    13378: ("        wa = hold(w[live])",
            "        wa = hold(_xsg_rows(13378, w, live))   # opus-engineer a62 XSG"),
    3469: ("            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_bd_base32[:, selected]), _e2_tsel, _e2_ssel)",
           "            newborn = (int(_e2_H(pp.shape[1])), selected, _e2_H(_xsg_cols(3469, _bd_base32, selected)), _e2_tsel, _e2_ssel)   # opus-engineer a62 XSG"),
    5119: ("    old_plane = _shared_old384_mm(inner, state['pre_basis'][live_indices, :].T)",
           "    old_plane = _shared_old384_mm(inner, _xsg_rows(5119, state['pre_basis'], live_indices).T)   # opus-engineer a62 XSG"),
}
for n, (old, new) in EDITS.items():
    assert lines[n - 1] == old, (n, lines[n - 1])
    lines[n - 1] = new
text = '\n'.join(lines).rstrip('\n') + '''


# BEGIN OPUS-ENGINEER a62 XSG v2 (gathers by scatter, per site; see O/engineer/a62_modules/make_xsg_v2.py)
_XSG_ON = %s
_XSG_INV = {}
_XSG_STATS = dict(rows=0, cols=0, plain=0)


def _xsg_inverse(idx, m):
    key = (id(idx), m)
    got = _XSG_INV.get(key)
    if got is not None and got[0] is idx:
        return got[1]
    k = int(idx.shape[0])
    inverse = fnp.full(m, k, dtype=fnp.int64)
    fnp.put(inverse, idx, fnp.arange(k, dtype=fnp.int64))
    if len(_XSG_INV) > 32:
        _XSG_INV.clear()
    _XSG_INV[key] = (idx, inverse)
    return inverse


def _xsg_rows(site, A, idx):
    m, width = int(A.shape[0]), int(A.shape[1])
    k = int(idx.shape[0])
    if not (_XSG_ON is True or (_XSG_ON and site in _XSG_ON)) or str(A.dtype) != 'float32' or len(idx.shape) != 1 or k > m or (4 * k - m) * width <= 2 * m + 6 * k:
        _XSG_STATS['plain'] += 1
        return A[idx]
    inverse = _xsg_inverse(idx, m)
    answer = fnp.zeros((k + 1, width), dtype=A.dtype)
    fnp.put_along_axis(answer, inverse[:, None], A, axis=0)
    _XSG_STATS['rows'] += 1
    return answer[:k]


def _xsg_cols(site, A, idx):
    n_rows, m = int(A.shape[0]), int(A.shape[1])
    k = int(idx.shape[0])
    if not (_XSG_ON is True or (_XSG_ON and site in _XSG_ON)) or str(A.dtype) != 'float32' or len(idx.shape) != 1 or k > m or (4 * k - m) * n_rows <= 2 * m + 6 * k:
        _XSG_STATS['plain'] += 1
        return A[:, idx]
    inverse = _xsg_inverse(idx, m)
    answer = fnp.zeros((n_rows, k + 1), dtype=A.dtype)
    fnp.put_along_axis(answer, inverse[None, :], A, axis=1)
    _XSG_STATS['cols'] += 1
    return answer[:, :k]
# END OPUS-ENGINEER a62 XSG v2
''' % on
(out/'candidate.py').write_text(text)
print(name, hashlib.sha256(text.encode()).hexdigest()[:8])
