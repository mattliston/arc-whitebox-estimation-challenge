"""TAG opus_engineer a62 XSG any-parent v1 (2026-10-12 01:4xZ, lead: XSG goes into the AGGRESSIVE file only, as a 2-minute append to the next
twin): make_xsg_v2.py's 9 certified sites (byte-identical on S48b: 7100136 / 137 in-process, 35 / 100 at the boundary) applied to ANY
parent by TEXT, not by line number: each site's expected line is replaced at its LAST occurrence in the parent (the active definition:
every file of this stack re-defines these functions in order; the earlier copies are dead code) and the replacement count is printed.
The births column gather (S48b line 3469) stays out (by scatter it changed predictions). The helper block is make_xsg_v2.py's, all sites
on; on S48b it equals s49/xsgF except xsgF's inert call at the excluded births site (asserted with --check-s48b).
usage: make_xsg_any_v1.py <parent.py> <out name> [--check-s48b]   -> a62_modules/s49/<out name>/candidate.py + build.json"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name = Path(sys.argv[1]).resolve(), sys.argv[2]
check = '--check-s48b' in sys.argv[3:]
out = HERE/'s49'/name
out.mkdir(parents=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
text = parent.read_text()
SITES = [   # (site tag, old line, new line): make_xsg_v2.py's EDITS for the 9 certified sites
    (5154, "    wp = _e2_H(w[previous, :]).T",
           "    wp = _e2_H(_xsg_rows(5154, w, previous)).T   # opus-engineer a62 XSG: gather by scatter"),
    (5159, "    common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])",
           "    common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,_xsg_rows(5159, base, previous))   # opus-engineer a62 XSG"),
    (5161, "    directions = _e2_H(w[selected, :]).T",
           "    directions = _e2_H(_xsg_rows(5161, w, selected)).T   # opus-engineer a62 XSG"),
    (8412, "    Qp = state['Q'][previous, :] if old_columns else None",
           "    Qp = _xsg_rows(8412, state['Q'], previous) if old_columns else None   # opus-engineer a62 XSG"),
    (8480, "    wx = w32 if rows is None else w32[:, rows]   # PT-15 deadcut: projection rows only; the join above keeps the full W",
           "    wx = w32 if rows is None else _xsg_cols(8480, w32, rows)   # PT-15 deadcut: projection rows only; the join above keeps the full W (opus-engineer a62 XSG)"),
    (8481, "    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm, _e2_H(_e2_H(wx[previous, :]).T), active_basis)",
           "    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm, _e2_H(_e2_H(_xsg_rows(8481, wx, previous)).T), active_basis)   # opus-engineer a62 XSG"),
    (9566, "        pushed = _YNT_PARENT_PUSH(w[:, rows], p, previous, newborn)",
           "        pushed = _YNT_PARENT_PUSH(_xsg_cols(9566, w, rows), p, previous, newborn)   # opus-engineer a62 XSG"),
    (13378, "        wa = hold(w[live])",
            "        wa = hold(_xsg_rows(13378, w, live))   # opus-engineer a62 XSG"),
    (5119, "    old_plane = _shared_old384_mm(inner, state['pre_basis'][live_indices, :].T)",
           "    old_plane = _shared_old384_mm(inner, _xsg_rows(5119, state['pre_basis'], live_indices).T)   # opus-engineer a62 XSG"),
]
lines = text.split('\n')
report = []
for tag, old, new in SITES:
    hits = [i for i, s in enumerate(lines) if s == old]
    assert hits, 'site %d: expected line not found: %r' % (tag, old)
    lines[hits[-1]] = new
    report.append(dict(site=tag, occurrences=len(hits), replaced_line=hits[-1] + 1))
    print('site %5d: %d occurrence(s), replaced the last at line %d' % (tag, len(hits), hits[-1] + 1))
BLOCK = '''

# BEGIN OPUS-ENGINEER a62 XSG v2 (gathers by scatter, per site; see O/engineer/a62_modules/make_xsg_v2.py)
_XSG_ON = {5154, 5159, 5161, 8412, 8481, 13378, 5119, 8480, 9566}
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
'''
result = '\n'.join(lines).rstrip('\n') + BLOCK
if check:   # s49/xsgF also routed the births site 3469 through the helper (off there, so the plain gather ran): the only other difference
    ref = (HERE/'s49/xsgF/candidate.py').read_text().split('\n')
    mine = result.split('\n')
    diff = [i for i in range(min(len(ref), len(mine))) if ref[i] != mine[i]]
    assert len(ref) - len(mine) in (0, 1) and len(diff) == 1 and '_xsg_cols(3469,' in ref[diff[0]], (len(ref), len(mine), diff[:3])
    print('CHECK: = s49/xsgF except its inert site-3469 call (line %d) and one trailing blank line' % (diff[0] + 1))
(out/'candidate.py').write_text(result)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), sites=report,
                                               candidate=str(out/'candidate.py'), candidate_sha256=sha(result.encode())), indent=1) + '\n')
print(name, sha(result.encode())[:8])
