"""[opus-u64idea] build NT-join arms on S2tr 46e9315d (constants only). In the live _astra_bcc_push_impl join: the static Nystrom sketch
takes rank + p rows (energy order) and the QR frame is truncated back to rank by the eigenvectors of the (rank+p) Nystrom core
(full: eigh of the whole core; block q: keep the leading rank-q frame columns, eigh of the last q+p coordinates' first-order Frobenius
criterion). Downstream (Fj, overlap, factors, schedule, rank) unchanged. usage: build_nt_v2.py TAG P_BY_JOIN Q   e.g. ntf64 7:64,10:64,13:64 0"""
import hashlib, sys
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
tag, pspec, q = sys.argv[1], sys.argv[2], int(sys.argv[3])
P = {int(a): int(b) for a, b in (x.split(':') for x in pspec.split(','))}
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
fn = src.index('def _astra_bcc_push_impl(')


def patch(old, new):
    global src
    i = src.rindex(old); assert i > fn and src.count(old) >= 1, old[:60]
    assert src.index('\ndef ', fn + 10) > i, 'anchor outside _astra_bcc_push_impl'
    src = src[:i] + new + src[i + len(old):]


patch("""        Omega = w32[previous, :rank]
        if _JOIN_NYSTROM:""", """        Omega = w32[previous, :rank]
        _nt_p = int(_NT_P.get(layer, 0)) if _JOIN_NYSTROM else 0   # opus-u64idea NT: extra static rows at this join
        if _JOIN_NYSTROM:""")
patch("""            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])
            Y = _shared_old384_mm(J,""", """            _jn_rows = (fnp.argsort(-_jn_d.astype(fnp.float64), kind='stable')[:rank + _nt_p] if _nt_p else
                        fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]))
            Y = _shared_old384_mm(J,""")
patch("""        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:""", """        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _nt_p:
            Qn, core = _nt_truncate(layer, rank, Qn, unused_R, _jn_rows, _need_core)
        elif _need_core:""")
frag = '''

# BEGIN OPUS-U64IDEA NT JOIN (oversampled static Nystrom sketch + eigen truncation back to the join rank; build_nt_v2.py)
_NT_P = %r   # join layer -> extra static rows p (absent = today's static rank-r Nystrom)
_NT_Q = %d   # 0 = full eigen truncation of the (r+p) core; q > 0 = block: keep the leading r-q frame, truncate the last q+p
import sys as _nt_sys


def _nt_truncate(layer, rank, Q, R, rows, need_core):
    k = int(Q.shape[1])
    C = fnp.linalg.solve(Q[rows, :].astype(fnp.float32), R.astype(fnp.float32).T).astype(fnp.float32)   # Nystrom core in frame Q
    C = (C + C.T) * 0.5
    if not _NT_Q:
        ev, V = fnp.linalg.eigh(C)
        V = (V[:, k - rank:][:, ::-1]).copy()           # top-rank eigenvectors, descending
        Qn = _shared_old384_mm(Q, V)
        core = fnp.diag((ev[k - rank:][::-1]).copy()).astype(fnp.float32) if need_core else None
    else:
        h = rank - _NT_Q
        CT = (C[:, h:]).copy()                            # k x (q+p): columns of the tail
        CTT = (CT[h:, :]).copy(); CHT = (CT[:h, :]).copy()
        M = fnp.matmul(CTT, CTT) + 2.0 * fnp.matmul(CHT.T, CHT)
        M = (M + M.T) * 0.5
        ev, V = fnp.linalg.eigh(M)
        X = (V[:, (k - h) - _NT_Q:][:, ::-1]).copy()     # top-q of the tail criterion
        Qn = fnp.concatenate([Q[:, :h], fnp.matmul(Q[:, h:], X)], axis=1)
        core = None
        if need_core:
            CX = fnp.matmul(CT, X)                                      # k x q
            HX = (CX[:h, :]).copy()
            core = fnp.concatenate([fnp.concatenate([C[:h, :h], HX], axis=1),
                                    fnp.concatenate([HX.T, fnp.matmul(X.T, CX[h:, :])], axis=1)], axis=0).astype(fnp.float32)
            core = (core + core.T) * 0.5
    print('[nt] layer %%d rows %%d -> rank %%d q %%d core %%s' %% (layer, k, int(Qn.shape[1]), _NT_Q, need_core), file=_nt_sys.stderr, flush=True)
    return Qn.astype(fnp.float32), core
# END OPUS-U64IDEA NT JOIN
''' % (P, q)
o = D/f'cand_s2tr_{tag}_v2.py'
with o.open('x') as f:
    f.write(src + frag)
compile(o.read_text(), str(o), 'exec'); print(o.name, hashlib.sha256(o.read_bytes()).hexdigest())
