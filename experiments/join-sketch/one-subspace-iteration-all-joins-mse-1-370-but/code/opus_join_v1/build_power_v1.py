"""Deployable join: ONE subspace (power) iteration on today's Nystrom sketch at selected joins (metered; existing fast-matmul route).
Q0 = qr(G[:, rows]); Y1 = G Q0 = J diag(w) (J^T Q0) + Qp core (Qp^T Q0); Qn, R1 = qr(Y1); core = (Q0^T Qn)^-1 R1^T (consistent
Nystrom core with sketch Q0, symmetrized as today). Downstream unchanged. Screen switch: env PJ_LAYERS (e.g. '7,10,13')."""
from pathlib import Path
import hashlib
D=Path(__file__).resolve().parent;R=D.parent
src=(R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('2b10843a')
old='''        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:   # opus-engineer a08: R K^-1 R' with K = Y[rows] = Qn[rows] R is Qn[rows]^-1 R' (one solve, no product)
            R64 = unused_R.astype(fnp.float32)'''
new='''        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _PJ_LAYERS and layer in _PJ_LAYERS:
            Qn, unused_R, core = _pj_power(_e2_H, Qn, J, weight, Qp if old_columns else None, state['core'] if old_columns else None, _need_core)
        elif _need_core:   # opus-engineer a08: R K^-1 R' with K = Y[rows] = Qn[rows] R is Qn[rows]^-1 R' (one solve, no product)
            R64 = unused_R.astype(fnp.float32)'''
i=src.rindex(old);src=src[:i]+new+src[i+len(old):]
frag='''

# BEGIN OPUS-JOIN2 one-step subspace iteration at the joins (screen switch PJ_LAYERS; metered, existing fast-matmul route)
import os as _pj_os
_PJ_LAYERS = tuple(int(x) for x in _pj_os.environ.get('PJ_LAYERS', '').split(',') if x)
_PJ_STATS = dict(calls=0)


def _pj_power(h, Q0, J, weight, Qp, core_old, need_core):
    Z = h(_shared_old384_mm(h(Q0.T), J))                                   # rank x cols = Q0^T J
    Y1 = _shared_old384_mm(J, h(h(weight[:, None]) * h(Z.T)))              # n x rank = J diag(w) J^T Q0
    if Qp is not None:
        Y1 = Y1 + h(_shared_old384_mm(Qp, h(_shared_old384_mm(core_old, h(_shared_old384_mm(h(Qp.T), Q0))))))
    Qn, R1 = fnp.linalg.qr(Y1, mode='reduced')
    core = None
    if need_core:
        M = _shared_old384_mm(h(Q0.T), Qn)                                 # rank x rank = Q0^T Qn  (Q0^T G Q0 = M R1)
        core = fnp.linalg.solve(M.astype(fnp.float32), R1.astype(fnp.float32).T).astype(fnp.float32)
        core = (core + core.T) * 0.5
    _PJ_STATS['calls'] += 1
    return Qn, R1, core
# END OPUS-JOIN2
'''
o=D/'candidate_2b108_join_power_v1.py'
with o.open('x') as f:f.write(src+frag)
compile(o.read_text(),str(o),'exec');print(hashlib.sha256(o.read_bytes()).hexdigest())
