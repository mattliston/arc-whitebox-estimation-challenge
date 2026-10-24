"""ORACLE (not deployable): at selected joins replace the Nystrom (Qn, core) by the exact top-rank eigenpairs of the join Gram
G = J diag(w) J^T + Qp core Qp^T (unmetered float64 numpy); everything downstream unchanged. Logs relative Frobenius errors."""
from pathlib import Path
import hashlib
D=Path(__file__).resolve().parent;R=D.parent
src=(R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('2b10843a')
anchor='''            core = (core + core.T) * 0.5
            del R64
        del K, unused_R, Y, Omega, _jn_rows'''
assert src.count(anchor)>=1
inject='''            core = (core + core.T) * 0.5
            del R64
        if _OJ_LAYERS is not None and layer in _OJ_LAYERS:
            Qn, core = _oj_oracle(layer, rank, J, weight, Qp if old_columns else None, state.get('core') if old_columns else None, Qn, unused_R, _jn_rows, core, _need_core)
        del K, unused_R, Y, Omega, _jn_rows'''
i=src.rindex(anchor);src=src[:i]+inject+src[i+len(anchor):]
frag='''

# BEGIN OPUS-JOIN ORACLE (NOT DEPLOYABLE; unmetered numpy eigen-decomposition at the joins)
import os as _oj_os, json as _oj_json
_OJ_LAYERS = tuple(int(x) for x in _oj_os.environ.get('OJ_LAYERS', '').split(',') if x) or None
_OJ_NAME = ['']


def _oj_oracle(layer, rank, J, weight, Qp, core_old, Qn, R, rows, core_nys, need_core):
    import numpy as _np
    Jd = _np.array(J, copy=True).astype(_np.float64); w = _np.array(weight, copy=True).astype(_np.float64)
    G = (Jd * w[None, :]) @ Jd.T
    if Qp is not None:
        Qd = _np.array(Qp, copy=True).astype(_np.float64); Cd = _np.array(core_old, copy=True).astype(_np.float64)
        G = G + Qd @ Cd @ Qd.T
    G = 0.5 * (G + G.T)
    ev, V = _np.linalg.eigh(G)
    order = _np.argsort(ev)[::-1][:rank]
    Vr, lr = V[:, order], ev[order]
    Qn_d = _np.array(Qn, copy=True).astype(_np.float64)
    if core_nys is not None:
        Cn = _np.array(core_nys, copy=True).astype(_np.float64)
    else:
        Rd = _np.array(R, copy=True).astype(_np.float64); ri = _np.array(rows, copy=True)
        Cn = _np.linalg.solve(Qn_d[ri, :], Rd.T); Cn = 0.5 * (Cn + Cn.T)
    nG = _np.linalg.norm(G)
    rec = dict(name=_OJ_NAME[0], layer=int(layer), rank=int(rank), n=int(G.shape[0]), cols=int(Jd.shape[1]),
               err_nystrom=float(_np.linalg.norm(G - Qn_d @ Cn @ Qn_d.T) / nG), err_eigen=float(_np.linalg.norm(G - (Vr * lr) @ Vr.T) / nG),
               tail_energy=float(_np.sqrt(max(0.0, (ev[_np.argsort(ev)[::-1][rank:]] ** 2).sum())) / nG))
    path = _oj_os.environ.get('OJ_STATS')
    if path:
        with open(path, 'a') as f:
            f.write(_oj_json.dumps(rec) + '\\n')
    Qn2 = fnp.asarray(_np.ascontiguousarray(Vr.astype(_np.float32)))
    core2 = fnp.asarray(_np.diag(lr).astype(_np.float32)) if need_core else None
    return Qn2, core2


_OJ_PARENT_PREDICT = Estimator.predict


def _oj_predict(self, mlp, *args, **kwargs):
    _OJ_NAME[0] = str(getattr(mlp, 'name', ''))
    return _OJ_PARENT_PREDICT(self, mlp, *args, **kwargs)
Estimator.predict = _oj_predict
# END OPUS-JOIN ORACLE
'''
o=D/'candidate_2b108_join_oracle_v1.py'
with o.open('x') as f:f.write(src+frag)
compile(o.read_text(),str(o),'exec');print(hashlib.sha256(o.read_bytes()).hexdigest())
