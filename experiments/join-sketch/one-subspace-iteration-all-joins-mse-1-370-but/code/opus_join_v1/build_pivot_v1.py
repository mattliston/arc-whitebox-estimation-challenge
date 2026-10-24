"""One-shot pivot refinement at selected joins (screen switches PR_LAYERS, PR_K; metered). Static Nystrom as today, then
d_res = diag(G) - diag(Qn core0 Qn^T); swap the k sketch rows of smallest diag(G) for the k non-sketch rows of largest d_res;
Y' = [Y[:, kept], G[:, new]] (one extra n x k block via the factored products); QR(Y') and the Nystrom core as today, rank unchanged."""
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
        if _PR_LAYERS and layer in _PR_LAYERS:
            Qn, unused_R, core, _jn_rows = _pr_refine(_e2_H, Y, Qn, unused_R, _jn_rows, _pr_d, J, weight, Qp if old_columns else None, state['core'] if old_columns else None, _need_core)
        elif _need_core:   # opus-engineer a08: R K^-1 R' with K = Y[rows] = Qn[rows] R is Qn[rows]^-1 R' (one solve, no product)
            R64 = unused_R.astype(fnp.float32)'''
i=src.rindex(old);src=src[:i]+new+src[i+len(old):]
od='''            del _jn_d
        else:'''
j=src.rindex(od,0,i);src=src[:j]+'''            _pr_d = _jn_d
            del _jn_d
        else:'''+src[j+len(od):]
frag='''

# BEGIN OPUS-JOIN3 one-shot pivot refinement (screen switches PR_LAYERS, PR_K; metered)
import os as _pr_os
_PR_LAYERS = tuple(int(x) for x in _pr_os.environ.get('PR_LAYERS', '').split(',') if x)
_PR_K = int(_pr_os.environ.get('PR_K', '32'))
_PR_STATS = dict(calls=0)


def _pr_refine(h, Y, Qn, R, rows, d, J, weight, Qp, core_old, need_core):
    k = _PR_K
    core0 = fnp.linalg.solve(Qn[rows, :].astype(fnp.float32), R.astype(fnp.float32).T).astype(fnp.float32)
    core0 = (core0 + core0.T) * 0.5
    dn = fnp.sum(h(h(_shared_old384_mm(Qn, core0)) * Qn), axis=1)           # diag of the Nystrom approximation
    dres = (d - dn).astype(fnp.float64)
    fnp.put(dres, rows, -1.0e300)                                            # sketch rows are never candidates
    new_rows = fnp.argsort(-dres, kind='stable')[:k]
    order = fnp.argsort(d[rows].astype(fnp.float64), kind='stable')
    keep_pos = fnp.sort(order[k:])
    rows2 = fnp.concatenate([rows[keep_pos], new_rows])
    Ynew = _shared_old384_mm(J, h(h(weight[:, None]) * h(h(J[new_rows, :]).T)))
    if Qp is not None:
        Ynew = Ynew + h(_shared_old384_mm(Qp, h(_shared_old384_mm(core_old, h(h(Qp[new_rows, :]).T)))))
    Y2 = fnp.concatenate([h(Y[:, keep_pos]), Ynew], axis=1)
    Qn2, R2 = fnp.linalg.qr(Y2, mode='reduced')
    core = None
    if need_core:
        core = fnp.linalg.solve(Qn2[rows2, :].astype(fnp.float32), R2.astype(fnp.float32).T).astype(fnp.float32)
        core = (core + core.T) * 0.5
    _PR_STATS['calls'] += 1
    return Qn2, R2, core, rows2
# END OPUS-JOIN3
'''
o=D/'candidate_2b108_join_pivot_v1.py'
with o.open('x') as f:f.write(src+frag)
compile(o.read_text(),str(o),'exec');print(hashlib.sha256(o.read_bytes()).hexdigest())
