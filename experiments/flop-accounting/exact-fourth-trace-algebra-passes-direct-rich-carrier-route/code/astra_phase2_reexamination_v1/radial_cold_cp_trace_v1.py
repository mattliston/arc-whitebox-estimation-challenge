"""Approximate only discarded cold CP rows; keep active factor carry exact."""
import numpy as np
import radial_compact_feed_fullpair_v1 as base


def trace_vector(p, lam):
    x = p.astype(np.float64)
    return x @ (lam.astype(np.float64)*np.sum(x*x, axis=0))


def trace_fit(t, C):
    trace = float(np.trace(C))
    assert trace > 0
    step = 1/(trace+2*float(np.linalg.norm(C)))
    b = t/trace
    for _ in range(24):
        b = b+step*(t-trace*b-2*(C@b))
    return b


def cold_read(p, lam, C, w, Bpre):
    if not len(lam):
        return np.zeros(len(w))
    b = trace_fit(trace_vector(p,lam), C)
    return 3*(w.T@b)*np.diag(Bpre)


source = base.PREDICT_SOURCE
edits = {
    '        pp = w32.T @ p': '''        cold_diagonal = cold_read(p,lam,metric_cov,w,Bpre)
        pp = np.zeros((n,p.shape[1]),np.float32)
        pp[live_indices] = w32[:,live_indices].T@p''',
    '            np.fill_diagonal(cp_pre, diagonal)': '''            diagonal = np.where(alive,diagonal,cold_diagonal)
            np.fill_diagonal(cp_pre, diagonal)''',
    'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))':
        'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1], previous_active=int(previous_alive.sum()), pushed_columns=len(lam)-2*len(selected)))',
}
for old,new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old,new)
env = dict(base.env, cold_read=cold_read)
exec(source,env)
predict = env['predict']
