"""Independent dense central-moment and affine-query certificates."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
began = time.monotonic()
code = (HERE / 'helper.py').read_text()
ns = dict(fnp=fnp)
exec(compile(code, str(HERE / 'helper.py'), 'exec'), ns)
apply = ns['_cvr_apply']
rng = np.random.Generator(np.random.PCG64DXSM(920260921))
rows = []


def full_factors(p, newborn):
    inherited, selected, base, t, s = newborn
    assert inherited == p.shape[1]
    plus, minus = base.copy(), base.copy()
    for col, ix in enumerate(selected):
        plus[ix, col] += t[col]
        minus[ix, col] += s[col]
    return np.concatenate([p, plus, minus], axis=1).astype(np.float64)


def tensor(C, p, A, b, newborn, lam, alive):
    n = C.shape[0]
    P = full_factors(p, newborn)
    T = np.einsum('ir,jr,kr,r->ijk', P, P, P, lam)
    B = C * alive[:, None] * alive[None, :]
    S = A * alive[:, None] * alive[None, :]
    for k in range(n):
        if not alive[k]:
            S[k, k] = A[k, k]
        e = np.eye(n)[k]
        v = S[:, k]
        T += (np.einsum('i,j,k->ijk', v, e, e)
              + np.einsum('i,j,k->ijk', e, v, e)
              + np.einsum('i,j,k->ijk', e, e, v)) / 3
    T += (np.einsum('i,jk->ijk', b, B)
          + np.einsum('j,ik->ijk', b, B)
          + np.einsum('k,ij->ijk', b, B))
    radial = .17
    Q = radial / 3 * (np.einsum('ij,kl->ijkl', C, C)
                    + np.einsum('ik,jl->ijkl', C, C)
                    + np.einsum('il,jk->ijkl', C, C))
    return T, Q


def error(a, b):
    return float(np.linalg.norm(a-b) / max(1., np.linalg.norm(b)))


for mode in ('empty', 'plain', 'lazy'):
    n, rank, oldcols, young = 7, 3, (0 if mode == 'empty' else 5), 4
    alive = np.array([True, True, False, True, False, True, True])
    M = rng.normal(size=(n, n))
    C = M @ M.T + np.eye(n)
    # Exact inactive zero variance must not be divided by zero.
    C[2, :] = 0; C[:, 2] = 0
    D = np.array([.3, 1.7, 1., .9, 1., 2.2, .65])
    host = np.diag(C) * D**2
    q = rng.normal(size=(n, rank)).astype(np.float32)
    if mode == 'lazy':
        fmap = rng.normal(size=(rank, rank)).astype(np.float32)
        fbase = rng.normal(size=(rank, 3)).astype(np.float32)
        fjoin = rng.normal(size=(rank, 2)).astype(np.float32)
        F = np.concatenate([fmap @ fbase, fjoin], axis=1)
    else:
        F = rng.normal(size=(rank, oldcols)).astype(np.float32)
    p = np.concatenate([q @ F, rng.normal(size=(n, young)).astype(np.float32)], axis=1)
    A = rng.normal(size=(n, n)).astype(np.float32)
    b = rng.normal(size=n)
    selected = np.array([6, 1, 3], dtype=np.int64)
    base = rng.normal(size=(n, len(selected))).astype(np.float32)
    t, s = rng.normal(size=(2, len(selected))).astype(np.float32)
    newborn = (p.shape[1], selected, base, t, s)
    lam = rng.normal(size=p.shape[1]+2*len(selected))
    beforeT, beforeQ = tensor(C, p, A, b, newborn, lam, alive)
    expectedC = D[:, None] * C * D[None, :]
    expectedT = beforeT * D[:, None, None] * D[None, :, None] * D[None, None, :]
    expectedQ = beforeQ * D[:, None, None, None] * D[None, :, None, None] * D[None, None, :, None] * D[None, None, None, :]
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        fp = [fnp.asarray(z) for z in (C, p, A, b, host, alive)]
        nb = (p.shape[1], fnp.asarray(selected), fnp.asarray(base), fnp.asarray(t), fnp.asarray(s))
        state = dict(columns=oldcols, Q=fnp.asarray(q) if oldcols else None,
                     pre_basis=None, core=fnp.asarray(np.eye(rank)),
                     F=fnp.asarray(F) if mode != 'lazy' else None, lazy=None)
        if mode == 'lazy':
            state['lazy'] = {key: fnp.asarray(value) for key, value in
                             [('map', fmap), ('base', fbase), ('join', fjoin)]}
        retained = {k: state[k] for k in ('F', 'core', 'lazy')}
        result = apply(*fp[:4], nb, state, fp[4], fp[5])
        assert all(state[k] is v for k, v in retained.items())
        C2, p2, A2, b2 = [np.asarray(v).copy() for v in result[:4]]
        nb2 = (result[4][0], *[np.asarray(v).copy() for v in result[4][1:]])
        D2 = np.asarray(result[5]).copy()
        Qbasis = np.asarray(state['Q']).copy() if oldcols else None
    T2, Q2 = tensor(C2, p2, A2, b2, nb2, lam, alive)
    W = rng.normal(size=(n, 4))
    true_push = np.einsum('ia,jb,kc,ijk->abc', W, W, W, expectedT, optimize=True)
    test_push = np.einsum('ia,jb,kc,ijk->abc', W, W, W, T2, optimize=True)
    qtrue = np.einsum('ia,ja,ka,la,ijkl->a', W, W, W, W, expectedQ, optimize=True)
    qtest = np.einsum('ia,ja,ka,la,ijkl->a', W, W, W, W, Q2, optimize=True)
    e = dict(C=error(C2, expectedC), T=error(T2, expectedT), Q=error(Q2, expectedQ),
             next_T=error(test_push, true_push), next_Q_diagonal=error(qtest, qtrue),
             active_diagonal=error(np.diag(C2)[alive], host[alive]))
    if oldcols:
        e['shared_reconstruction'] = error(Qbasis.astype(np.float64) @ F.astype(np.float64),
                                          D[:, None] * (q.astype(np.float64) @ F.astype(np.float64)))
    assert error(D2, D) <= 1e-14
    assert all(v <= (1e-12 if k in ('C', 'Q', 'next_Q_diagonal', 'active_diagonal') else 2e-6) for k,v in e.items())
    # Null action and persistent-factor ownership are exact at the stored dtypes.
    ns['_CVR_RESTORE'] = False
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        state0 = dict(columns=0, pre_basis=None)
        null = apply(*fp[:4], nb, state0, fp[4], fp[5])
        for original, got in zip((C, p, A, b), null[:4]):
            assert original.tobytes() == np.asarray(got).tobytes()
        for original, got in zip((selected, base, t, s), null[4][1:]):
            assert original.tobytes() == np.asarray(got).tobytes()
    ns['_CVR_RESTORE'] = True
    rows.append(dict(mode=mode, errors=e, flops=ctx.summary_dict()['flops_used']))

rejections = 0
for bad in (0., -1., np.nan, np.inf):
    h = host.copy(); h[0] = bad
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        try:
            apply(*fp[:4], nb, dict(columns=0), fnp.asarray(h), fp[5])
        except RuntimeError as e:
            assert str(e) == 'CVR strictly positive finite active variances'
            rejections += 1
assert rejections == 4
rec = dict(passed=True, rows=rows, invalid_domain_rejections=rejections,
           helper_sha256=hashlib.sha256(code.encode()).hexdigest(),
           elapsed_s=time.monotonic()-began,
           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE / 'unit.json').open('x') as stream:
    json.dump(rec, stream, indent=2)
print(json.dumps(rec), flush=True)
