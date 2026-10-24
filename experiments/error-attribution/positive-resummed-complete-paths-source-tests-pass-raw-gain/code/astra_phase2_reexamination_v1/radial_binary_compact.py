"""Compact live-neuron reference: full means and exact cold diagonals.

Only active-active CP plane entries are materialized. The complete factor
bank is pushed to every output so all scalar third-cumulant reads remain
available. Active and birth counts round upward to 32 for billed kernels.
All arrays are NumPy here; this is the independent accuracy reference.
"""
import numpy as np
from scipy.special import ndtr
from k3_radial_numpy import nonlinear, star_push
from radial_cubic import plane


def select(score, floor):
    count = min(len(score), ((int(np.count_nonzero(score >= floor))+31)//32)*32)
    indices = np.argsort(-score, kind="stable")[:count]
    mask = np.zeros(len(score), dtype=bool)
    mask[indices] = True
    return indices, mask


def predict(W, hook=None, cov32=True, metric="covariance", collect_states=False, gate_floor=.001):
    depth, n, _ = W.shape
    m = np.zeros(n)
    C = np.eye(n)
    radial = 0.0
    p = np.zeros((n, 0), np.float32)
    lam = np.zeros(0, np.float32)
    star = np.zeros((n, n), np.float32)
    means, states = [], []
    for l in range(depth):
        w32 = W[l].astype(np.float32)
        w = W[l].astype(np.float64)
        mu = w.T @ m
        Cpre = (w32.T @ C.astype(np.float32) @ w32).astype(np.float64) if cov32 else w.T @ C @ w
        vv = np.diag(Cpre)
        live_indices, alive = select(ndtr(mu/np.sqrt(np.maximum(vv, 1e-10))), gate_floor)
        live = alive[:, None]*alive[None, :]
        pp = w32.T @ p
        cp_pre = np.zeros((n, n))
        if len(lam):
            sub = pp[live_indices]
            cp_pre[np.ix_(live_indices, live_indices)] = plane(sub, lam)
            diagonal = np.sum(pp.astype(np.float64)**3*lam.astype(np.float64), axis=1)
            np.fill_diagonal(cp_pre, diagonal)
        star_pre = star_push(W[l], star)
        sd = np.diag(star_pre).copy()
        star_pre = star_pre*live
        np.fill_diagonal(star_pre, sd)
        K21 = cp_pre+star_pre
        k4 = radial*vv*vv
        k22 = radial*(np.outer(vv, vv)+2*Cpre*Cpre)/3
        m, C, target, radial_new, aux = nonlinear(mu, Cpre, K21, k4, k22, metric)
        host = m.copy()
        if hook is not None:
            dm = np.asarray(hook(l, m, C, K21, aux, W[l]), np.float64)
            C = C-np.outer(m, dm)-np.outer(dm, m)-np.outer(dm, dm)
            m = m+dm
        means.append(m.copy())
        if l == depth-1:
            break
        g = aux["gate"]
        birth = aux["pdf"]/aux["sigma"]
        selected, births = select(np.minimum(g, 1-g), gate_floor)
        cv = np.diag(C).copy()
        C = C*live
        np.fill_diagonal(C, cv)
        td = np.diag(target).copy()
        target = target*live
        np.fill_diagonal(target, td)
        inherited = g[:, None]**2*g[None, :]*cp_pre
        bp = g[:, None]*Cpre
        np.fill_diagonal(bp, 0)
        residual = target-inherited
        beta = 2*(aux["gaussian_mean"]-host*g)
        innovation = np.diag(residual).copy()
        safe_birth = np.maximum(birth, 1e-12)
        u = np.clip(beta/safe_birth, -8*vv, 8*vv)
        disc = 4*innovation/safe_birth-3*u*u
        delta = np.minimum(np.sqrt(np.maximum(disc, .01*vv*vv)), 16*vv)
        t = (u+delta)/2
        s = (u-delta)/2
        a = birth/delta
        p1, p2 = bp.copy(), bp.copy()
        np.fill_diagonal(p1, t)
        np.fill_diagonal(p2, s)
        born = bp*bp*(birth*births)[None, :]+(beta*births)[:, None]*bp.T
        np.fill_diagonal(born, birth*births*(3*u*u+delta*delta)/4)
        cp_post = (inherited+born)*live
        p = np.concatenate([g.astype(np.float32)[:, None]*pp,
                            p1[:, selected].astype(np.float32), p2[:, selected].astype(np.float32)], axis=1)
        p = alive.astype(np.float32)[:, None]*p
        lam = np.concatenate([lam, a[selected].astype(np.float32), -a[selected].astype(np.float32)])
        rest = target-cp_post
        star = (3*rest.T-2*np.diag(np.diag(rest))).astype(np.float32)
        if collect_states:
            states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))
        radial = radial_new
    return np.stack(means), states
