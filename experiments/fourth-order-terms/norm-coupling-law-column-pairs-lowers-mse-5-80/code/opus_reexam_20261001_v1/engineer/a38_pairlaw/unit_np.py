"""TAG opus_engineer a38 NP unit check (light, seconds, no estimator): the pair-law fragment against a float64 numpy transcription of the
lead's O/ncar6.py at level 'read' (dfit 2: the multiple of C = RHO[l] x the diagonal regression's; forms 1; arest False/True; k21 on),
on random data with the loop's shapes in small. Two things are checked at one layer l with a given state (w, x, partner):
  the read block   (pp^2 lam)[resp] pp[live]' - f_i (B - aB C)_ij,  B = (pp w + pp[:, partner] x) pp', and the stored diagB (response rows), aB
  the update       carried w, x and the newborn pairs' w+, w-, x from the law (S2, rv, D at the hubs), compared as w / lam and x / lam of the
                   NEXT layer's roster (lam' = lam s_carry for carried columns, +- a s_new for the newborn)
The transcription regresses over the response rows (the native choice); everything else is the reference's arithmetic.
usage: unit_np.py"""
from pathlib import Path
import math
import numpy as np
import flopscope as flops
fnp = flops.numpy
HERE = Path(__file__).resolve().parent


class _E2Hold(list):
    def __call__(self, x):
        self.append(x)
        return x


def _unique_rows_via_scatter(A, rows):
    return A[rows]


rng = np.random.default_rng(21)
n, R, K = 192, 520, 70
worst = 0.0
for l, arest, diag_on, k21_on in ((5, False, 1.0, True), (5, True, 1.0, True), (7, False, 0.0, True), (0, False, 1.0, True)):
    ns = dict(fnp=fnp, _E2Hold=_E2Hold, _unique_rows_via_scatter=_unique_rows_via_scatter, _S19_LAM=(1.0032287195131746, 0.9932375877606482))
    exec(compile((HERE/'fragment_np_v1.py').read_text(), 'fragment_np_v1.py', 'exec'), ns)
    strength = 0.9
    ns.update(_NP_AREST=arest, _NP_DIAG=diag_on, _NP_K21=k21_on, _NP_STRENGTH=(0.0,) + (strength,) * 14 + (0.0,))
    carry, s_new = ns['_S19_LAM']
    rho = ns['_NP_RHO'][l]
    Rl = R if l else 0
    half = Rl // 2
    pp = rng.standard_normal((n, Rl)).astype(np.float32) * 0.1
    lam = np.concatenate([rng.uniform(0.5, 1.5, half), -rng.uniform(0.5, 1.5, half)]).astype(np.float32)
    partner = np.concatenate([np.arange(half, Rl), np.arange(0, half)])
    w = (rng.standard_normal(Rl) * 0.3); x = rng.standard_normal(half) * 0.2; x = np.concatenate([x, x])
    X0 = rng.standard_normal((n, n)); C = (X0 @ X0.T / n + 0.2).astype(np.float32).astype(np.float64); C = (C + C.T) / 2
    K21 = (rng.standard_normal((n, n)) * 0.05).astype(np.float32).astype(np.float64)
    mu = rng.standard_normal(n) * 0.7; vv = np.diag(C).copy(); sg = np.sqrt(vv); al = mu / sg
    ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al]); d = ph / sg
    m = sg * ph + mu * g + rng.standard_normal(n) * 0.01
    alive = np.ones(n, dtype=bool); resp = np.sort(rng.choice(n, K, replace=False)); rmask = np.zeros(n, dtype=bool); rmask[resp] = True
    live = np.arange(n)
    tp = rng.uniform(0.5, 1.5, K) * sg[resp]; tm = -rng.uniform(0.5, 1.5, K) * sg[resp]; asel = rng.uniform(0.2, 1.0, K)
    radial = 0.07
    t_full = rng.uniform(0.5, 1.5, n) * sg; s_full = -rng.uniform(0.5, 1.5, n) * sg; t_full[resp] = tp; s_full[resp] = tm
    # ---- reference (ncar6.py, level 'read', dfit 2), float64
    p64, l64 = pp.astype(np.float64), lam.astype(np.float64)
    if Rl:
        B = (p64 * w + p64[:, partner] * x) @ p64.T; B = (B + B.T) / 2
        trC = vv.sum(); Cmu = C @ mu
        Xr = np.stack([trC * vv, mu * mu, 2 * mu * Cmu], 1)
        co = np.linalg.lstsq(Xr[resp], np.diag(B)[resp], rcond=None)[0]
        aB = rho * co[0] * trC
        Rm = B - aB * C; Rm = Rm - np.diag(np.diag(Rm))
        Fd = -(strength * mu / (n * vv))[:, None] * Rm
        want_plane = np.zeros((n, n)); want_plane[np.ix_(resp, live)] = ((p64 * p64 * l64)[resp] @ p64[live].T) + Fd[np.ix_(resp, live)]
        diagB = np.diag(B).copy()
    else:
        aB = 0.0; diagB = np.zeros(n); want_plane = None
    r2 = vv * ((1 + al * al) * g + al * ph); r3 = vv * sg * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = vv * vv * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
    gam = g - m * d; V = 2 * m * (1 - g); S2 = 2 * (gam - g * g); S3 = 3 * (1 - g) * (r2 - 2 * m * m)
    S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2; e3 = -al * ph / vv; varG = r2 - (sg * ph + mu * g) ** 2
    gb = gam.mean(); Co = C - np.diag(vv); Co2 = Co * Co; g2 = g * g
    if Rl:
        c = l64 * (V @ p64); K3Vd = (p64 * p64) @ c; w_car, x_car = 2 * (gb * w + c), 2 * gb * x
    else:
        K3Vd = np.zeros(n); w_car = x_car = np.zeros(0)
    a_rest = 0.75 * radial / 3 * np.trace(C) - aB
    Ntd = diagB + a_rest * vv; Pv = gb * Ntd + 2 * (Co2 @ gam)
    rv = d * (Co @ V) + (e3 / 2) * Pv + S3
    dgn = S4 + V * (Co @ V) + 2 * gam * (Co2 @ gam) - 2 * g2 * (Co2 @ g2) + gam * gb * Ntd
    if k21_on and l >= 1:
        Ko = K21 - np.diag(np.diag(K21)); KoV = Ko @ V; KoTg = Ko.T @ gam; KC = Ko * Co
        rv = rv + d * KoTg + (e3 / 2) * KoV
        dgn = dgn + V * KoTg + gam * KoV - 2 * g * d * (KC @ g2) - 2 * g2 * (KC.T @ (g * d))
    D = dgn - g2 * ((Co2 @ S2) + gb * diagB + K3Vd)
    if arest: D = D - gb * a_rest * varG
    gap = tp - tm; a11, a12, a22 = S2[resp], rv[resp], diag_on * D[resp]
    wp = 2 * (a11 * tm * tm - 2 * a12 * tm + a22) / (gap * gap); wm = 2 * (a11 * tp * tp - 2 * a12 * tp + a22) / (gap * gap)
    xx = 2 * (-a11 * tp * tm + a12 * (tp + tm) - a22) / (gap * gap)
    lam_next = np.concatenate([l64 * carry, asel * s_new, -asel * s_new])
    want_q = np.concatenate([w_car, wp, wm]) / lam_next; want_xq = np.concatenate([x_car, xx, xx]) / lam_next
    # ---- fragment
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        S = ns['_NP_STATE']
        F = fnp.asarray
        ops0 = len(ctx._op_log); used0 = int(ctx.summary_dict()['flops_used'])
        if Rl:
            S.update(q=F((w / l64).astype(np.float32)), xq=F((x / l64).astype(np.float32)), partner=F(partner))
            P, L = F(pp), F(lam)
            left2 = (P * P)[F(resp)]
            ops0 = len(ctx._op_log); used0 = int(ctx.summary_dict()['flops_used'])
            context = ns['_np_fold'](l, left2, P, F(resp), L, F(mu), F(np.maximum(vv, 1e-10)), F(vv), F(C))
            ops1 = len(ctx._op_log); used1 = int(ctx.summary_dict()['flops_used'])
            block = left2 @ (P[F(live)] * L).T
            plane = fnp.zeros((n, n), dtype=fnp.float32)
            fnp.put(plane, F((resp * n)[:, None] + live), block)
            ops2 = len(ctx._op_log)
            ns['_np_forms'](l, plane, context, F(rmask), F(alive), F(C.astype(np.float32)))
            ops3 = len(ctx._op_log)
            got_plane = np.asarray(plane, dtype=np.float64)
            got_diag = np.asarray(S['diag'], dtype=np.float64); got_aB = float(np.asarray(S['aB']))
        else:
            S.update(q=None, xq=None, partner=None, diag=None, aB=None, rows=None)
            P, L = F(pp), F(lam); ops1 = ops0; ops2 = ops3 = ops0; used1 = used0
        K32 = K21.astype(np.float32)
        loc = dict(lam=L, pp=P, m=F(m), mu=F(mu), vv=F(vv), aux=dict(gate=F(g), alpha=F(al), pdf=F(ph)), d=F(d), Cpre=F(C), _plane_cov32=F(C.astype(np.float32)) if l else None,
                   selected=F(resp), response_indices=F(resp), radial=radial, _ps_K2132=F(K32), _ps_kdiag=F(np.diag(K21).copy()), t=F(t_full), s=F(s_full), _e2_asel=F(asel.astype(np.float32)))
        ops4 = len(ctx._op_log); used4 = int(ctx.summary_dict()['flops_used'])
        ns['_np_update'](l, loc)
        ops5 = len(ctx._op_log); used5 = int(ctx.summary_dict()['flops_used'])
        got_q = np.asarray(S['q'], dtype=np.float64); got_xq = np.asarray(S['xq'], dtype=np.float64); got_partner = np.asarray(S['partner'])
    errs = {}
    if Rl:
        fold = want_plane.copy(); fold[np.ix_(resp, live)] -= (p64 * p64 * l64)[resp] @ p64[live].T
        off = ~np.eye(n, dtype=bool)                                    # the loop overwrites the plane's diagonal afterwards
        errs['plane'] = np.abs((got_plane - want_plane)[off]).max() / np.abs(fold[off]).max()
        errs['diagB'] = np.abs(got_diag - diagB[resp]).max() / np.abs(diagB[resp]).max(); errs['aB'] = abs(got_aB - aB) / abs(aB)
    # the hubs' t+ and t- pass through float32, as in the loop
    errs['q'] = np.abs(got_q - want_q).max() / np.abs(want_q).max(); errs['xq'] = np.abs(got_xq - want_xq).max() / max(np.abs(want_xq).max(), 1e-300)
    want_partner = np.concatenate([partner, np.arange(Rl + K, Rl + 2 * K), np.arange(Rl, Rl + K)])
    errs['partner'] = float(not np.array_equal(got_partner, want_partner))
    worst = max(worst, max(errs.values()))
    print('layer %d arest %d diag %g: %s | charged ops: fold %d, forms %d, update %d | flops: fold %d, update %d' % (
        l, arest, diag_on, ' '.join('%s %.1e' % kv for kv in errs.items()), ops1 - ops0, ops3 - ops2, ops5 - ops4, used1 - used0, used5 - used4))
print('PASS' if worst < 2e-3 else 'FAIL')
