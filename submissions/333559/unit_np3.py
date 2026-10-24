"""TAG opus_engineer a38 NP v3 unit check (light, seconds, no estimator): the lean pair-law fragment against a float64 numpy
transcription of the lead's O/ncar6.py at level 'read' (dfit 2 or 3, forms 1-3, the carried weights' source V or a given beta),
with the native choices of v1: the diagonal regression over
the response rows, diagB and (pp o pp) c equal to 0 at the hubs that are no response rows, the loop's unmasked t+ / t-.
Random data with the loop's shapes in small; the hubs and the response rows are DIFFERENT sets. Checked at one layer l with a state:
  the read block   (left factor returned by _np_left) (pp[live] lam)' + the forms, against (pp^2 lam)[resp] pp[live]' - f_i (B - aB C)_ij
  the stored diagB (every unit, 0 outside the response rows) and aB, both times kap = strength / n
  the update       QX = kap' (w / lam, x / lam) on the next layer's roster, and the partner positions
and the requests each of the three functions makes (logged operations + views, as the platform counts round trips).
usage: unit_np3.py [fragment file]"""
from pathlib import Path
import math
import sys
import time
import numpy as np
import flopscope as flops
from flopscope import _ndarray
fnp = flops.numpy
HERE = Path(__file__).resolve().parent
FRAGMENT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE/'fragment_np_v3.py'

VIEWS = [0]
_getitem = _ndarray.FlopscopeArray.__getitem__


def counted_getitem(self, key):
    VIEWS[0] += 1
    return _getitem(self, key)


_ndarray.FlopscopeArray.__getitem__ = counted_getitem
_ndarray.FlopscopeArray.T = property(lambda self: (VIEWS.__setitem__(0, VIEWS[0] + 1), self.transpose())[1])


class _E2Hold(list):
    def __call__(self, x):
        self.append(x)
        return x


def _unique_rows_via_scatter(A, rows):
    return A[rows]


def requests(ctx, start):
    """Logged operations + views since `start` (an advanced index is both a logged operation and a counted __getitem__: once)."""
    ops = len(ctx._op_log) - start[0]
    gathers = sum(1 for x in list(ctx._op_log)[start[0]:] if 'getitem' in x.op_name)
    return ops + (VIEWS[0] - start[1]) - gathers


rng = np.random.default_rng(21)
n, R, K = 192, 520, 70
worst = 0.0
for l, arest, diag_on, strength, forms, dfit, beta_on in ((5, False, 1.0, 0.9, 1, 2, False), (5, True, 1.0, 0.9, 2, 2, False), (7, False, 0.0, 0.9, 3, 2, True), (0, False, 1.0, 0.9, 3, 2, True),
                                                          (13, False, 1.0, 0.9, 3, 2, True), (14, False, 1.0, 0.9, 3, 3, False), (6, False, 1.0, 0.9, 3, 3, True), (6, False, 1.0, 0.9, 1, 3, False)):
    E2L = {}
    ns = dict(fnp=fnp, _E2Hold=_E2Hold, _unique_rows_via_scatter=_unique_rows_via_scatter, _S19_LAM=(1.0032287195131746, 0.9932375877606482), _E2_LAYER=E2L)
    exec(compile(FRAGMENT.read_text(), FRAGMENT.name, 'exec'), ns)
    bands = (0.0,) + (strength,) * 6 + (1.3 * strength,) * 8 + (0.0,)             # two strengths, to check the per-layer constants
    ns.update(_NP_AREST=arest, _NP_DIAG=diag_on, _NP_STRENGTH=bands, _NP_FORMS=forms, _NP_DFIT=dfit, _NP_BETA=beta_on, _NP_DSCALE=0.95)
    carry, s_new = ns['_S19_LAM']
    rho = ns['_NP_RHO'][l]
    kap = (bands[l] or bands[1]) / n; kap_next = (bands[l + 1] or 1.0) / n
    Rl = R if l else 0
    half = Rl // 2
    pp = rng.standard_normal((n, Rl)).astype(np.float32) * 0.1
    lam = np.concatenate([rng.uniform(0.5, 1.5, half), -rng.uniform(0.5, 1.5, half)]).astype(np.float32)
    partner = np.concatenate([np.arange(half, Rl), np.arange(0, half)])
    if 'sizes' in ns['_NP_STATE'] and Rl:   # v5: two cohorts of different sizes, roster (+, -) (+, -); lam and the partners follow that roster
        k1 = half // 3; k2 = half - k1
        partner = np.concatenate([np.arange(k1, 2 * k1), np.arange(0, k1), np.arange(2 * k1 + k2, Rl), np.arange(2 * k1, 2 * k1 + k2)])
    w = (rng.standard_normal(Rl) * 0.3); x = rng.standard_normal(Rl) * 0.2; x = (x + x[partner]) / 2 if Rl else x   # one cross weight per pair
    X0 = rng.standard_normal((n, n)); C = (X0 @ X0.T / n + 0.2).astype(np.float32).astype(np.float64); C = (C + C.T) / 2
    K21 = (rng.standard_normal((n, n)) * 0.05).astype(np.float32).astype(np.float64)
    mu = rng.standard_normal(n) * 0.7; vv = np.diag(C).copy(); sg = np.sqrt(vv); al = mu / sg
    ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al]); d = ph / sg
    m = sg * ph + mu * g + rng.standard_normal(n) * 0.01
    alive = np.ones(n, dtype=bool); alive[rng.choice(n, 9, replace=False)] = False
    resp = np.sort(rng.choice(np.flatnonzero(alive), K, replace=False)); rmask = np.zeros(n, dtype=bool); rmask[resp] = True
    hubs = np.sort(np.concatenate([resp[:K - 8], rng.choice(np.flatnonzero(~rmask), 8, replace=False)]))   # 8 hubs are no response rows
    live = np.flatnonzero(alive)
    radial = 0.07 if l else 0.0
    t_full = rng.uniform(0.5, 1.5, n) * sg; s_full = -rng.uniform(0.5, 1.5, n) * sg
    delta = t_full - s_full; birth = d * (1 + radial / 3); a_full = birth / delta
    tp, tm, asel = t_full[hubs], s_full[hubs], a_full[hubs]
    # ---- reference (ncar6.py, level 'read', dfit 2, forms 1), float64
    p64, l64 = pp.astype(np.float64), lam.astype(np.float64)
    f = bands[l] * mu / (n * vv)
    if Rl:
        B = (p64 * w + p64[:, partner] * x) @ p64.T; B = (B + B.T) / 2
        trC = vv.sum(); Cmu = C @ mu
        Xr = np.stack([trC * vv] + ([np.ones(n)] if dfit == 3 else []) + [mu * mu, 2 * mu * Cmu], 1)
        co = np.linalg.lstsq(Xr[resp], np.diag(B)[resp], rcond=None)[0]
        aB = 0.95 * (rho if dfit == 2 else 1.0) * co[0] * trC
        Rm = B - aB * C
        if forms >= 2: Rm = Rm - co[-2] * np.outer(mu, mu)
        if forms >= 3: Rm = Rm - co[-1] * (np.outer(mu, Cmu) + np.outer(Cmu, mu))
        want_plane = ((p64 * p64 * l64)[resp] @ p64[live].T) - (f[:, None] * Rm)[np.ix_(resp, live)]
        diagB = np.diag(B) * rmask
    else:
        aB = 0.0; diagB = np.zeros(n); want_plane = None
    r2 = vv * ((1 + al * al) * g + al * ph); r3 = vv * sg * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = vv * vv * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
    gam = g - m * d; V = 2 * m * (1 - g); S2 = 2 * (gam - g * g); S3 = 3 * (1 - g) * (r2 - 2 * m * m)
    S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2; e3 = -al * ph / vv; varG = r2 - (sg * ph + mu * g) ** 2
    gb = gam.mean(); Co = C - np.diag(vv); Co2 = Co * Co; g2 = g * g
    beta = (V + rng.standard_normal(n) * 0.05) * alive                        # stands for the card's beta
    if Rl:
        c = l64 * ((beta if beta_on else V) @ p64); K3Vd = ((p64 * p64) @ c) * rmask; w_car, x_car = 2 * (gb * w + c), 2 * gb * x
    else:
        K3Vd = np.zeros(n); w_car = x_car = np.zeros(0)
    a_rest = 0.75 * radial / 3 * np.trace(C) - aB
    Ntd = diagB + a_rest * vv; Pv = gb * Ntd + 2 * (Co2 @ gam)
    rv = d * (Co @ V) + (e3 / 2) * Pv + S3
    dgn = S4 + V * (Co @ V) + 2 * gam * (Co2 @ gam) - 2 * g2 * (Co2 @ g2) + gam * gb * Ntd
    if l >= 1:
        Ko = K21 - np.diag(np.diag(K21)); KoV = Ko @ V; KoTg = Ko.T @ gam; KC = Ko * Co
        rv = rv + d * KoTg + (e3 / 2) * KoV
        dgn = dgn + V * KoTg + gam * KoV - 2 * g * d * (KC @ g2) - 2 * g2 * (KC.T @ (g * d))
    D = dgn - g2 * ((Co2 @ S2) + gb * diagB + K3Vd)
    if arest: D = D - gb * a_rest * varG
    gap = tp - tm; a11, a12, a22 = S2[hubs], rv[hubs], diag_on * D[hubs]
    wp = 2 * (a11 * tm * tm - 2 * a12 * tm + a22) / (gap * gap); wm = 2 * (a11 * tp * tp - 2 * a12 * tp + a22) / (gap * gap)
    xx = 2 * (-a11 * tp * tm + a12 * (tp + tm) - a22) / (gap * gap)
    lam_next = np.concatenate([l64 * carry, asel * s_new, -asel * s_new])
    want_q = np.concatenate([w_car, wp, wm]) / lam_next; want_xq = np.concatenate([x_car, xx, xx]) / lam_next
    # ---- fragment
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        S = ns['_NP_STATE']
        F = fnp.asarray
        t0 = time.perf_counter(); const = ns['_np_const'](n); t_const = time.perf_counter() - t0
        P, L = F(pp), F(lam)
        counts = {}; bill = {}
        if Rl:
            S.update(QX=F((kap * np.stack([w / l64, x / l64])).astype(np.float32)), **(dict(sizes=[half // 3, half // 3, half - half // 3, half - half // 3]) if 'sizes' in S else dict(partner=F(partner[None, :]))))
            pp2 = P * P
            args = (l, pp2, P, F(resp), F(rmask), F(alive), L, F(mu), F(np.maximum(vv, 1e-10)), F(vv), F(C))
            start = (len(ctx._op_log), VIEWS[0]); used = int(ctx.summary_dict()['flops_used'])
            left2 = ns['_np_left'](*args)
            counts['left'] = requests(ctx, start) - 1; bill['left'] = int(ctx.summary_dict()['flops_used']) - used   # - 1: the stub's row gather stands for the parent's own
            if l <= const['last']:
                block = left2 @ (P[F(live)] * L).T
                plane = fnp.zeros((n, n), dtype=fnp.float32)
                fnp.put(plane, F((resp * n)[:, None] + live), block)
                c32 = F(C.astype(np.float32))
                start = (len(ctx._op_log), VIEWS[0])
                ns['_np_forms'](l, plane, c32)
                counts['forms'] = requests(ctx, start)
                got_plane = np.asarray(plane, dtype=np.float64)[np.ix_(resp, live)]
                got_diag = np.asarray(S['diag'], dtype=np.float64); got_aB = float(np.asarray(S['ab'])[0])
        E2L.update(ps_Cpre32=F(C.astype(np.float32)), Cpre=F(C), np_column_sum=F(beta @ p64))
        K32 = K21.astype(np.float32)
        args = (l, L, P, F(mu), F(vv), F(m), F(d), dict(gate=F(g), alpha=F(al), pdf=F(ph), sigma=F(sg)), F(t_full), F(s_full), F(delta), F(birth),
                F(np.maximum(vv, 1e-10)), F(rmask), F(hubs), (F(np.float64(radial)) if l else 0.0), F(K32), F(np.diag(K21).copy()))
        start = (len(ctx._op_log), VIEWS[0]); used = int(ctx.summary_dict()['flops_used'])
        ns['_np_update'](*args)
        counts['update'] = requests(ctx, start); bill['update'] = int(ctx.summary_dict()['flops_used']) - used
        got = None if S['QX'] is None else (np.asarray(S['QX'], dtype=np.float64), np.asarray(S['partner']) if 'partner' in S else S['sizes'])
    errs = {}
    if l >= const['last']:
        errs['no update at the last layer'] = float(got is not None)
    if Rl and l <= const['last']:
        fold = (f[:, None] * Rm)[np.ix_(resp, live)]
        offd = resp[:, None] != live[None, :]                              # the loop overwrites the plane's diagonal afterwards
        errs['plane'] = np.abs((got_plane - want_plane)[offd]).max() / np.abs(fold[offd]).max()
        errs['diagB'] = np.abs(got_diag - kap * diagB).max() / np.abs(kap * diagB).max(); errs['aB'] = abs(got_aB - kap * aB) / abs(kap * aB)
    if l < const['last']:
        errs['q'] = np.abs(got[0][0] - kap_next * want_q).max() / np.abs(kap_next * want_q).max()
        errs['xq'] = np.abs(got[0][1] - kap_next * want_xq).max() / np.abs(kap_next * want_xq).max()
        want_partner = np.concatenate([partner, np.arange(Rl + K, Rl + 2 * K), np.arange(Rl, Rl + K)])
        errs['partner'] = float(not np.array_equal(got[1][0], want_partner)) if 'partner' in S else float(got[1] != (([half // 3, half // 3, half - half // 3, half - half // 3] if Rl else []) + [K, K]))
    worst = max(worst, max(errs.values()))
    print('layer %2d arest %d diag %g forms %d dfit %d beta %d: %s | requests %s | flops %s | tables built in %.1f ms' % (
        l, arest, diag_on, forms, dfit, beta_on, ' '.join('%s %.1e' % kv for kv in errs.items()), counts, bill, 1e3 * t_const))
print('PASS' if worst < 2e-3 else 'FAIL')
