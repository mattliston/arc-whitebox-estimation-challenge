# BEGIN OPUS-ENGINEER BATCH-6 HEAD COLUMNS (a46 b6 v2, 2026-10-10; v2 = v1 + qb leaves its c and den for the batch-7 columns (_B67_SHARE), no new operation; inputs of the learned mean correction, made only when the head's
# columns name them: lever HC@<weights with "columns">, fragment_hc_v2). The physicist's definitions (O/physicist/b6cols.py, from what
# the pair map receives at layers 1-14), each standardised over the read's response rows like column 6:
#   'm2'    the second-order mean term over sigma: t3^2 He4(a) phi(a) / 72,  t3 = k3 / (v s), k3 = the K21 plane's diagonal, a = mu / s
#   'qb'    the read's plane (response rows x live columns) without its diagonal against c = (g / s)^2 mu:  (cp c)_i / (c.mu - c_i mu_i) / v_i
#   'eold'  sum over the columns of the cohorts aged 3 and more (born at j <= l - 3) of |lam_r| pp_ir^2, over v
#   'qold'  the local quartic of those cohorts' pairs (K4L's hub terms, strength 1, the isotropic part kept), over v^2
# Native: m2 and qb from vectors the loop holds (about 14 and 13 requests a layer; qb one float32 matrix-vector product, 2 Mflop);
# eold one float32 matrix-vector product over the old columns' squares (pp^2 exists); qold K4L v6's Horner block on the old pairs, with
# the birth coefficients K4L keeps for every cohort of the prediction (_K4L_RAW), 16 flops per element of (units x old pairs).
_B6_NAMES = ('m2', 'qb', 'eold', 'qold')
_B6_CONST = {}
_B67_SHARE = {}


def _b6_m2(l, a, vc, s, k3, response_mask, nrows):
    if 'm2' not in _HC_COLUMNS or _HC_U is None or not (1 <= l <= 14):
        return
    hold = _E2Hold()
    a2 = hold(a * a)
    t3 = hold(k3 / hold(vc * s))
    he4 = hold(hold(hold(a2 * a2) - hold(a2 * 6.0)) + 3.0)
    ph = hold(hold(fnp.exp(hold(a2 * -0.5))) * 0.3989422804014327)
    _hc_put(l, 'm2', hold(hold(hold(t3 * t3) * hold(he4 * ph)) * (1.0 / 72.0)), response_mask, nrows)


def _b6_qb(l, cp32, cpdiag, mu, g, s, vc, response_mask, nrows):
    if 'qb' not in _HC_COLUMNS or _HC_U is None or not (1 <= l <= 14):
        return
    hold = _E2Hold()
    gs = hold(g / s)
    c = hold(hold(gs * gs) * mu)
    cm = hold(c * mu)
    den = hold(hold(fnp.sum(cm)) - cm)
    num = hold(hold(hold(cp32 @ hold(c.astype(fnp.float32))).astype(fnp.float64)) - hold(hold(cpdiag.astype(fnp.float32).astype(fnp.float64)) * c))
    _hc_put(l, 'qb', hold(num / hold(den * vc)), response_mask, nrows)
    _B67_SHARE['cden'] = (l, c, den)   # for qk (fragment b7 v2): the same c and den


def _b6_old(l, pp, pp2, lam, state, vc, response_mask, nrows):
    want_e, want_q = 'eold' in _HC_COLUMNS, 'qold' in _HC_COLUMNS
    if _HC_U is None or not (want_e or want_q) or not (1 <= l <= 14):
        return
    cohorts = []
    for j in range(0, l - 2):                        # born at j <= l - 3: aged 3 and more
        start, width = state['cohorts'][j]
        start, K = int(start), int(width) // 2
        if K:
            cohorts.append((start, K, j))
    if not cohorts:
        return
    hold = _E2Hold()
    if want_e:
        spans = [(start, start + 2 * K) for start, K, _j in cohorts]
        if all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1)):
            a0, a1 = spans[0][0], spans[-1][1]
            e = hold(pp2[:, a0:a1] @ hold(fnp.abs(lam[a0:a1])))
        else:
            e = hold(fnp.concatenate([pp2[:, x0:x1] for x0, x1 in spans], axis=1) @ hold(fnp.abs(hold(fnp.concatenate([lam[x0:x1] for x0, x1 in spans])))))
        _hc_put(l, 'eold', hold(hold(e.astype(fnp.float64)) / vc), response_mask, nrows)
    if want_q:
        cuts = sorted({c for start, K, _j in cohorts for c in (start, start + K, start + 2 * K)})
        pieces = hold(fnp.split(pp, cuts, axis=1))
        at = {c: i + 1 for i, c in enumerate(cuts)}
        if len(cohorts) == 1:
            start, K, j = cohorts[0]
            P, Q, table = pieces[at[start]], pieces[at[start + K]], _K4L_RAW[j]
        else:
            P = hold(fnp.concatenate([pieces[at[start]] for start, K, _j in cohorts], axis=1))
            Q = hold(fnp.concatenate([pieces[at[start + K]] for start, K, _j in cohorts], axis=1))
            table = hold(fnp.concatenate([_K4L_RAW[j] for _start, _K, j in cohorts], axis=1))
        e1, e2, e3, e4 = hold(fnp.unstack(table))
        D = hold(P - Q)
        h = hold(D * e4)                             # D (e1 p^3 + D (e2 p^2 + D (e3 p + e4 D))), summed over the pairs (K4L v6's block)
        x = hold(P * e3)
        hold(fnp.add(h, x, out=h))
        hold(fnp.multiply(h, D, out=h))
        p2 = hold(P * P)
        hold(fnp.multiply(p2, e2, out=x))
        hold(fnp.add(h, x, out=h))
        hold(fnp.multiply(h, D, out=h))
        hold(fnp.multiply(p2, P, out=p2))
        hold(fnp.multiply(p2, e1, out=p2))
        hold(fnp.add(h, p2, out=h))
        q = hold(hold(fnp.einsum('ij,ij->i', h, D, optimize=False)).astype(fnp.float64))
        _hc_put(l, 'qold', hold(q / hold(vc * vc)), response_mask, nrows)
# END OPUS-ENGINEER BATCH-6 HEAD COLUMNS
