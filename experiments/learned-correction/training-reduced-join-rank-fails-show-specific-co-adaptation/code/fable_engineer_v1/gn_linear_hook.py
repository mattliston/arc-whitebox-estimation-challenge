"""TAG fable-engineer (E-65b): torch drop-in for the GN v3 covariance-knob hook, the native way. fable_covhunter_v1/gn3_knobs.configure
installs a 'cov_pm' hook that, at layers 2-14, calls the WHOLE pair map a second time with the knob-modified K21 (iso, k21n) and a K31
remainder (k31), and keeps only that call's C: 13 of the 27 pair-map calls per net (~20% of a refit step). The pair series is linear
in T21 and T31, and the knobs leave diag(K21) = k3 unchanged, so C_second = C_first + (the T21- and T31-linear series terms of the
knob increments) exactly. The native card applies the knobs that way (_gn_dcov in the fired files, fable-porter PT-7); this hook is
that function in torch, returning C + dcov at layers 2-14 with the same knob strengths.
usage: import gn_linear_hook as gl; gl.install(card, torch, th)   (th = the SAME dict given to gn3_knobs.configure, default V3)"""

V3 = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212}


def install(card, torch, th=None):
    t = torch
    if th is not None and th.get('k22', 0):
        raise ValueError('k22 knob is not linear in the pair series: keep the original hook')
    th = dict(V3) if th is None else {k: v for k, v in th.items() if k in ('iso', 'k21n', 'k31')}
    iso, k21n, k31 = th.get('iso', 0.0), th.get('k21n', 0.0), th.get('k31', 0.0)
    derivatives = card.env['derivatives']
    rowf = card.env.get('_ROWF', [None])[0]

    def hook(l, C, ctx):
        if l < 2 or l > 14 or not (iso or k21n or k31):
            return C
        mu, Cpre, K21, k4, aux = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['aux']
        s = aux['sigma']; a = aux['alpha']
        v = t.diagonal(Cpre)
        A, _unused = derivatives(a, aux['pdf'], aux['gate'], aux['gaussian_mean'] / s, 5)
        A1, A2, A3, A4, A5 = [x.to(t.float32) for x in A[1:6]]
        ss = t.outer(s, s)
        rho = (Cpre / ss).to(t.float32)
        rho = rho - t.diag(t.diagonal(rho))
        rho2 = rho * rho / 2
        k3 = t.diagonal(K21)
        kink = (a.abs() < 1.5).to(t.float64)
        dK21 = ((iso * kink) * k3 / v)[:, None] * Cpre + (k21n * (1 - kink))[:, None] * K21
        dT21 = (dK21 / (v[:, None] * s[None, :])).to(t.float32)
        t4 = k4.expand(v.shape) / v ** 2
        r31 = rowf(mu, k3, v) if rowf is not None else None
        dT31 = (k31 * (t4 * r31 if r31 is not None else t4)).to(t.float32)[:, None] * rho
        t21a = t.outer(A2, A1) + rho * t.outer(A3, A2) + rho2 * t.outer(A4, A3)
        t31a = t.outer(A3, A1) + rho * t.outer(A4, A2) + rho2 * t.outer(A5, A3)
        X = dT21 * t21a / 2 + dT31 * t31a / 6
        dcov = (X + X.T).to(t.float64) * ss
        dcov = dcov - t.diag(t.diagonal(dcov))
        return C + dcov

    card.hooks['cov_pm'] = hook
    return card
