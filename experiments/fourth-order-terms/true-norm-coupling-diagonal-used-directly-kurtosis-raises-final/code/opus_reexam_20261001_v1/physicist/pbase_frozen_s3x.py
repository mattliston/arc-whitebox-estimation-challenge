"""TAG opus-physicist PBASE (2026-10-02): THE PHYSICS BASE FOR THE REFIT. Terms that are right against the truth and that the present learned parts were trained without,
ONE carrier module with ONE KEYWORD PER TERM (lead 18:01Z), interface of O/ncar6.py / O/k31l.py: install(cd, torch, st, **keywords). st (the pool card's side table) is not used:
every term is elementwise on arrays the step already holds (no matrix product, no state of its own), so a step that runs again in a gradient run recomputes it and the terms that
sit inside the step and the pair map are differentiable. All keywords at 0 (the default) leave the card bit for bit as it is.

  so2m   second-order mean term INSIDE the pair map, layers lo..hi: mean += so2m t3^2 sigma He4(alpha) pdf / 72, with the consistent E relu^2 term (the per-unit post variance follows)
  so2c   second-order covariance list (the products of two third cumulants a b q c d e of so2.py, up to the first power of rho), layers lo..hi
  so2t   second-order part of the third-cumulant target (the diagonal post k3 and the plane cum(f_i, f_i, f_j)), layers lo..hi
  so2f   the second-order mean term at the final layer 15, added after the learned correction
  bsk    birth weight with the skew-corrected kink density: d (1 + radial / 3) (1 - bsk (t3 / 6) He3(alpha)), layers 0-14
  gsk    transport gate with the skew correction: g + gsk (t3 / 6) He2(alpha) pdf        gk4   and the kurtosis correction: - gk4 (t4 / 24) He3(alpha) pdf  (t4 = the card's per-unit k4 / sigma^4)
  gbk    transport gate with the collective shift: + gbk d bpre  (the feed of the cross form below into the carried third cumulant; a pure multiplicative mode gives 1.5)
         (the corrected gate moves the b vector, the inherited plane, the carried columns and the old-tier basis; the mean's own gate is not touched)
  x1, x2 the cross form of the collective mode and the SPECIFIC third cumulant in the (3,1) plane, layers lo..hi, through remainder31:
         K31_ij += k31 (x1 k3s_i b_j + x2 b_i Ks_ij),  Ks = bank + star planes (ctx['cp'] + ctx['sp']), k3s its diagonal, b = bpre; truth planes of networks 35 / 100: x1 3.2, x2 10
  k31    overall strength of the cross form (the pool card passes its leaf nk31 here)
  t3 = k3 / sigma^3 with k3 = the diagonal of the K21 plane the pair map receives.

The so2 terms are physicist/so2_frozen_s2b.py (certificate t15_so2_cert.py) copied here unchanged; bsk / gsk / gk4 / gbk are the single-line changes of the step that e16b_f4b.py makes
(the step's source re-executed with those lines changed); x1 / x2 are physicist/bk3.py at level 'r31' with the specific planes. Parity with the screen harness's own arms:
t4_ora59.log (arms ending in p). On the trainer's per-layer card (otr_pl.py: the step already has its two lines birth and b replaced) the same two replacements are kept.
The strengths are read when the step uses them (cd.pbase[...]), each only as a factor of a product."""
import inspect
import math
import sys
import textwrap
import types

ARGS = dict(so2m=0.0, so2c=0.0, so2t=0.0, so2f=0.0, bsk=0.0, gsk=0.0, gk4=0.0, gbk=0.0, x1=0.0, x2=0.0, lo=1, hi=14)      # the install arguments besides k31 (for a module table)


def _A(t, a, ph, g, mgs, order):
    A = [mgs, g]; h0 = t.ones_like(a); h1 = a
    for r in range(2, order + 1):
        if r == 2: h = h0
        elif r == 3: h = h1
        else: h = a * h1 - (r - 3) * h0; h0, h1 = h1, h
        A.append((-1) ** (r - 2) * h * ph)
    return A


def single(t, mu, v, k3, order=1):
    """second-order additions (mean, E relu^2, E relu^3, E relu^4) for one unit."""
    v = v.clamp_min(1e-10); s = v.sqrt(); a = mu / s; ph = t.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    t3 = k3 / (v * s); q = t3 * t3
    he2, he3, he4 = a * a - 1, a ** 3 - 3 * a, a ** 4 - 6 * a * a + 3
    return q * s * he4 * ph / 72, -q * v * he3 * ph / 36, q * v * s * he2 * ph / 12, -q * v * v * a * ph / 3


def pair_list(t, A, Bi, rho, T, t3, kmax, parts='abqcde'):
    """(1/72) x the connected second-order list; A = list of standardized derivative vectors for the j side, Bi = the same for the i
    side (A itself for the covariance, the relu^2 list for E f_i^2 f_j, where the list is NOT symmetric and both orientations are written out)."""
    P = [None, rho, rho * rho / 2][:kmax + 1]; Tt = T.T
    def D(p, q, k0=0, left=Bi, right=A):
        acc = None
        for k in range(k0, kmax + 1):
            term = t.outer(left[p + k], right[q + k]); term = term if P[k] is None else P[k] * term
            acc = term if acc is None else acc + term
        return acc
    sym = Bi is A; X = t.zeros_like(rho); ti, tj = t3[:, None], t3[None, :]
    if 'a' in parts:
        Y = 6 * ti * T * D(5, 1); X = X + Y + (Y.T if sym else 6 * tj * Tt * D(1, 5))
    if 'b' in parts or 'q' in parts:
        cb, cq = 6.0 * ('b' in parts), 9.0 * ('q' in parts)
        Y = (cb * ti * Tt + cq * T * T) * D(4, 2); X = X + Y + (Y.T if sym else (cb * tj * T + cq * Tt * Tt) * D(2, 4))
    if 'c' in parts: X = X + 18 * T * Tt * D(3, 3)
    if 'd' in parts and kmax >= 1: X = X + 2 * ti * tj * D(3, 3, 1)
    if 'e' in parts and kmax >= 1:
        Y = ti * ti * D(6, 0, 1); X = X + Y + (Y.T if sym else tj * tj * D(0, 6, 1))
    return X / 72


def _so2(cd, t, mean, cov, tgt, final, lo, hi, kmax=1, parts='abqcde'):
    f64 = t.float64; raw2 = mean
    pair0 = cd.env['_nonlinear_pair_batch']
    offd = lambda x: x - t.diag(t.diagonal(x))

    def pair(mu, C, K21, k4, k22_override=None, metric='identity', **kw):
        m, covm, post21, radial, aux = pair0(mu, C, K21, k4, k22_override, metric, **kw)
        l = cd.call_layer
        if not (lo <= l <= min(hi, 14)) or not (mean or cov or tgt): return m, covm, post21, radial, aux
        s, a, ph, g, mg = (aux[k].to(f64) for k in ('sigma', 'alpha', 'pdf', 'gate', 'gaussian_mean')); v = s * s
        k3 = t.diagonal(K21).to(f64); t3 = k3 / (v * s); k4v = (k4 if t.is_tensor(k4) else t.tensor(float(k4), dtype=f64)).to(f64).expand(v.shape)
        m2, q2, q3, q4 = single(t, mu.to(f64), v, k3)
        d, e = ph / s, -a * ph / v
        r2g = (mu * mu + v) * g + mu * s * ph; r3g = mu * (mu * mu + 3 * v) * g + (mu * mu + 2 * v) * s * ph
        r2 = r2g + k3 * d / 3 + k4v * e / 12; r3 = r3g + k3 * g + k4v * d / 4      # the card's own raw moments (zero.py)
        mn = m + mean * m2; var = (r2 + raw2 * q2) - mn * mn
        covn, p21n = covm, post21; aux = dict(aux)
        if cov or tgt:
            rho = offd(C / t.outer(s, s)); T = offd(K21 / (v[:, None] * s[None, :])); off = offd(covm)
            A = _A(t, a, ph, g, mg / s, kmax + 7)
            dc1 = offd(pair_list(t, A, A, rho, T, t3, kmax, parts) * t.outer(s, s))
            if cov: covn = covm + cov * dc1
            if tgt:
                B = [(a * a + 1) * g + a * ph] + [2 * x for x in A[:-1]]
                dp1 = pair_list(t, A, B, rho, T, t3, kmax, parts) * t.outer(v, s)
                m1 = m + m2; dt = offd(dp1 - 2 * (m1[:, None] * (off + dc1) - m[:, None] * off))
                r2f = r2 + q2; pk3 = (r3 + q3) - 3 * m1 * r2f + 2 * m1 ** 3; pk3_0 = r3 - 3 * m * r2 + 2 * m ** 3
                p21n = post21 + tgt * t.diagonal_scatter(dt, pk3 - pk3_0)
                aux['post_k3'] = t.diagonal(p21n)
        if mean:
            covn = t.diagonal_scatter(covn, var); aux['post_v'] = var
        return mn, covn, p21n, radial, aux
    if mean or cov or tgt: cd.env['_nonlinear_pair_batch'] = pair
    if final:
        old = cd.hooks.get('delta')
        def delta(l, dl, ctx):
            if old is not None: dl = old(l, dl, ctx)
            if l != 15: return dl
            m2 = single(t, ctx['mu'].to(f64), ctx['vv'].to(f64), ctx['aux']['pre_k3'].to(f64))[0]
            return dl + (final * m2).to(dl.dtype)
        cd.hooks['delta'] = delta


def _step(cd, bsk, gate):
    """the card's step with the birth line and / or the transport gate changed (the source of the class's step, re-executed)."""
    owner = next(k for k in type(cd).__mro__ if 'step' in k.__dict__); src = textwrap.dedent(inspect.getsource(owner.step))
    pl = hasattr(cd, 'pl_kappa')      # the trainer's per-layer card: its step = the class's with the two lines below replaced (otr_pl._patched_step); they are kept
    now = getattr(cd.step, '__func__', None)
    assert now is owner.__dict__['step'] or (pl and 'per-layer kappa' in now.__code__.co_filename), 'the step is already replaced by something this module does not know'
    def edit(old, new):
        nonlocal src
        assert src.count(old) == 1, (src.count(old), old); src = src.replace(old, new)
    T3 = "(aux['pre_k3'] / aux['sigma'] ** 3)"; AL = "aux['alpha']"
    birth = 'birth = d * (1 + self.pl_kappa(layer) * radial / 3)' if pl else 'birth = d * (1 + radial / 3)'
    bline = 'b = %s * bpre + self.pl_bsrc(layer) * radial * d * vv / 6' if pl else 'b = %s * bpre + radial * d * vv / 6'
    edit('birth = d * (1 + radial / 3)\n', birth + (f" * (1 - self.pbase['bsk'] * {T3} * ({AL} ** 3 - 3 * {AL}) / 6)" if bsk else '') + '\n')
    if gate:
        edit("    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n",          # the dedented source: method body at four spaces
             "    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n"
             f"    gq = g + self.pbase['gsk'] * {T3} * ({AL} ** 2 - 1) * aux['pdf'] / 6 - self.pbase['gk4'] * (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 3 - 3 * {AL}) * aux['pdf'] / 24"
             " + self.pbase['gbk'] * d * bpre\n")
        edit('inherited = g[:, None] ** 2 * g[None, :] * cp', 'inherited = gq[:, None] ** 2 * gq[None, :] * cp')
        edit('p = t.cat((alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * pp),', 'p = t.cat((alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * pp),')
        edit('Q = alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * U)', 'Q = alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * U)')
    edit('b = g * bpre + radial * d * vv / 6\n', bline % ('gq' if gate else 'g') + '\n')
    space = dict(vars(sys.modules[(type(cd) if pl else owner).__module__]))      # the namespace the trainer's replacement uses (the card class's module) / the screen harness's (the step's module)
    exec(compile(src,'<%s.step with the physics base (pbase)%s>' % (owner.__name__, ', per-layer kappa and b source' if pl else ''), 'exec'), space)
    cd.step = types.MethodType(space['step'], cd)


def install(cd, torch, st=None, k31=1.0, so2m=0.0, so2c=0.0, so2t=0.0, so2f=0.0, bsk=0.0, gsk=0.0, gk4=0.0, gbk=0.0, x1=0.0, x2=0.0, lo=1, hi=14):
    t = torch; f64 = t.float64; lo, hi = int(lo), int(hi)
    cd.pbase = dict(bsk=bsk, gsk=gsk, gk4=gk4, gbk=gbk)
    if so2m or so2c or so2t or so2f: _so2(cd, t, so2m, so2c, so2t, so2f, lo, hi)
    if bsk or gsk or gk4 or gbk: _step(cd, bool(bsk), bool(gsk or gk4 or gbk))
    old_pin = cd.hooks.get('pair_in')

    def pin_hook(l, pin, ctx):      # always installed: the pool card calls every module's 'pair_in'
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if not (x1 or x2) or not (lo <= l <= hi) or not t.is_tensor(pin[2]) or pin[2].ndim != 2: return pin
        mu, Cpre, k21, k4v, k22, rem = pin
        eye = t.eye(k21.shape[0], dtype=t.bool); v = t.diagonal(Cpre).detach().to(f64); s = v.sqrt()
        b = ctx['bpre'].detach().to(f64); K = (ctx['cp'] + ctx['sp']).detach().to(f64); k3 = t.diagonal(K)
        add = (k31 * ((x1 * t.outer(k3, b) + x2 * b[:, None] * K) / (v[:, None] * s[:, None] * s[None, :]))).masked_fill(eye, 0.)
        return (mu, Cpre, k21, k4v, k22, add if rem is None else rem + add)
    cd.hooks['pair_in'] = pin_hook
    return cd
