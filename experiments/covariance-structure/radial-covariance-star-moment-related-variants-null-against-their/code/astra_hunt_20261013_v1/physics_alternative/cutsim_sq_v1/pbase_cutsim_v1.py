# v8 (+ 01:3xZ: gbg_mode 'eig' = the physicist's lambda_2 / lambda_1 gate: r_l = lambda_2 / lambda_1 of the CURRENT layer's Cpre (eigvalsh, f64),
#   z = (r_l - gbg_mu_eig[l]) / gbg_sd_eig[l], gbg_c clip(z, -1, 1); the gate now receives the step's Cpre as a third argument, ignored by the other modes)
# v8 (+ 01:3xZ: net / nets print layer, f, z to stderr = the worker log; no numerical change)
# v8 (+ 01:4xZ: gbg_mode 'nets' = the signed network gate c min(max(z, -1), 1), same z as 'net')
# v8 (O/scout_card/cutsim_sc8/, the physicist's EXACT feature-gated beta spec, 01:3xZ): only when gbg_mode != 'none' (and the gate is on,
#   gbs == 0, gbeta == 0), the birth leg's ORIGINAL beta line becomes beta = 2 (gm - host g) + self._gbg(layer, aux) (-2 host (gq - g)); at 'none'
#   the line is untouched (the control). install() stores gbg_mode / gbg_c / gbg_a0 / gbg_lo / gbg_hi / gbg_mu / gbg_sd in cd.pbase and sets
#   cd._gbg = the physicist's function verbatim (reads cd.pbase at call time).
# v7 (O/scout_card/cutsim_sc7/, physicist 23:2xZ): gbs gets a layer band, gbs_lo <= layer <= gbs_hi (defaults 0 / 14 = all birth layers = cutsim_sc6):
#   beta = 2 (gm - host g) + gbs [gbs_lo <= layer <= gbs_hi] (2 host (1 - gq) - 2 (gm - host g)).
# v6 (O/scout_card/cutsim_sc6/): + keyword gbs = x (physicist 22:5xZ, a two-sided strength test of the birth leg's beta):
#   beta = 2 (gaussian_mean - host g) + x (2 host (1 - gq) - 2 (gaussian_mean - host g)); x = 1 is gbeta = 1. Needs the gate on. Default 0 = cutsim_sc5.
# v5 (O/scout_card/cutsim_sc5/, with the physicist's certified law file n6y_cutsim_v2_lq.py as n6y_cutsim_v2.py): + keyword bk4l = x, the birth
#   density's kurtosis term with the LOCAL kurtosis only (physicist 22:1xZ): birth = d (1 + kappa radial / 3)(1 - bsk (T3/6) He3 + x (T4loc / 24) He4),
#   T4loc = (aux['pre_k4'] - radial vv^2) / vv^2 (the (1 + radial / 3) factor already carries the radial part). Default 0 = cutsim_sc4.
# v4 (O/scout_card/cutsim_sc4/): + keyword gbu (physicist's v2 #4): the birth-leg update _bd_update gets the corrected gate and density,
#   aux gate -> g + gbu (gq - g), d -> d + gbu (dq - d) (dq = the Edgeworth kink density, clamped at 0); needs the gate on. Default 0 = cutsim_sc3.
# SCOUT-CARD PRIVATE COPY v3 (O/scout_card/cutsim_sc3/, 2026-10-12, radlab's 22:0xZ budget: put fixes at L12-15): = cutsim_sc2's pbase + per-layer
#   gkL  = 16 multipliers of the gate's skew AND kurtosis terms by layer (gq = g + gkL[l] (gsk (T3/6) He2 phi - gk4 (T4/24) He3 phi) + gbk d bpre)
#   bskL = 16 per-layer values replacing bsk in the birth line (the birth density's skew term by layer)
#   both None (default) = cutsim_sc2.
# SCOUT-CARD PRIVATE COPY v2 (O/scout_card/cutsim_sc2/, 2026-10-12, physicist's 21:5xZ consistency sites): = cutsim_sc1's pbase + keywords
#   gbd   = 1 / 2: the b vector's radial source with the Edgeworth kink density dq (1: skew term only, 2: skew + kurtosis): b = gq bpre + bsrc radial dq vv / 6
#   gbeta = 1: beta = 2 host (1 - gq) (as in sc1);  2: the half form beta = 2 (gaussian_mean - host gq)
#   gomit = 1: the omitted-radial source with the corrected gate / density: aa = (gq - host dq) omitted, beta = 2 host (1 - gq) alive
#   bk4   = x: the birth density's kurtosis term, birth = d (1 + kappa radial / 3)(1 - bsk (T3 / 6) He3 + x (T4 / 24) He4)
#   dq = d (1 - (T3 / 6) He3 + (T4 / 24) He4) clamped at 0; gq as gk (unclamped, the shipped gk). All 0 = cutsim_sc1 = the engineer's file.
# SCOUT-CARD PRIVATE COPY (O/scout_card/cutsim_sc1/, 2026-10-12) of engineer/a22_cutsim/pbase_cutsim_v1.py (sha 5af2eb00) with two keywords added, both
# acting only when the transport gate gq is on (gsk / gk4 / gbk): gbp = 1 -> the birth plane uses gq (bp = gq o Cpre, the newborn pairs' coupling
# vector; the law reads the same columns); gbeta = 1 -> the birth leg (a, a, i) uses beta = 2 host (1 - gq) (E[h_a 1_a] = E[h_a] = host, P(z_a > 0) = gq)
# in place of 2 (gaussian_mean - host g). Both 0 (default) = the engineer's file bit for bit in arithmetic.
"""TAG opus-physicist PBASE (2026-10-02): THE PHYSICS BASE FOR THE REFIT. Terms that are right against the truth and that the present learned parts were trained without,
ONE carrier module with ONE KEYWORD PER TERM (lead 18:01Z), interface of O/ncar6.py / O/k31l.py: install(cd, torch, st, **keywords). st (the pool card's side table) is not used:
every term is elementwise on arrays the step already holds (no matrix product, no state of its own), so a step that runs again in a gradient run recomputes it and the terms that
sit inside the step and the pair map are differentiable. All keywords at 0 (the default) leave the card bit for bit as it is.

  so2m   second-order mean term INSIDE the pair map, layers lo..hi: mean += so2m t3^2 sigma He4(alpha) pdf / 72, with the consistent E relu^2 term (the per-unit post variance follows)
  so2c   second-order covariance list (the products of two third cumulants a b q c d e of so2.py, up to the first power of rho), layers lo..hi
  so2t   second-order part of the third-cumulant target (the diagonal post k3 and the plane cum(f_i, f_i, f_j)), layers lo..hi
  so2k   highest power of rho kept in the two lists (0, 1 or 2; default 1);  so2p  digits 1-6 = which of the parts a b q c d e of the lists (default 123456)
  so2td  1 = so2t changes only the DIAGONAL of the target (the per-unit post k3), not the plane (no list is formed: free natively)
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

ARGS = dict(so2m=0.0, so2c=0.0, so2t=0.0, so2f=0.0, so2k=1, so2p=123456, so2td=0, bsk=0.0, gsk=0.0, gk4=0.0, gbk=0.0, x1=0.0, x2=0.0, lo=1, hi=14, gbp=0, gbeta=0, gbd=0, gomit=0, bk4=0.0, gkL=None, bskL=None, gbu=0.0, bk4l=0.0, gbs=0.0, gbs_lo=0, gbs_hi=14, gbg_mode='none', gbg_c=1.0, gbg_a0=0.75, gbg_lo=8, gbg_hi=14, gbg_mu=None, gbg_sd=None, gbg_mu_eig=None, gbg_sd_eig=None, sq=0.0, sq_lo=0, sq_hi=14)      # the install arguments besides k31 (for a module table)


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


def _so2(cd, t, mean, cov, tgt, final, lo, hi, kmax=1, parts='abqcde', tdiag=False):
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
        if tgt and tdiag and not cov:      # the per-unit post k3 alone
            m1 = m + m2; r2f = r2 + q2; pk3 = (r3 + q3) - 3 * m1 * r2f + 2 * m1 ** 3; pk3_0 = r3 - 3 * m * r2 + 2 * m ** 3
            p21n = post21 + tgt * t.diag_embed(pk3 - pk3_0); aux['post_k3'] = t.diagonal(p21n)
        elif cov or tgt:
            rho = offd(C / t.outer(s, s)); T = offd(K21 / (v[:, None] * s[None, :])); off = offd(covm)
            A = _A(t, a, ph, g, mg / s, kmax + 7)
            dc1 = offd(pair_list(t, A, A, rho, T, t3, kmax, parts) * t.outer(s, s))
            if cov: covn = covm + cov * dc1
            if tgt:
                B = [(a * a + 1) * g + a * ph] + [2 * x for x in A[:-1]]
                dp1 = pair_list(t, A, B, rho, T, t3, kmax, parts) * t.outer(v, s)
                m1 = m + m2; dt = offd(dp1 - 2 * (m1[:, None] * (off + dc1) - m[:, None] * off))
                r2f = r2 + q2; pk3 = (r3 + q3) - 3 * m1 * r2f + 2 * m1 ** 3; pk3_0 = r3 - 3 * m * r2 + 2 * m ** 3
                p21n = post21 + tgt * (t.diag_embed(pk3 - pk3_0) if tdiag else t.diagonal_scatter(dt, pk3 - pk3_0))
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


def _step(cd, bsk, gate, gbp=False, gbeta=0, gbd=0, gomit=False, bk4=0.0, gkl=False, bskl=False, gbu=0.0, bk4l=0.0, gbs=0.0, gbg=False):
    """the card's step with the birth line and / or the transport gate changed (the source of the class's step, re-executed)."""
    owner = next(k for k in type(cd).__mro__ if 'step' in k.__dict__); src = textwrap.dedent(inspect.getsource(owner.step))
    if hasattr(cd, '_cutsim_class_source'):
        src = cd._cutsim_class_source   # private clone: rowcut hooks before the original PL/pbase patches
    pl = hasattr(cd, 'pl_kappa')      # the trainer's per-layer card: its step = the class's with the two lines below replaced (otr_pl._patched_step); they are kept
    now = getattr(cd.step, '__func__', None)
    assert now is owner.__dict__['step'] or (pl and 'per-layer kappa' in now.__code__.co_filename), 'the step is already replaced by something this module does not know'
    def edit(old, new):
        nonlocal src
        assert src.count(old) == 1, (src.count(old), old); src = src.replace(old, new)
    T3 = "(aux['pre_k3'] / aux['sigma'] ** 3)"; AL = "aux['alpha']"
    birth = 'birth = d * (1 + self.pl_kappa(layer) * radial / 3)' if pl else 'birth = d * (1 + radial / 3)'
    bline = 'b = %s * bpre + self.pl_bsrc(layer) * radial * d * vv / 6' if pl else 'b = %s * bpre + radial * d * vv / 6'
    K4B = (f" + {bk4!r} * (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 4 - 6 * {AL} ** 2 + 3) / 24" if bk4 else '')      # scout-card sc2: bk4
    K4B = K4B + (f" + {bk4l!r} * ((aux['pre_k4'] - radial * vv * vv) / (vv * vv)) * ({AL} ** 4 - 6 * {AL} ** 2 + 3) / 24" if bk4l else '')      # scout-card sc5: bk4l (local kurtosis)
    BSK = "self.pbase['bskL'][layer]" if bskl else "self.pbase['bsk']"      # scout-card sc3
    edit('birth = d * (1 + radial / 3)\n', birth + (f" * (1 - {BSK} * {T3} * ({AL} ** 3 - 3 * {AL}) / 6{K4B})" if bsk else '') + '\n')
    if gate:
        edit("    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n",          # the dedented source: method body at four spaces
             "    g = aux['gate']\n    d = aux['pdf'] / aux['sigma']\n"
             + (f"    gq = g + self.pbase['gkL'][layer] * (self.pbase['gsk'] * {T3} * ({AL} ** 2 - 1) * aux['pdf'] / 6 - self.pbase['gk4'] * (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 3 - 3 * {AL}) * aux['pdf'] / 24)" if gkl else
                f"    gq = g + self.pbase['gsk'] * {T3} * ({AL} ** 2 - 1) * aux['pdf'] / 6 - self.pbase['gk4'] * (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 3 - 3 * {AL}) * aux['pdf'] / 24")
             + " + self.pbase['gbk'] * d * bpre\n"
             + (f"    dqb = (d * (1 - {T3} * ({AL} ** 3 - 3 * {AL}) / 6" + (f" + (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 4 - 6 * {AL} ** 2 + 3) / 24" if gbd == 2 else '') + ")).clamp_min(0)\n" if gbd else ''))      # scout-card sc2: gbd
        edit('inherited = g[:, None] ** 2 * g[None, :] * cp', 'inherited = gq[:, None] ** 2 * gq[None, :] * cp')
        if gbu:      # scout-card sc4: the birth-leg update with the corrected gate and density
            edit("stored, increment = f['_bd_update'](bp, U_('bd'), k21, aux, d, alive, births, coefficient, plus, minus, c)",
                 f"stored, increment = f['_bd_update'](bp, U_('bd'), k21, dict(aux, gate=g + {gbu!r} * (gq - g)), d + {gbu!r} * ((d * (1 - {T3} * ({AL} ** 3 - 3 * {AL}) / 6 + (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 4 - 6 * {AL} ** 2 + 3) / 24)).clamp_min(0) - d), alive, births, coefficient, plus, minus, c)")
        edit('p = t.cat((alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * pp),', 'p = t.cat((alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * pp),')
        edit('Q = alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * U)', 'Q = alive.to(t.float32)[:, None] * (gq.to(t.float32)[:, None] * U)')
        if gbp:      # scout-card: the newborn pairs' coupling vector with the corrected gate
            edit("bp = self.diagonal(g[:, None] * U_('births'), 0.)", "bp = self.diagonal(gq[:, None] * U_('births'), 0.)")
        if gbeta:    # scout-card: the birth leg (a, a, i); 1 = 2 E[h_a] (1 - P(z_a > 0)) with the corrected gate, 2 = the half form (sc2)
            edit("beta = 2 * (aux['gaussian_mean'] - host * g)", "beta = 2 * host * (1 - gq)" if gbeta == 1 else "beta = 2 * (aux['gaussian_mean'] - host * gq)")
        if gbg:      # scout-card sc8: the physicist's feature-gated beta (only when gbg_mode != 'none')
            assert not gbs and not gbeta, 'gbg, gbs and gbeta edit the same line'
            edit("beta = 2 * (aux['gaussian_mean'] - host * g)", "beta = 2 * (aux['gaussian_mean'] - host * g) + self._gbg(layer, aux, Cpre) * (-2 * host * (gq - g))")
        if gbs:      # scout-card sc6: two-sided beta strength (physicist 22:5xZ)
            assert not gbeta, 'gbs and gbeta edit the same line'
            edit("beta = 2 * (aux['gaussian_mean'] - host * g)", f"beta = 2 * (aux['gaussian_mean'] - host * g) + {gbs!r} * float(self.pbase['gbs_lo'] <= layer <= self.pbase['gbs_hi']) * (2 * host * (1 - gq) - 2 * (aux['gaussian_mean'] - host * g))")      # scout-card sc7: band
        if gomit:    # scout-card sc2: the omitted-radial source with the corrected gate and density (gq / dq recomputed there: it precedes the gate block)
            _q = (f"(aux['gate'] + self.pbase['gsk'] * {T3} * ({AL} ** 2 - 1) * aux['pdf'] / 6 - self.pbase['gk4'] * (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 3 - 3 * {AL}) * aux['pdf'] / 24 + self.pbase['gbk'] * (aux['pdf'] / aux['sigma']) * bpre)")
            _dq = (f"((aux['pdf'] / aux['sigma']) * (1 - {T3} * ({AL} ** 3 - 3 * {AL}) / 6 + (aux['pre_k4'] / aux['sigma'] ** 4) * ({AL} ** 4 - 6 * {AL} ** 2 + 3) / 24)).clamp_min(0)")
            edit("aa = (aux['gate'] - host * aux['pdf'] / aux['sigma']) * omitted", f"aa = ({_q} - host * {_dq}) * omitted")
            edit("beta = 2 * (aux['gaussian_mean'] - host * aux['gate']) * alive", f"beta = 2 * host * (1 - {_q}) * alive")
    edit('b = g * bpre + radial * d * vv / 6\n', (bline % ('gq' if gate else 'g')).replace(' * d * vv / 6', ' * dqb * vv / 6' if (gate and gbd) else ' * d * vv / 6') + '\n')      # scout-card sc2: gbd
    space = dict(vars(sys.modules[(type(cd) if pl else owner).__module__]))      # the namespace the trainer's replacement uses (the card class's module) / the screen harness's (the step's module)
    exec(compile(src,'<%s.step with the physics base (pbase)%s>' % (owner.__name__, ', per-layer kappa and b source' if pl else ''), 'exec'), space)
    cd.step = types.MethodType(space['step'], cd)
    if hasattr(cd, '_cutsim_class_source'):
        cd._cutsim_step_applied = True


def _sq_install(cd, t, st, strength, lo, hi):
    """Forward-only Sq: non-Gaussian newborn mixed fourth-cumulant responses.

    Update only the S1, S2, S3 coefficients used by K4L and K31L. The pure
    marginal S4 is deliberately retained: replacing it by Edgeworth k4(h)
    would count incoming k4 as a new birth even when ReLU is the identity.
    The carrier dispatcher calls this after its main state_out has created
    the current cohort. The attached state's birth dictionary is updated as
    well as the detached side table. This probe is for mode=eval only.
    """
    assert st is not None and 'births' in st
    previous = cd.hooks.get('state_out')

    def state_out(layer, out, ctx):
        if previous is not None:
            out = previous(layer, out, ctx)
        if not lo <= layer <= hi or layer > 14:
            return out
        aux = ctx['aux']
        birth = st['births'][layer]
        sel = birth['sel']
        read = lambda x: x.detach().to(t.float64)[sel]
        a, s, g, ph = (read(aux[k]) for k in ('alpha', 'sigma', 'gate', 'pdf'))
        m = read(ctx['host'])
        v = s * s
        t3 = read(aux['pre_k3']) / (v * s)
        t4 = read(aux['pre_k4']) / (v * v)
        a2 = a * a
        h2 = a2 - 1
        h3 = a * (a2 - 3)
        h4 = a2 * (a2 - 6) + 3
        h5 = a * (a2 * (a2 - 10) + 15)
        dg = ph * (t3 * h2 / 6 - t4 * h3 / 24)
        dd = (ph / s) * (-t3 * h3 / 6 + t4 * h4 / 24)
        dr2 = v * ph * (t3 / 3 - t4 * a / 12)
        r2 = v * ((1 + a2) * g + a * ph)
        ds1 = (ph / v) * (t3 * h4 / 6 - t4 * h5 / 24)
        ds2 = 2 * ((1 - 2 * g) * dg - dg * dg - m * dd)
        ds3 = 3 * ((1 - g) * dr2 - dg * (r2 - 2 * m * m + dr2))
        increments = (ds1, ds2, ds3)

        def corrected(old):
            return tuple(old[j] + strength * increments[j] for j in range(3)) + (old[3],)

        birth['S'] = corrected(birth['S'])
        if len(out) == 15:
            attached = out[14]['births'][layer]
            # The detached and attached entries are distinct when BIRTH_ATTACH
            # is enabled. Guard against applying twice if a future card aliases.
            if attached is not birth:
                attached['S'] = corrected(attached['S'])
        return out

    cd.hooks['state_out'] = state_out


def install(cd, torch, st=None, k31=1.0, so2m=0.0, so2c=0.0, so2t=0.0, so2f=0.0, so2k=1, so2p=123456, so2td=0, bsk=0.0, gsk=0.0, gk4=0.0, gbk=0.0, x1=0.0, x2=0.0, lo=1, hi=14, gbp=0, gbeta=0, gbd=0, gomit=0, bk4=0.0, gkL=None, bskL=None, gbu=0.0, bk4l=0.0, gbs=0.0, gbs_lo=0, gbs_hi=14, gbg_mode='none', gbg_c=1.0, gbg_a0=0.75, gbg_lo=8, gbg_hi=14, gbg_mu=None, gbg_sd=None, gbg_mu_eig=None, gbg_sd_eig=None, sq=0.0, sq_lo=0, sq_hi=14):
    t = torch; f64 = t.float64; lo, hi = int(lo), int(hi)
    cd.pbase = dict(gbg_mode=str(gbg_mode), gbg_c=float(gbg_c), gbg_a0=float(gbg_a0), gbg_lo=int(gbg_lo), gbg_hi=int(gbg_hi), gbg_mu=gbg_mu, gbg_sd=gbg_sd, gbg_mu_eig=gbg_mu_eig, gbg_sd_eig=gbg_sd_eig,
                    gbs_lo=int(gbs_lo), gbs_hi=int(gbs_hi), bsk=bsk, gsk=gsk, gk4=gk4, gbk=gbk, gkL=[float(x) for x in gkL] if gkL is not None else None, bskL=[float(x) for x in bskL] if bskL is not None else None)
    if so2m or so2c or so2t or so2f: _so2(cd, t, so2m, so2c, so2t, so2f, lo, hi, kmax=int(so2k), parts=''.join('abqcde'[int(ch) - 1] for ch in str(int(so2p))), tdiag=bool(so2td))
    assert str(gbg_mode) in ('none', 'net', 'nets', 'core', 'shoulder', 'eig'), gbg_mode
    assert str(gbg_mode) == 'none' or float(gbs) == 0.0, 'gbg needs gbs == 0'
    assert str(gbg_mode) not in ('net', 'nets') or (gbg_mu is not None and gbg_sd is not None and len(gbg_mu) == len(gbg_sd) == 16), 'gbg net needs gbg_mu / gbg_sd (16 each)'
    assert str(gbg_mode) != 'eig' or (gbg_mu_eig is not None and gbg_sd_eig is not None and len(gbg_mu_eig) == len(gbg_sd_eig) == 16), 'gbg eig needs gbg_mu_eig / gbg_sd_eig (16 each)'

    def _gbg(layer, aux, Cpre=None):      # the physicist's function, verbatim (scout-card sc8; + 'eig' 01:3xZ)
        P = cd.pbase
        if P['gbg_mode'] == 'none' or not (P['gbg_lo'] <= layer <= P['gbg_hi']):
            return 0.0
        if P['gbg_mode'] == 'eig':      # the physicist's lambda_2 / lambda_1 gate: the current layer's pre-activation covariance, signed like 'nets'
            ev = t.linalg.eigvalsh(Cpre.detach().to(t.float64))
            r = float(ev[-2] / ev[-1])
            z = (r - P['gbg_mu_eig'][layer]) / P['gbg_sd_eig'][layer]
            print(f"gbg eig L{layer} r {r:.5f} z {z:+.3f}", file=sys.stderr, flush=True)      # scout-card: the runtime gate input, in the worker log
            return P['gbg_c'] * min(max(z, -1.0), 1.0)
        a = aux['alpha'].detach().abs()
        if P['gbg_mode'] == 'core':
            return (a < P['gbg_a0']).to(aux['alpha'].dtype)
        if P['gbg_mode'] == 'shoulder':
            return (a >= P['gbg_a0']).to(aux['alpha'].dtype)
        f = float((a < 1.5).to(t.float64).mean())          # 'net': this layer's kink fraction over ALL 1024 units
        z = (f - P['gbg_mu'][layer]) / P['gbg_sd'][layer]
        print(f"gbg {P['gbg_mode']} L{layer} f {f:.4f} z {z:+.3f}", file=sys.stderr, flush=True)      # scout-card: the runtime gate input, in the worker log
        if P['gbg_mode'] == 'nets':                          # the signed network gate (physicist 01:4xZ)
            return P['gbg_c'] * min(max(z, -1.0), 1.0)
        return P['gbg_c'] * min(max(z, 0.0), 2.0)           # rectified: never reverses gbeta2 on kink-poor networks
    cd._gbg = _gbg
    if bsk or gsk or gk4 or gbk: _step(cd, bool(bsk), bool(gsk or gk4 or gbk), bool(gbp), int(gbeta), int(gbd), bool(gomit), float(bk4), gkL is not None, bskL is not None, float(gbu), float(bk4l), float(gbs), (bool(gsk or gk4 or gbk) and str(gbg_mode) != 'none'))
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
    if sq:
        _sq_install(cd, t, st, float(sq), int(sq_lo), int(sq_hi))
    return cd
