"""TAG fable-covhunter (CH-71): TRAINABLE version of s6bcard_joint2 (CH-60, native-consistent dial semantics) for the lead's JOINT refit
(learned parts + dials trained together by gradient). The 12 dials are 0-dim float64 torch tensors (make_dials), and every dial hook is
ALWAYS installed (joint2 skips hooks at base values, which would cut the gradient), with the same NATIVE-SHARED meaning:
  X1          j1 physical coefficient (replaces c1*orth(dir1));   X0 X2 X3  physical j0/j2/j3 directions ADDED to c_j*orth(dir_j)
  bscale      carried b x bscale every layer (after the b-fix)     s_carry / s_new  lam of carried columns every layer / newborn at birth
  c0 c1       K31 row factor c0 + c1 mu k3 / v^2, SHARED by the pair map (env _ROWF) and the GN r31 remainder (gn3 re-configured with
              the tensors; gn_linear_hook installed after the card reads the same _ROWF = native _K31R_TABLE)
  k4a/b/c     per-unit k4 x band at its DEFINITION (pair call AND ctx['k4'] for the GN term = native k4 at definition; L1-7/8-11/12-14,
              L15 k4 via k4_15 with k4c)
Export: values(D) -> the flat FIT dict (floats) that s6bcard_joint2 / the porter's --gn=shared build read as literals.
usage: D = make_dials(torch, json_or_dict, trainable=KEYS); cd = card(torch, w, p, a, c, e, dials=D, fine_ckpt=...), or, for a refit
driver that cannot pass dials (LOOP_DIALMOD wrappers): set the module global DIALS_T = make_dials(...) once and add its trainable tensors
to the optimizer; card() then uses DIALS_T. Build with fine_ckpt=False when join_nystrom_torch is installed (it needs the plain push).
Certified 09-25 on net 37 (S23 structure, loopR1 params, round-1 dials): forward bit-exact with s6bcard_joint2; backward finite for all
12 dials (fine_ckpt=False, peak RSS 8.4 GB)."""
import s6bcard as S6
import s2card
import gn3_knobs as gn3
import s6bcard_joint2 as J2
KEYS = ('X1', 'X0', 'X2', 'X3', 'bscale', 's_carry', 's_new', 'c0', 'c1', 'k4a', 'k4b', 'k4c')
BASE = J2.BASE
DIALS_T = None
STAR = None   # LOOP_DIALMOD wrappers set this; the star is installed by the wrapper
model, weights, truthpost, payload, PARAMS = S6.model, S6.weights, S6.truthpost, S6.payload, S6.PARAMS


def make_dials(torch, values=None, trainable=KEYS):
    v = dict(BASE if values is None else values)
    return {k: torch.tensor(float(v[k]), dtype=torch.float64, requires_grad=k in trainable) for k in KEYS}


def values(D):
    return {k: float(D[k].detach()) for k in KEYS}


def make_bd(torch, D):   # joint2.make_bd with the physical adds ALWAYS on (gradient to X0/X2/X3 at any value)
    add = {0: D['X0'], 2: D['X2'], 3: D['X3']}
    def diag0(x): return torch.diagonal_scatter(x, torch.zeros(x.shape[0], dtype=x.dtype))
    def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
        n = int(bp.shape[0]); c = coefficients.to(torch.float64)
        alpha, sigma, g = aux['alpha'], aux['sigma'], aux['gate']; eta = alpha / sigma
        half_d = 0.5 * d; negative_half_g = -0.5 * g; negative_half_d = -half_d
        skew = aux['pre_k3'] / (sigma * sigma * sigma); gate_jet = skew * (alpha * alpha - 1) * aux['pdf'] / 6
        denominator = torch.clamp_min(torch.sum(bp * bp, dim=0), 1e-20)
        change = torch.zeros((n, n), dtype=torch.float64)
        for j in range(4):
            if j == 0: direction = half_d[:, None] * K21
            elif j == 1: direction = (negative_half_g[:, None] * K21.T) * eta[None, :]
            elif j == 2: direction = (negative_half_d[:, None] * (Cpre * Cpre)) * eta[None, :]
            else: direction = Cpre * gate_jet[:, None]
            direction = diag0(direction)
            if j == 1:
                change = change + direction * D['X1']
            else:
                raw = direction
                projection = torch.sum(bp * direction, dim=0) / denominator
                direction = direction - bp * projection[None, :]
                change = change + direction * c[j] + raw * add[j]
        change = diag0(change)
        old32 = (bp * alive[:, None]).to(torch.float32); new32 = ((bp + change) * alive[:, None]).to(torch.float32)
        old = old32.to(torch.float64); new = new32.to(torch.float64); difference = new - old
        a64 = a.to(torch.float32).to(torch.float64)
        t64 = (t * alive).to(torch.float32).to(torch.float64); s64 = (s * alive).to(torch.float32).to(torch.float64)
        weight = (a64 * (t64 - s64)) * births; quadratic_weight = (a64 * (t64 * t64 - s64 * s64)) * births
        delta = ((2 * old * difference + difference * difference) * weight[None, :] + quadratic_weight[:, None] * difference.T)
        return new32, diag0(delta)
    return _bd_update


def card(torch, w, p, a, c, e, dials=None, fine_ckpt=False, **kw):
    D = dials if dials is not None else (DIALS_T if DIALS_T is not None else make_dials(torch, trainable=()))
    cd = S6.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)
    gn3.configure(cd, dict(gn3.V3, c0=D['c0'] - 0.85, c1=D['c1'] - 0.5))   # pair-map row factor + GN r31 share c0/c1 (tensors)
    cd.env['_bd_update'] = make_bd(torch, D)
    cd.hooks['b_post'] = lambda l, b, ctx: b * D['bscale']
    band = lambda l: D['k4a'] if l <= 7 else (D['k4b'] if l <= 11 else D['k4c'])
    def pair_in(l, pin, ctx):
        if l == 0: return pin
        ctx['k4'] = ctx['k4'] * band(l)
        return (pin[0], pin[1], pin[2], pin[3] * band(l), pin[4], pin[5])
    cd.hooks['pair_in'] = pair_in
    cd.hooks['k4_15'] = lambda l, k4, ctx: k4 * D['k4c']
    def state_out(l, st, ctx):
        if l == 15: return st
        lam, stops = st[3], st[12]
        n_old = stops[-2] if len(stops) >= 2 else 0
        return st[:3] + (torch.cat((lam[:n_old] * D['s_carry'], lam[n_old:] * D['s_new'])),) + st[4:]
    cd.hooks['state_out'] = state_out
    if fine_ckpt:
        for k in ('_nonlinear_pair_batch', '_bd_update', '_flms_star'): cd.env[k] = s2card._ckpt(torch, cd.env[k])
        for k in ('push', 'read'): cd.shared[k] = s2card._ckpt(torch, cd.shared[k])
    return cd
