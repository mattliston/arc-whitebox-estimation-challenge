"""TAG fable-covhunter (CH-60) = s6bcard_joint.py (CH-52) with NATIVE-CONSISTENT dial semantics (porter's catch, lead's option (a)):
  k4 bands scale k4 AT ITS DEFINITION: the main pair call AND the GN v3 cov_pm recompute (ctx['k4']) both see the scaled k4
  (natively k4 x band reaches pair_map and _gn_dcov); c0/c1 are SHARED by the pair map row factor and the GN r31 remainder (gn3 is
  re-configured with the dialed c0/c1, as natively _gn_dcov reads the same _K31R_TABLE). All other dials as in CH-52; at BASE the
  card is the j05 card bit-exactly. s6bcard_joint.py (CH-52) stays as the reference for the §25/§26 fits. Original header follows:
TAG fable-covhunter (CH-52): the j05 card (the physicist's s6bcard_j05 pattern: s6bcard + make_bd 'phys' j1) with the lead's L1 physics
DIALS for a joint learned-OFF Gauss-Newton fit. At BASE the card is the j05 card bit-exactly. DIALS (module dict, read at card() time):
  X1          j1 physical coefficient, replacing c1*orth(dir1) as in variant_bd 'phys' (base 0.5)
  X0 X2 X3    the physical (unorthogonalized) j0/j2/j3 directions ADDED to the learned c_j*orth(dir_j) (base 0)
  bscale      carried b x bscale at every layer (b_post hook, as the physicist's BSCALE; base 1)
  s_carry     CP weights (lam) of the carried columns x s_carry at every layer (state_out hook; base 1)
  s_new       CP weights of the newborn columns x s_new at birth (state_out hook; base 1)
  c0 c1       K31 row factor c0 + c1 mu k3 / v^2 (covstep_s2.set_rowf; base .85 / .5)
  k4a k4b k4c per-unit k4 multipliers on pair layers 1-7 / 8-11 / 12-15 (pair_in hook; k4_15 at L15; base 1)
Drop-in card module (payload/model/card)."""
import math
import s6bcard as S6
import s2card
import covstep_s2
import gn3_knobs as gn3
BASE = dict(X1=0.5, X0=0.0, X2=0.0, X3=0.0, bscale=1.0, s_carry=1.0, s_new=1.0, c0=0.85, c1=0.5, k4a=1.0, k4b=1.0, k4c=1.0)
DIALS = dict(BASE)
model, weights, truthpost, payload, PARAMS = S6.model, S6.weights, S6.truthpost, S6.payload, S6.PARAMS


def make_bd(torch, D):   # variant_bd.make_bd('phys', X1) + physical j0/j2/j3 adds (skipped at 0, so bit-exact at BASE)
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
                change = change + direction * c[j]
                if add[j] != 0: change = change + raw * add[j]
        change = diag0(change)
        old32 = (bp * alive[:, None]).to(torch.float32); new32 = ((bp + change) * alive[:, None]).to(torch.float32)
        old = old32.to(torch.float64); new = new32.to(torch.float64); difference = new - old
        a64 = a.to(torch.float32).to(torch.float64)
        t64 = (t * alive).to(torch.float32).to(torch.float64); s64 = (s * alive).to(torch.float32).to(torch.float64)
        weight = (a64 * (t64 - s64)) * births; quadratic_weight = (a64 * (t64 * t64 - s64 * s64)) * births
        delta = ((2 * old * difference + difference * difference) * weight[None, :] + quadratic_weight[:, None] * difference.T)
        return new32, diag0(delta)
    return _bd_update


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    D = dict(DIALS)
    cd = S6.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)
    if (D['c0'], D['c1']) != (BASE['c0'], BASE['c1']):   # pair-map row factor AND the GN r31 remainder share c0/c1 (native _K31R_TABLE)
        gn3.configure(cd, dict(gn3.V3, c0=D['c0'] - 0.85, c1=D['c1'] - 0.5))
    cd.env['_bd_update'] = make_bd(torch, D)
    if D['bscale'] != 1.0: cd.hooks['b_post'] = lambda l, b, ctx, s=D['bscale']: b * s
    band = lambda l: 1.0 if l == 0 else (D['k4a'] if l <= 7 else (D['k4b'] if l <= 11 else D['k4c']))
    if (D['k4a'], D['k4b'], D['k4c']) != (1.0, 1.0, 1.0):
        def pair_in(l, pin, ctx):   # k4 x band at its definition: the pair call and the GN cov_pm recompute (ctx['k4']) both see it
            ctx['k4'] = ctx['k4'] * band(l)
            return (pin[0], pin[1], pin[2], pin[3] * band(l), pin[4], pin[5])
        cd.hooks['pair_in'] = pair_in
        cd.hooks['k4_15'] = lambda l, k4, ctx: k4 * band(l)
    if (D['s_carry'], D['s_new']) != (1.0, 1.0):
        def state_out(l, st, ctx):
            if l == 15: return st
            lam, stops = st[3], st[12]
            n_old = stops[-2] if len(stops) >= 2 else 0
            lam = torch.cat((lam[:n_old] * D['s_carry'], lam[n_old:] * D['s_new']))
            return st[:3] + (lam,) + st[4:]
        cd.hooks['state_out'] = state_out
    if fine_ckpt:   # as s6bcard.card, applied last so every replaced function is wrapped
        for k in ('_nonlinear_pair_batch', '_bd_update', '_flms_star'): cd.env[k] = s2card._ckpt(torch, cd.env[k])
        for k in ('push', 'read'): cd.shared[k] = s2card._ckpt(torch, cd.shared[k])
    return cd
