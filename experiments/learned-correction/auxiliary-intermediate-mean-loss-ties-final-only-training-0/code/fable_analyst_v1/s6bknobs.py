"""TAG fable-analyst (AN-47): the S6b torch card (s6bcard.py) with TRAINABLE covariance knobs: the v3 knob strengths iso, k21n, k31,
edge, rc as one scalar per layer band (L2-5, L6-9, L10-12, L13-14: 20 scalars) plus the K31 row constants c0, c1 (the S2 row factor
T31 row r = c0 + c1 mu k3 / v^2 in the pair map at layers 1-14, also used by the k31 knob). Each band scalar replaces the shipped strength
at the layers of its band (cov knobs act at L2-14, edge at L8-14 so edge@L2-5 is structurally inert, recentering at L1-14 so L1 keeps
the shipped rc); layers outside the bands keep the shipped values. The formulas and their operation order are copied from
fable_covhunter_v1/gn3_knobs.configure (cov_pm hook re-running the pair map with the modified inputs, edge x (1 + t), recentering
alpha = 1 + t), so the card at init is bit-identical to s6bcard.card (checked by smoke_knobs.py).
init(torch) -> dict of float64 leaf tensors; card(torch, w, p, a, c, e, K, fine_ckpt=False) -> the configured HookCard; values(K) -> floats."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import s6bcard
cs, gn3 = s6bcard.cs, s6bcard.gn3
import covstep_s2
BANDS = {'L2-5': range(2, 6), 'L6-9': range(6, 10), 'L10-12': range(10, 13), 'L13-14': range(13, 15)}
KNOBS = ('iso', 'k21n', 'k31', 'edge', 'rc')
def band(l):
    for b, r in BANDS.items():
        if l in r: return b
    return None
def init(torch):
    K = {f'{k}@{b}': torch.tensor(gn3.V3[k], dtype=torch.float64) for k in KNOBS for b in BANDS}
    K['c0'] = torch.tensor(0.85, dtype=torch.float64); K['c1'] = torch.tensor(0.5, dtype=torch.float64)
    return K
def values(K): return {k: float(v.detach()) for k, v in K.items()}
def card(torch, w, p, a, c, e, K, fine_ckpt=False, zc=s6bcard.ZC):
    cd = cs.HookCard(torch, w, p, a, c, e)
    covstep_s2.make_s2(cd, rowf=(0.85, 0.5), gamma=0.1)                        # S2 pair source, every3 push, b-fix .1 (as gn3.configure)
    cd.env['_ROWF'][0] = lambda mu, k3, v: K['c0'] + K['c1'] * mu * k3 / (v * v)
    n = 1024; eye = torch.eye(n, dtype=torch.bool)
    def cov(l, C, ctx):                                                          # gn3_knobs.configure's cov_pm hook, band strengths
        if l < 2 or l > 14: return C
        b = band(l); c0, c1 = K['c0'], K['c1']
        mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
        vv = torch.diagonal(Cpre); s = vv.sqrt(); al = mu / s
        kink = (al.abs() < 1.5).double(); km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
        k2, k22n, r31 = k21, k22, None
        for k in ('iso', 'k21n', 'k31'):
            t = K[f'{k}@{b}']
            if k == 'iso': k2 = k2 + t * km * (k3[:, None] * Cpre / vv[:, None])
            elif k == 'k21n': k2 = k2 + t * nm * k21
            elif k == 'k31': r31 = t * ((k4 / vv ** 2) * (c0 + c1 * mu * k3 / (vv * vv)))[:, None] * rho
        return pair(mu, Cpre, k2, k4, k22n, 'covariance', remainder31=r31)[1]
    cd.hooks['cov_pm'] = cov
    shipped_edge = gn3.V3['edge']
    cd.hooks['edge'] = lambda l, v, ctx: v * (1 + (K[f'edge@{band(l)}'] if band(l) else shipped_edge))
    shipped_rc = 1 + gn3.V3['rc']
    def state_in(l, state, ctx):                                                 # per-layer recentering strength, read by HookCard.step
        cd.knobs['recenter'] = (1 + K[f'rc@{band(l)}']) if band(l) else shipped_rc
        return state
    cd.hooks['state_in'] = state_in
    if zc is not None: cd.hooks['k3_15'] = s6bcard.l15_rowcut(torch, zc)
    if fine_ckpt:
        for k in ('_nonlinear_pair_batch', '_bd_update', '_flms_star'): cd.env[k] = s6bcard.s2card._ckpt(torch, cd.env[k])
        for k in ('push', 'read'): cd.shared[k] = s6bcard.s2card._ckpt(torch, cd.shared[k])
    return cd
