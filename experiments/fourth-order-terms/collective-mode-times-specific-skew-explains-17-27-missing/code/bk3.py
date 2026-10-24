"""TAG opus-physicist BK3 (2026-10-02): THE CROSS FORM OF THE COLLECTIVE MODE AND THE SPECIFIC THIRD CUMULANT IN THE (3,1) PLANE, as a carrier module with the interface of
O/ncar6.py / O/k31l.py (install(cd, torch, st, **arguments); st is not used: everything comes from the step's own context).

Physics. The pre-activations carry a collective fluctuation S (the layer's overall scale): the covariance scales with it and the means shift with it, which is the card's b-form of the
third cumulant, K21_ij contains v_i b_j + 2 C_ij b_i (b = the pushed vector bpre). The SPECIFIC third cumulant (bank + star) scales with S too, and that gives the fourth cumulant
    kappa4(a, b, c, d)  contains  kappa Sym4( b (x) kappa3_specific ),      in the (3,1) plane:   K31_ij  contains  x1 k3s_i b_j + x2 b_i Ks_ij,   x2 = 3 x1,
Ks = the card's bank + star planes (ctx['cp'] + ctx['sp']), k3s = its diagonal. A common multiplicative fluctuation gives kappa = 3/2 (x1 1.5, x2 4.5).
Measured against the 8M truth planes of networks 35 / 100 (physicist/t33c_bk3_split.py, RESULT.md T28): on the missing part of the K31 plane (truth minus card, weighted as the
covariance uses it) the least-squares coefficients are x1 = 3.1-3.4, x2 = 9.6-11 at layers 8-14 on both networks (ratio 3), and the form explains 17-27% of the missing energy;
the form of second order in b (v_i b_i b_j) has coefficient about 0: only the specific part belongs here (default spec=True).
In the loop with the learned parts ON it does not pay on average (eight networks, t4_ora53.log); it is a candidate for the refit's physics base.

Equivalent reading (why it is free natively): x2 b_i Ks_ij in the K31 slot is, to first order, the K21 term with the row unit's kink density taken at the shifted mean
mu_i + (x2 / 3) b_i; x1 k3s_i b_j is the response of unit i's mean to its skew times the mean shift of unit j.

install(cd, torch, st=None, k31=1.0, x1=3.2, x2=10.0, lo=1, hi=14, level='r31', spec=True)
   level 'r31'   remainder31 += k31 (x1 k3s_i b_j + x2 b_i Ks_ij) / (v_i sigma_i sigma_j), off the diagonal   (research form; what the measurements used)
   level 'pair'  the fold: the K21 plane given to the pair map loses (mu_i / (3 v_i)) x the same form, off the diagonal (native form: one row factor on the bank and star planes and
                 one outer product; like every fold it also reaches the pair map's third-cumulant target)
   spec False    the whole K21 plane and the whole k3 in place of the specific parts (includes the terms of second order in b; loses, t4_ora51.log)
cd.bk3_info[layer] = rms of the added standardised plane."""


def install(cd, torch, st=None, k31=1.0, x1=3.2, x2=10.0, lo=1, hi=14, level='r31', spec=True):
    t = torch; f64 = t.float64; info = {}; cd.bk3_info = info
    old_pin = cd.hooks.get('pair_in')

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if not (lo <= l <= hi) or not k31 or not t.is_tensor(pin[2]) or pin[2].ndim != 2: return pin
        mu, Cpre, k21, k4v, k22, rem = pin
        n = k21.shape[0]; eye = t.eye(n, dtype=t.bool); v = t.diagonal(Cpre).detach().to(f64).clamp_min(1e-12); s = v.sqrt()
        b = ctx['bpre'].detach().to(f64); K = ((ctx['cp'] + ctx['sp']) if spec else k21).detach().to(f64); k3 = t.diagonal(K)
        P = (k31 * (x1 * t.outer(k3, b) + x2 * b[:, None] * K)).masked_fill(eye, 0.)
        info[l] = float(((P / (v[:, None] * s[:, None] * s[None, :])) ** 2).mean().sqrt())
        if level == 'pair':
            return (mu, Cpre, k21 - ((mu.detach().to(f64) / (3 * v))[:, None] * P).to(k21.dtype), k4v, k22, rem)
        add = P / (v[:, None] * s[:, None] * s[None, :])
        return (mu, Cpre, k21, k4v, k22, add if rem is None else rem + add)
    cd.hooks['pair_in'] = pin_hook
    return cd
