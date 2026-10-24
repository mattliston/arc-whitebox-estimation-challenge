"""TAG opus-reexam NCAR (lead, 2026-10-02): a free-running VECTOR carrier of the network-specific norm coupling of the fourth cumulant.

Monte Carlo (E31, E32, physicist's T19): the second-chaos matrix Ntilde_ij = sum_k kappa4(z_i, z_j, z_k, z_k) of a layer is
    Ntilde = a tr C x C + b C^2 + (mean forms) + sum_m (u_m q_m' + q_m u_m') + rest,
u_m = top eigenvectors of the pre-activation covariance C, q_m = per-unit partner vectors orthogonal to those eigenvectors
(E32c: this one-sided block holds 70% of the network-specific energy), and the true network-specific matrix as an extra K31 plane
(3/n) Ntilde_ij is worth -6..-8% of final-layer error on network 35 (physicist 09:19Z). E32d: for modes 2..8 the partner vectors
follow a one-layer law to 81-97% (mode 1 = mean direction: 55-63%):
    Ntilde(l+1) = ct 2 gbar W' g Ntilde(l) g W + 2 W' (g C) S2 (C g) W + 2 W' diag(S4) W + 2 sum_r c_r pp_r pp_r',
    gamma = g - m d, gbar = mean gamma, S2 = 2 (g - g^2 - m d), c_r = lam_r (pp_r(l) . 2 m (1 - g))   (conversion from the third-order bank).
Here the law is applied to VECTORS only: q_m(l+1) = (I - P') Ntilde(l+1) u_m(l+1), P' = projector on the new top-K eigenvectors, with
Ntilde(l) y = a tr C (C y) + b C (C y) + U (Q' y) + Q (U' y). Everything is matrix-vector products except the eigenvectors (prototype: eigh;
natively a block power iteration on K columns).
Use: K31 extra plane (3/n) sum_m (u_m q_m' + q_m u_m'), through the pair map's remainder31 input (standardised as in the physicist's ntor.py).

install(cd, torch, st, k31=1.0, K=8, skip1=True, ct=0.9, cs=1.0, ca=0.5, cb=0.25, lo=2, hi=14, dump=None, law=2)
  law=2 (E35c): the ReLU step of the matrix = gate term + Gaussian births + own (3,1) + mean-tilt and variance-tilt row forms + diagonal,
  N_post y = g o (gbar Ntilde (g o y)) + g o (Co (S2 o (Co (g o y)))) + r o (Co (g o y)) + g o (Co (r o y)) + Nd o y,  Co = C without its diagonal,
  r = S3 + d tc - (d alpha / 2 sigma) Pc,  tc = Co beta + gamma-weighted trace of the bank,  Pc = 2 (Co o Co) gamma + gbar diag(Ntilde) + conversion diagonal,
  Nd = S4 + beta (Co beta) + 2 gamma (Co o Co) gamma - 2 g^2 (Co o Co) g^2;  then q = 2 W' (N_post (W u)) + conversion.  law=1: the first version.
  st: the harness state dict holding st['pp'] (pushed bank columns of the current layer), filled by the harness's push wrapper.
  skip1: leave the partner vector of mode 1 (the mean direction) out of the plane (it is still carried).
  q1c > 0: mode 1's partner vector is set to q1c x ramp x sign x the bank's trace vector (and mode 1 enters the plane).
  sweeps > 0: the K vectors are the previous layer's vectors pushed (w' (g o U)) and refined by that many power sweeps with C (the first carried layer
  uses the exact eigenvectors); 0 = exact eigenvectors at every layer.
  ca: a = ca x radial / 3;  cb: b (Monte Carlo 0.23-0.29).  dump: dict to receive U, Q per layer (open-loop comparison with the truth)."""
import math


def install(cd, torch, st, k31=1.0, K=8, skip1=True, ct=0.9, cs=1.0, ca=0.5, cb=0.25, lo=2, hi=14, conv=1.0, dump=None, law=2, q1c=0.0, sweeps=0):
    t = torch; f64 = t.float64; n = 1024; S = dict(prev=None); info = {}; cd.ncar_info = info
    old_pin = cd.hooks.get('pair_in')

    def apply_N(P, y):          # Ntilde(l) y for the previous layer's model, y: (n, k)
        Cy = P['C'] @ y
        out = P['a'] * P['trC'] * Cy + P['b'] * (P['C'] @ Cy)
        if P['Q'] is not None: out = out + P['U'] @ (P['Q'].T @ y) + P['Q'] @ (P['U'].T @ y)
        return out

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if l < 1 or l > 14: return pin
        mu, Cpre, k21, k4v, k22, rem = pin
        C = Cpre.to(f64); P = S['prev']; Q = None
        if sweeps and P is not None:
            # native form of the subspace: the previous layer's vectors pushed through the gates and the weights, then `sweeps` power sweeps with C
            U = t.linalg.qr(cd.ws[l].to(f64).T @ (P['g'][:, None] * P['U']))[0]
            for _ in range(int(sweeps)): U = t.linalg.qr(C @ U)[0]
            lam = t.zeros(3, dtype=f64)
        else:
            lam, V = t.linalg.eigh(C); U = V[:, -K:].flip(1)
        if P is not None:
            w = cd.ws[l].to(f64)                                   # pre(l) = w.T @ post(l-1)
            y = w @ U; yg = P['g'][:, None] * y
            q = ct * 2 * P['gbar'] * (w.T @ (P['g'][:, None] * apply_N(P, yg)))
            if law == 1:
                src = 2 * (w.T @ (P['g'][:, None] * (P['C'] @ (P['S2'][:, None] * (P['C'] @ yg))))) + 2 * (w.T @ (P['S4'][:, None] * y))
            else:
                Co, r, Nd, g_ = P['Co'], P['r'], P['Nd'], P['g']; Cyg = Co @ yg
                src = 2 * (w.T @ (g_[:, None] * (Co @ (P['S2'][:, None] * Cyg)) + r[:, None] * Cyg + g_[:, None] * (Co @ (r[:, None] * y)) + Nd[:, None] * y))
            if conv and P['c'] is not None and P['c'].numel():
                pp = st['pp'].to(f64)[:, :P['c'].numel()]
                src = src + conv * 2 * (pp @ (P['c'][:, None] * (pp.T @ U)))
            q = q + cs * src
            Q = q - U @ (U.T @ q)
            if q1c and P.get('lam_next') is not None:
                # the partner vector of mode 1 (the mean direction) is mostly the trace vector of the third cumulant (ncar_mode1.py: cosine 0.6-0.68 at
                # layers 9-14, coefficient 0.6-1.0, growing from layer 4): q_1 = q1c x ramp(l) x sign(u_1 . mu) x (I - P) tau, tau_i = sum_r lam_r pp_ri |pp_r|^2
                ppq = st['pp'].to(f64); lq = P['lam_next']
                if lq.numel() >= ppq.shape[1] and ppq.shape[1]:
                    tau = ppq @ (lq[:ppq.shape[1]] * (ppq * ppq).sum(0)); tau = tau - U @ (U.T @ tau)
                    Q[:, 0] = q1c * min(1.0, max(0.0, (l - 3) / 6.0)) * t.sign(U[:, 0] @ mu.to(f64)) * tau
            info[l] = dict(q_norm=[float(x) for x in Q.norm(dim=0)[:4]], top=[float(x) for x in lam[-3:].flip(0)])
            if dump is not None:
                ppd = st['pp'].to(f64); lamd = ctx.get('lam_cur')
                dump[l] = dict(U=U.clone(), Q=Q.clone(), bpre=ctx['bpre'].detach().to(f64).clone(), mu=mu.detach().to(f64).clone(),
                               tau=(ppd @ (P['lam_next'][:ppd.shape[1]] * (ppd * ppd).sum(0))) if P.get('lam_next') is not None and P['lam_next'].numel() >= ppd.shape[1] else t.zeros(n, dtype=f64))
            if k31 and lo <= l <= hi:
                sel = slice(1, K) if (skip1 and not q1c) else slice(0, K)
                M = U[:, sel] @ Q[:, sel].T; M = M + M.T
                s = t.diagonal(C).clamp_min(1e-10).sqrt()
                rm = k31 * (3.0 / n) * (M - t.diag(t.diagonal(M))) / (s ** 3)[:, None] / s[None, :]
                rem = rm if rem is None else rem + rm
        S['cur'] = dict(C=C, U=U, Q=Q, trC=float(t.trace(C)), radial=float(ctx['radial']))
        return (mu, Cpre, k21, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook

    old_out = cd.hooks.get('state_out')

    def out_hook(l, out, ctx):
        if old_out is not None: out = old_out(l, out, ctx)
        if l <= 14 and 'cur' in S and l >= 1:
            cur = S.pop('cur'); aux = ctx['aux']; d64 = lambda x: x.detach().to(f64)
            g, m, sg, pdf, al = d64(aux['gate']), d64(ctx['host']), d64(aux['sigma']), d64(aux['pdf']), d64(aux['alpha'])
            v = sg * sg; d = pdf / sg
            r2 = v * ((1 + al * al) * g + al * pdf); r3 = v * sg * ((al ** 3 + 3 * al) * g + (al * al + 2) * pdf); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * pdf)
            S2 = 2 * (g - g * g - m * d); S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2
            pp = st['pp'].to(f64); lam_all = d64(out[3]); nold = pp.shape[1]
            c = lam_all[:nold] * (pp.T @ (2 * m * (1 - g))) if nold else None
            prev = dict(C=cur['C'], U=cur['U'], Q=cur['Q'], trC=cur['trC'], a=ca * cur['radial'] / 3, b=cb, g=g, gbar=float((g - m * d).mean()), S2=S2, S4=S4, c=c)
            if law != 1:
                Cc = cur['C']; Co = Cc - t.diag(t.diagonal(Cc)); gam = g - m * d; beta = 2 * m * (1 - g); S3 = 3 * (1 - g) * (r2 - 2 * m * m); Co2 = Co * Co
                tc = Co @ beta
                if nold: tc = tc + pp @ (lam_all[:nold] * ((pp * pp).T @ gam))                      # gamma-weighted trace of the bank's third cumulant
                Ndiag = prev['a'] * prev['trC'] * v + prev['b'] * (Cc * Cc).sum(1)                   # diagonal of the carried matrix
                if cur['Q'] is not None: Ndiag = Ndiag + 2 * (cur['U'] * cur['Q']).sum(1)
                Pc = 2 * (Co2 @ gam) + prev['gbar'] * Ndiag
                if nold: Pc = Pc + (pp * pp) @ c
                prev.update(Co=Co, r=S3 + d * tc - (d * al / (2 * sg)) * Pc, Nd=S4 + beta * (Co @ beta) + 2 * gam * (Co2 @ gam) - 2 * g * g * (Co2 @ (g * g)))
            prev['lam_next'] = lam_all            # weights of all columns after this layer's births = the columns the next layer pushes
            S['prev'] = prev
        return out
    cd.hooks['state_out'] = out_hook
    return cd
