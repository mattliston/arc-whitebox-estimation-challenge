#!/usr/bin/env python3
"""TAG opus-physicist T1: ONE-STEP INJECTION BUDGET of the carried covariance on the CURRENT torch card (pcard.build), per pair-map input.
The card runs closed loop once (learned ON); then step l is re-run from the card's own state at l-1 with the post mean and covariance
set to the 32M Monte Carlo truth (fable_covhunter_v1/mcall). Inside that step the pair map's covariance output is recomputed with
EFFECTIVE inputs (what the card's covariance channel really reads, i.e. including the three covariance knobs iso / k21n / k31):
   K21eff = K21 + iso * kink_a k3_a C_ac / v_a + k21n * (1 - kink_a) K21_ac      (kink_a: |mu_a/sigma_a| < 1.5)
   T31eff = (1 + k31) * (k4_a/v_a^2) (c0 + c1 mu_a k3_a / v_a^2) rho_ac ;  K22 = radial (v v' + 2 C o C)/3 ;  k4 = radial v^2 x band dial
Arms (each = one pair-map call, covariance output):
   AT   all inputs at truth (8M planes K21/K22/K31, 32M per-unit k3/k4)             -> the floor (truth noise + series truncation)
   CA   all inputs the card's (must equal the card's own hook output: certificate 'ca_vs_hook')
   A:X  all truth, input X the card's      -> error CAUSED by the card's X alone  (the September 'one card plane' definition)
   B:X  all the card's, input X at truth   -> change when X alone is fixed
   X in K21k (K21 off-diagonal, kink rows), K21o (other rows), K31, K22 (off-diagonal), k4 (per-unit), k3 (per-unit)
Grades of a covariance difference dC (against AT for A arms, against CA for B arms, against the truth post covariance for totals):
   v     relative rms of the NEXT layer's per-unit variance change, diag(W' dC W) / v_true                         [%]
   mse   mean over units of (phi(a)/(2 sigma) dv)^2: the first-order mean-squared error it puts into the next post mean   [1e-9]
   perp  ||P dC P||_off / ||C_true||_off, P = I - h h'/|h|^2, h = true post mean                                    [%]
   share (B arms) fraction of the card's one-step injection removed by fixing X: -<d_X, inj>/<inj, inj> in the v metric and the perp metric
Noise: where the independent 8M plane draw exists (layers 2, 5, 9, 12) the A arms are repeated with it (suffix _n2) and the A-arm
values are also given as cross products of the two draws (noise-free).
usage: t1_onestep.py <rank> <threads> [layers, default 4..14]  -> t1_rank{rank}.json, t1_rank{rank}_dv.npz"""
import sys, json, time, math
import numpy as np, torch
import pcard as pc
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2])
layers = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(range(4, 15))
N = 1024; eye = torch.eye(N, dtype=torch.bool); offm = ~eye
MC = pc.mcall(rank); PL = pc.planes(rank); PU = pc.pu(rank)
try: PL2 = pc.planes(rank, '_noise2')
except FileNotFoundError: PL2 = None
T64 = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))
W = pc.weights(rank)
KN = dict(iso=-0.094, k21n=-0.028, k31=-0.212)     # gn_linear_hook.V3, the knobs the current card applies to its covariance output

cd = pc.build(W)
HOOKS0 = dict(cd.hooks)
states = {}
pc.wrap(cd, 'state_in', lambda l, st, ctx: states.__setitem__(l, st) or st)
tp = pc.truth(rank); t0 = time.time()
pred = pc.run(cd)
base_mse = pc.mse(pred, tp)
print(json.dumps(dict(TAG='opus-physicist T1 one-step injection budget, current torch card', rank=rank, closed_loop_final_x1e9=round(float(base_mse[15]) * 1e9, 3),
                      wall=round(time.time() - t0, 1))), flush=True)
rowf = cd.env['_ROWF'][0]
ARMS = ('K21k', 'K21o', 'K31', 'K22', 'k4', 'k3')
out = dict(TAG='opus-physicist T1', rank=rank, knobs=KN, dials=pc.DIALS, layers={}); DV = {}


def onestep(l):
    st = list(states[l]); st[0] = T64(MC[f'mean_post_{l - 1}']); st[1] = T64(MC[f'cov_post_{l - 1}'])
    got = {}

    def cov_pm(l_, C, ctx):
        mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
        v = torch.diagonal(Cpre); s = v.sqrt(); al = mu / s; kink = (al.abs() < 1.5).double()
        rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
        k3c = torch.diagonal(k21); k4c = k4.expand(N) if k4.ndim == 0 else k4
        # the card's effective covariance-channel inputs
        K21c = k21 + (KN['iso'] * kink * k3c / v)[:, None] * Cpre + (KN['k21n'] * (1 - kink))[:, None] * k21
        K21c = torch.diagonal_scatter(K21c, k3c)
        T31c = ((1 + KN['k31']) * (k4c / v ** 2) * rowf(mu, k3c, v))[:, None] * rho
        K22c = k22
        # truth inputs
        k3t, k4t = T64(PU['mc_pre_k3'][l_]), T64(PU['mc_pre_k4'][l_])

        def truth_planes(P):
            K21t = torch.diagonal_scatter(T64(P[f'K21_{l_}']), k3t)
            K22t = torch.diagonal_scatter(T64(P[f'K22_{l_}']), k4t)
            T31t = (T64(P[f'K31_{l_}']) / (v[:, None] * s[:, None] * s[None, :])).masked_fill(eye, 0.)
            return K21t, K22t, T31t

        def call(K21, k4, K22, T31):
            k3_ = torch.diagonal(K21); k4_ = k4
            internal = ((k4_ / v ** 2) * rowf(mu, k3_, v))[:, None] * rho          # what the pair map builds itself from (k4, k3)
            return pair(mu, Cpre, K21, k4_, K22, 'covariance', remainder31=(T31 - internal).masked_fill(eye, 0.))[1]

        def swap(base, X, src):
            """base, src: dicts K21, k4, K22, T31; returns base with input X taken from src."""
            r = dict(base)
            if X in ('K21k', 'K21o'):
                rows = kink if X == 'K21k' else 1 - kink
                msk = (rows[:, None] * torch.ones(1, N, dtype=torch.float64)).masked_fill(eye, 0.)
                r['K21'] = base['K21'] * (1 - msk) + src['K21'] * msk
            elif X == 'K31': r['T31'] = src['T31']
            elif X == 'K22': r['K22'] = torch.where(eye, base['K22'], src['K22'])
            elif X == 'k4':
                r['k4'] = src['k4']; r['K22'] = torch.diagonal_scatter(base['K22'], torch.diagonal(src['K22']))
            elif X == 'k3': r['K21'] = torch.diagonal_scatter(base['K21'], torch.diagonal(src['K21']))
            return r

        card = dict(K21=K21c, k4=k4c, K22=K22c, T31=T31c)
        K21t, K22t, T31t = truth_planes(PL); tru = dict(K21=K21t, k4=k4t, K22=K22t, T31=T31t)
        cov = dict(AT=call(**tru), CA=call(**card), hook=C.clone())
        for X in ARMS:
            cov['A:' + X] = call(**swap(tru, X, card)); cov['B:' + X] = call(**swap(card, X, tru))
        if PL2 is not None and f'K21_{l_}' in PL2.files:
            K21u, K22u, T31u = truth_planes(PL2); tru2 = dict(K21=K21u, k4=k4t, K22=K22u, T31=T31u)
            cov['AT_n2'] = call(**tru2)
            for X in ('K21k', 'K21o', 'K31', 'K22'): cov['A:' + X + '_n2'] = call(**swap(tru2, X, card))
        got.update(cov=cov, kink=kink, alive=ctx['alive'].clone(), resp=ctx['response_mask'].clone(), radial=float(ctx['radial']))
        return C

    cd.hooks = dict(HOOKS0); pc.wrap(cd, 'cov_pm', cov_pm)
    pc.wrap(cd, 'cov_out', lambda l_, C, ctx: got.__setitem__('carried', C.clone()) or C)
    cd.step(l, tuple(st))
    return got


for l in layers:
    t1 = time.time(); got = onestep(l); cov = got['cov']
    Tl = T64(MC[f'cov_post_{l}']); TA = T64(MC[f'cov_post_A_{l}']); TB = T64(MC[f'cov_post_B_{l}'])
    h = T64(MC[f'mean_post_{l}']); hh = h / h.norm()
    Wn = T64(W[l + 1]); vt = torch.diagonal(T64(MC[f'cov_pre_{l + 1}'])); mu1 = T64(MC[f'mean_pre_{l + 1}'])
    s1 = vt.sqrt(); a1 = mu1 / s1; wgt = torch.exp(-a1 ** 2 / 2) / math.sqrt(2 * math.pi) / (2 * s1)
    nT = float(Tl[offm].norm())

    def perp(D):
        Dh = D @ hh
        return D - torch.outer(hh, Dh) - torch.outer(Dh, hh) + (hh @ Dh) * torch.outer(hh, hh)

    def dv(D): return ((Wn * (D @ Wn)).sum(0)) / vt

    def grade(D):
        x = dv(D)
        return dict(v=100 * float((x ** 2).mean().sqrt()), mse=1e9 * float(((wgt * vt * x) ** 2).mean()), perp=100 * float(perp(D)[offm].norm()) / nT,
                    off=100 * float(D[offm].norm()) / nT), x

    rec = dict(radial=got['radial'], n_kink=int(got['kink'].sum()), n_alive=int(got['alive'].sum()), n_resp=int(got['resp'].sum()),
               ca_vs_hook=float((cov['CA'] - cov['hook'])[offm].norm() / (cov['hook'] - cov['AT'])[offm].norm()))
    # totals against the truth post covariance: floor (all truth), the card at the pair-map stage, the card's carried covariance
    floorAB = (TA - TB) / 2
    for name, C_ in (('AT', cov['AT']), ('CA', cov['hook']), ('carried', got['carried'])):
        g, x = grade(C_ - Tl); rec['tot_' + name] = g; DV[f'{l}_tot_{name}'] = x.numpy()
    g, x = grade(floorAB); rec['halfdiff_32M'] = g
    inj = cov['hook'] - cov['AT']; ginj, xinj = grade(inj); rec['inj_CA_minus_AT'] = ginj; DV[f'{l}_inj'] = xinj.numpy()
    Pinj = perp(inj)[offm]
    for X in ARMS:
        ga, xa = grade(cov['A:' + X] - cov['AT']); gb, xb = grade(cov['B:' + X] - cov['hook'])
        PB = perp(cov['B:' + X] - cov['hook'])[offm]
        gb['share_v'] = -float((xb * xinj).sum() / (xinj * xinj).sum()); gb['share_perp'] = -float((PB * Pinj).sum() / (Pinj * Pinj).sum())
        rec['A:' + X] = ga; rec['B:' + X] = gb; DV[f'{l}_A_{X}'] = xa.numpy(); DV[f'{l}_B_{X}'] = xb.numpy()
        if 'A:' + X + '_n2' in cov:
            g2, x2 = grade(cov['A:' + X + '_n2'] - cov['AT_n2'])
            D1, D2 = perp(cov['A:' + X] - cov['AT'])[offm], perp(cov['A:' + X + '_n2'] - cov['AT_n2'])[offm]
            rec['A:' + X + '_x'] = dict(v=100 * math.sqrt(max(float((xa * x2).mean()), 0.)), perp=100 * math.sqrt(max(float((D1 * D2).sum()), 0.)) / nT,
                                        mse=1e9 * float(((wgt * vt) ** 2 * xa * x2).mean()))
    if 'AT_n2' in cov:
        g, x = grade((cov['AT'] - cov['AT_n2']) / math.sqrt(2)); rec['plane_noise_AT'] = g
    out['layers'][l] = rec
    f = lambda d: f"{d['v']:.4f}/{d['mse']:.3f}/{d['perp']:.3f}"
    print(f"L{l:02d} kink {rec['n_kink']} resp {rec['n_resp']} alive {rec['n_alive']} | v%/mse/perp%: floor(AT) {f(rec['tot_AT'])} card-pm {f(rec['tot_CA'])} carried {f(rec['tot_carried'])} inj {f(ginj)} | "
          + ' '.join(f"{X}: A {f(rec['A:' + X])} B-share v {rec['B:' + X]['share_v']:+.2f} perp {rec['B:' + X]['share_perp']:+.2f}" for X in ARMS)
          + f" | ca_vs_hook {rec['ca_vs_hook']:.1e} | {time.time() - t1:.0f}s", flush=True)
    json.dump(out, open(pc.HERE / f't1_rank{rank:04d}.json', 'w'), indent=1)
np.savez(pc.HERE / f't1_rank{rank:04d}_dv.npz', **DV)
print(json.dumps(dict(done=True, rank=rank, wall=round(time.time() - t0, 1))), flush=True)
