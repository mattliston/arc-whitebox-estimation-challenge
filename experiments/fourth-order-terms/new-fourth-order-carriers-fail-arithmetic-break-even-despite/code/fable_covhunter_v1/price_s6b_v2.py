#!/usr/bin/env python3
"""TAG fable-covhunter (CH-40, v2 of CH-35): PRICE LIST of the remaining levers on the CURRENT card (lead's request). Card: the analyst's torch S6b
(fable_analyst_v1/s6bcard.py = S2 physics + v3 knobs + L15 row cut + refit epoch-2 literals), learned parts ON. In closed loop ONE channel
is moved toward truth at ONE layer band, value = card + eps (truth - card), eps .5 and 1; final-MSE ratio vs the unchanged S6b.
Channels:
  a  post mean m_l (after step l)                      b  post covariance C_l, off-diagonal only (after step l; diagonal kept)
  c  per-unit pre variance v_l (diagonal of Cpre at l; v15 at L15)
  d  K31 plane, covariance output only                 e  K21 kink rows (|mu/sd|<1.5), covariance output only
  f  per-unit k3 and k4 together, all pair outputs (the card's K31 plane kept; k3_15/k4_15 at L15)
  g  all three truth planes (K21 all rows, K22, K31) on the covariance output, change projected off the host mean (as sec 12 on S2)
  k31k / q4k / jk : truth K31 / K31+K22 / K21+K31+K22 on KINK ROWS only (|mu/sd|<1.5), covariance output only (read-visible content)
d and e act on the S6b covariance call INCLUDING the v3 knobs: the covariance input is mixed from its v3 value toward truth.
Bands: L2-5, L6-9, L10-12, L13-14, L15. Truths: mcall_rank*_P32M.npz (a, b, c; nets 35/100/37/39), 8M planes public_g19 mps_planes
(d, e) and 32M per-unit mps_truth_P32M (f), both nets 35/100 only. Each run restarts from the base run's cached state at the band start.
usage: price_s6b_v2.py <ranks> <channels> <threads> <out.jsonl> [bands comma]"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
for _p in (str(HERE), str(P2 / 'fable_analyst_v1')):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np, torch
import s6bcard as s6
ranks = [int(x) for x in sys.argv[1].split(',')]; chans = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); out = Path(sys.argv[4])
n = 1024; eye = torch.eye(n, dtype=torch.bool)
V3 = s6.V3; C0, C1 = 0.85, 0.5
BANDS = {'L2-5': (2, 5), 'L6-9': (6, 9), 'L10-12': (10, 12), 'L13-14': (13, 14), 'L15': (15, 15), 'L6-14': (6, 14)}
SEL = sys.argv[5].split(',') if len(sys.argv) > 5 else list(BANDS)
T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)


def install(cd, mix, band, eps, mc, PL, TR):
    """replace the v3 covariance hook by an identical one that can mix d/e/f toward truth; add a/b/c/f hooks for layers in band."""
    lo, hi = BANDS[band]
    inb = lambda l: lo <= l <= hi
    def rowf(mu, k3, vv): return C0 + C1 * mu * k3 / (vv * vv)
    def cov(l, C, ctx):
        if l < 2 or l > 14: return C
        mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
        vv = torch.diagonal(Cpre); s = vv.sqrt(); al = mu / s
        kink = (al.abs() < 1.5).double(); km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
        k2, k22n, r31 = k21, k22, None                                   # --- v3 knobs, same arithmetic as gn3_knobs.configure
        k2 = k2 + V3['iso'] * km * (k3[:, None] * Cpre / vv[:, None])
        k2 = k2 + V3['k21n'] * nm * k21
        r31 = V3['k31'] * ((k4 / vv ** 2) * (C0 + C1 * mu * k3 / (vv * vv)))[:, None] * rho
        k4c = k4
        if inb(l) and mix == 'e':
            k2 = k2 + eps * km * (T(PL[f'K21_{l}']) - k2)
        if inb(l) and mix == 'd':
            Trow = ((k4 / vv ** 2) * rowf(mu, k3, vv))[:, None] * rho
            Tt = (T(PL[f'K31_{l}']) / (vv[:, None] * s[:, None] * s[None, :])).masked_fill(eye, 0.)
            r31 = r31 + eps * (Tt - (Trow + r31))
        if inb(l) and mix in ('k31k', 'q4k', 'jk'):                        # kink rows only, off-diagonal
            if mix == 'jk': k2 = k2 + eps * km * (T(PL[f'K21_{l}']) - k2)
            if mix in ('q4k', 'jk'): k22n = k22n + eps * km * (T(PL[f'K22_{l}']) - k22n)
            Trow = ((k4 / vv ** 2) * rowf(mu, k3, vv))[:, None] * rho
            Tt = (T(PL[f'K31_{l}']) / (vv[:, None] * s[:, None] * s[None, :])).masked_fill(eye, 0.)
            r31 = r31 + eps * km * (Tt - (Trow + r31))
        if inb(l) and mix == 'f':
            k3n = k3 + eps * (T(TR['mc_pre_k3'][l]) - k3); k4c = k4 + eps * (T(TR['mc_pre_k4'][l]) - k4)
            Trow_o = ((k4 / vv ** 2) * rowf(mu, k3, vv))[:, None] * rho; Trow_n = ((k4c / vv ** 2) * rowf(mu, k3n, vv))[:, None] * rho
            r31 = r31 + Trow_o - Trow_n                                  # keep the card's (v3) K31 plane
            k2 = torch.diagonal_scatter(k2, k3n)
        Cv3 = pair(mu, Cpre, k2, k4c, k22n, 'covariance', remainder31=r31)[1]
        if inb(l) and mix == 'g':                                        # all three truth planes, change projected off the host mean
            k21t = torch.where(eye, k21, T(PL[f'K21_{l}'])); k22t = torch.where(eye, k22, T(PL[f'K22_{l}']))
            Tt = (T(PL[f'K31_{l}']) / (vv[:, None] * s[:, None] * s[None, :])).masked_fill(eye, 0.)
            Trow = ((k4 / vv ** 2) * rowf(mu, k3, vv))[:, None] * rho
            D = pair(mu, Cpre, k21t, k4, k22t, 'covariance', remainder31=Tt - Trow)[1] - Cv3
            hh = ctx['host'] / ctx['host'].norm(); Dh = D @ hh; hD = hh @ D
            Cv3 = Cv3 + eps * (D - torch.outer(hh, hD) - torch.outer(Dh, hh) + (hh @ Dh) * torch.outer(hh, hh))
        return Cv3
    cd.hooks['cov_pm'] = cov
    if mix == 'f':
        def pin(l, v, ctx):
            if not (inb(l) and 1 <= l <= 14): return v
            mu, Cpre, k21, k4, k22, r31 = v
            vv = torch.diagonal(Cpre); s = vv.sqrt(); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
            k3 = torch.diagonal(k21); k3n = k3 + eps * (T(TR['mc_pre_k3'][l]) - k3); k4n = k4 + eps * (T(TR['mc_pre_k4'][l]) - k4)
            keep = ((k4 / vv ** 2) * rowf(mu, k3, vv))[:, None] * rho - ((k4n / vv ** 2) * rowf(mu, k3n, vv))[:, None] * rho
            return (mu, Cpre, torch.diagonal_scatter(k21, k3n), k4n, k22, keep.masked_fill(eye, 0.))
        cd.hooks['pair_in'] = pin
        if inb(15):
            cut = cd.hooks['k3_15']
            cd.hooks['k3_15'] = lambda l, k3, ctx: (lambda k: k + eps * (T(TR['mc_pre_k3'][15]) - k))(cut(l, k3, ctx))
            cd.hooks['k4_15'] = lambda l, k4, ctx: k4 + eps * (T(TR['mc_pre_k4'][15]) - k4)
    if mix in ('a', 'b'):
        def so(l, st, ctx):
            if not inb(l) or l > 14: return st
            m, C = st[0], st[1]
            if mix == 'a': m = m + eps * (T(mc[f'mean_post_{l}']) - m)
            else: C = torch.where(eye, C, C + eps * (T(mc[f'cov_post_{l}']) - C))
            return (m, C) + tuple(st[2:])
        cd.hooks['state_out'] = so
    if mix == 'c':
        cd.hooks['cpre'] = lambda l, Cp, ctx: torch.diagonal_scatter(Cp, torch.diagonal(Cp) + eps * (T(np.diag(mc[f'cov_pre_{l}'])) - torch.diagonal(Cp))) if inb(l) and l <= 14 else Cp
        if inb(15): cd.hooks['vv15'] = lambda l, vv, ctx: vv + eps * (T(np.diag(mc['cov_pre_15'])) - vv)


class Cached:
    """npz accessor that reads each array once (repeated NpzFile reads churn memory)"""
    def __init__(self, z): self.z, self.c = z, {}
    def __getitem__(self, k):
        if k not in self.c: self.c[k] = np.asarray(self.z[k])
        return self.c[k]


pl = s6.payload(torch)
for rk in ranks:
    w = s6.weights(rk); truth = s6.truthpost(rk); mc = Cached(np.load(HERE / f'mcall_rank{rk:04d}_P32M.npz'))
    plp = P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{rk:04d}.npz'; trp = P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rk:04d}_P32M.npz'
    PL = Cached(np.load(plp)) if plp.exists() else None; TR = Cached(np.load(trp)) if trp.exists() else None
    # base: the analyst's card as is, and with my identical cov hook (must agree bit-exactly); cache states at band starts
    with torch.no_grad():
        ref = s6.card(torch, w, *pl).run(False)[15].to(torch.float64).numpy()
    cd = s6.card(torch, w, *pl); install(cd, None, 'L2-5', 0.0, mc, PL, TR); states = {}
    cd.hooks['state_in'] = lambda l, st, ctx: (states.__setitem__(l, st) if l in (2, 6, 10, 13, 15) else None) or st
    with torch.no_grad():
        st = cd.initial()
        for l in range(16): st = cd.step(l, st)
    pred0 = st[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64)
    mse0 = float(((pred0 - truth[15]) ** 2).mean()); mref = float(((ref - truth[15]) ** 2).mean())
    rec = dict(rank=rk, cell='base', mse=mse0, analyst_card_mse=mref, identical=bool(mse0 == mref)); print(json.dumps(rec), flush=True)
    with out.open('a') as f: f.write(json.dumps(rec) + '\n')
    for ch in chans:
        if ch in ('d', 'e', 'g', 'k31k', 'q4k', 'jk') and PL is None: continue
        if ch == 'f' and TR is None: continue
        for band, (lo, hi) in ((b_, BANDS[b_]) for b_ in SEL):
            if lo == 15 and ch in ('a', 'b', 'd', 'e', 'g', 'k31k', 'q4k', 'jk'): continue
            for eps in (0.5, 1.0):
                t0 = time.time(); cd = s6.card(torch, w, *pl); install(cd, ch, band, eps, mc, PL, TR)
                with torch.no_grad():
                    st = states[lo]
                    for l in range(lo, 16): st = cd.step(l, st)
                pred = st[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64)
                mse = float(((pred - truth[15]) ** 2).mean())
                rec = dict(rank=rk, channel=ch, band=band, eps=eps, mse=mse, ratio=mse / mse0, wall=round(time.time() - t0, 1))
                print(json.dumps(rec), flush=True)
                with out.open('a') as f: f.write(json.dumps(rec) + '\n')
