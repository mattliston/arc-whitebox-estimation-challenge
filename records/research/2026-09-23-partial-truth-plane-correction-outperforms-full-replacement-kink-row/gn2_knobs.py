#!/usr/bin/env python3
"""TAG fable-covhunter (CH-18): Gauss-Newton screen of zero-cost covariance-channel knobs ON TOP OF the physicist's K31 row factor
(the fire candidate's physics: T31 = radial rho (c0 + c1 mu_a k3_a / v_a^2), all pair outputs, L1-14; c0 .85, c1 .5 by default).
Knobs (global strengths, L2-14, pair map COVARIANCE call only unless noted): iso (K21 kink += th k3_a C_ac/v_a), bt, k21n (non-kink
K21 x(1+th)), k31 (T31 of the covariance call x(1+th)), k22 (K22 off x(1+th)), edge (x(1+th), L8-14), rc (recentering x(1+th)), e47 (the frozen message edge also at L4-7 with scale th).
usage: gn2_knobs.py <ranks> <knobs> <threads> <out.npz> [validate theta json] [c0,c1]"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
n = 1024; eye = torch.eye(n, dtype=torch.bool)
STEP = dict(iso=0.05, bt=0.03, k21n=0.05, k31=0.05, k22=0.05, edge=0.1, rc=0.1, e47=0.1, e46=0.1)
C0C1 = (0.85, 0.5)


def row_r31(mu, Cpre, k21, radial):
    vv = torch.diagonal(Cpre); s = vv.sqrt(); k3 = torch.diagonal(k21)
    rowf = (C0C1[0] - 1) + C0C1[1] * mu * k3 / (vv * vv)
    return ((radial * rowf)[:, None] * (Cpre / (s[:, None] * s[None, :]))).masked_fill(eye, 0.0).to(torch.float32).to(torch.float64)


def configure(card, th, base=True):
    if base:
        def pin(l, v, ctx):
            if l < 1 or l > 14: return v
            mu, Cpre, k21, k4, k22, _ = v
            return (mu, Cpre, k21, k4, k22, row_r31(mu, Cpre, k21, ctx['radial']))
        card.hooks['pair_in'] = pin
    cov_knobs = {k: v for k, v in th.items() if k not in ('edge', 'rc', 'e47', 'e46') and v != 0}
    if cov_knobs:
        def cov(l, C, ctx):
            if l < 2 or l > 14: return C
            mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
            s = torch.sqrt(torch.diagonal(Cpre)); vv = torch.diagonal(Cpre); al = mu / s
            kink = (al.abs() < 1.5).double(); km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
            nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
            k3 = torch.diagonal(k21); bpre = ctx['bpre']; rho = Cpre / torch.outer(s, s); t4 = k4 / vv ** 2
            r31 = row_r31(mu, Cpre, k21, ctx['radial']) if base else torch.zeros_like(Cpre)
            k2, k22n = k21, k22
            for k, t in cov_knobs.items():
                if k == 'iso': k2 = k2 + t * km * (k3[:, None] * Cpre / vv[:, None])
                elif k == 'bt': k2 = k2 + t * km * (vv[:, None] * bpre[None, :] + 2 * Cpre * bpre[:, None])
                elif k == 'k21n': k2 = k2 + t * nm * k21
                elif k == 'k22': k22n = torch.where(eye, k22n, (1 + t) * k22n)
                elif k == 'k31': r31 = r31 + t * ((t4[:, None] * rho).masked_fill(eye, 0.) + r31)
            return pair(mu, Cpre, k2, k4, k22n, 'covariance', remainder31=r31)[1]
        card.hooks['cov_pm'] = cov
    if th.get('edge', 0): card.hooks['edge'] = lambda l, v, ctx, t=th['edge']: v * (1 + t)
    if th.get('e47', 0) or th.get('e46', 0):
        prev = card.hooks.get('cov_pm')
        def cov2(l, C, ctx, prev=prev):
            C = C if prev is None else prev(l, C, ctx)
            t = th.get('e47', 0) if 4 <= l <= 7 else 0.0
            if th.get('e46', 0) and 4 <= l <= 6: t = th['e46']
            if t == 0: return C
            f = card.env; context = {'left': card.edge['context_left'], 'right': card.edge['context_right']}
            return C + t * f['_message_edge_source'](8, ctx['host'], ctx['k21'], ctx['aux'], card.w[l], ctx['hstate'], card.P, context)
        card.hooks['cov_pm'] = cov2
    if th.get('rc', 0): card.knobs['recenter'] = 1 + th['rc']


def final_err(w, truth, th, pl, base=True):
    p, a, c, e = pl
    card = cs.HookCard(torch, w, p, a, c, e); configure(card, th, base)
    with torch.no_grad(): outs, _ = card.run_states()
    return outs[15].clamp_min(0).to(torch.float32).numpy().astype(np.float64) - truth[15]


if __name__ == '__main__':
    ranks = [int(x) for x in sys.argv[1].split(',')]; knobs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); out = sys.argv[4]
    if len(sys.argv) > 6: C0C1 = tuple(float(x) for x in sys.argv[6].split(','))
    pl = cs.model.torch_payload(torch, np, False)
    if len(sys.argv) > 5 and sys.argv[5] != '-':
        th = json.loads(sys.argv[5])
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk)
            eb = final_err(w, truth, {}, pl, base=False); e0 = final_err(w, truth, {}, pl); e1 = final_err(w, truth, th, pl)
            mb, m0, m1 = [float((x ** 2).mean()) for x in (eb, e0, e1)]
            rec = dict(rank=rk, theta=th, c0c1=C0C1, bank=mb, rowbase=m0, knobs=m1, row_vs_bank=m0 / mb, knobs_vs_row=m1 / m0, knobs_vs_bank=m1 / mb)
            print(json.dumps(rec), flush=True)
            if out != '/dev/null':
                with open(out, 'a') as f: f.write(json.dumps(rec) + '\n')
        sys.exit()
    E0, U = {}, {}
    for rk in ranks:
        w = cs.weights(rk); truth = cs.truthpost(rk); t0 = time.time()
        E0[rk] = final_err(w, truth, {}, pl); cols = []
        for k in knobs:
            ek = final_err(w, truth, {k: STEP[k]}, pl); cols.append((ek - E0[rk]) / STEP[k])
            print(json.dumps(dict(rank=rk, knob=k, step=STEP[k], ratio_fd=float((ek ** 2).mean() / (E0[rk] ** 2).mean()))), flush=True)
        U[rk] = np.stack(cols, 1)
        print(f'# rank {rk} done {time.time() - t0:.0f}s', flush=True)
    np.savez(out, knobs=np.array(knobs), ranks=np.array(ranks), **{f'e0_{r}': E0[r] for r in ranks}, **{f'U_{r}': U[r] for r in ranks})
