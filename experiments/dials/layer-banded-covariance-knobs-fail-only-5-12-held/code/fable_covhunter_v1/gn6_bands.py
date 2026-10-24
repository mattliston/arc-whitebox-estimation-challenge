#!/usr/bin/env python3
"""TAG fable-covhunter (CH-30): BAND KNOBS on the S2 card (lead's request). The three covariance-only v3 knobs (iso, k21n, k31) get
per-band DEVIATIONS from their v3 values in 4 bands, b1 = L2-5, b2 = L6-9, b3 = L10-12, b4 = L13-14 (12 knobs), plus global deviations
of the edge and recentering scales (2 knobs). Base = S2 + v3 (gn3_knobs). Fit: one finite-difference run per knob and net, restarted
from the cached state entering the knob's first layer (bit-identical to a full run); quadratic model with a ridge pull toward v3
(penalty lam * sum (th_k / step_k)^2, lam by leave-one-net-out). Eval: S2+v3 vs S2+v3+deviations, per net.
usage: gn6_bands.py fit <ranks> <threads> <out.npz> | gn6_bands.py eval <ranks|valid40> <theta json> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
import covstep_s2 as s2
n = 1024; eye = torch.eye(n, dtype=torch.bool)
V3 = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212, 'edge': 0.164, 'rc': -0.080}
BANDS = {'b1': (2, 5), 'b2': (6, 9), 'b3': (10, 12), 'b4': (13, 14)}
KNOBS = [f'{k}_{b}' for b in BANDS for k in ('iso', 'k21n', 'k31')] + ['edge', 'rc']
STEP = {k: (0.1 if k in ('edge', 'rc') else 0.05) for k in KNOBS}
FIRST = {**{f'{k}_{b}': lo for b, (lo, hi) in BANDS.items() for k in ('iso', 'k21n', 'k31')}, 'edge': 8, 'rc': 1}


def band_of(l):
    for b, (lo, hi) in BANDS.items():
        if lo <= l <= hi: return b
    return None


def configure(card, th):
    s2.make_s2(card)
    def cov(l, C, ctx):
        b = band_of(l)
        if b is None: return C
        iso = V3['iso'] + th.get(f'iso_{b}', 0.0); k21n = V3['k21n'] + th.get(f'k21n_{b}', 0.0); k31 = V3['k31'] + th.get(f'k31_{b}', 0.0)
        mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
        vv = torch.diagonal(Cpre); s = vv.sqrt(); kink = ((mu / s).abs() < 1.5).double()
        km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
        k2 = k21 + iso * km * (k3[:, None] * Cpre / vv[:, None]) + k21n * nm * k21
        r31 = k31 * ((k4 / vv ** 2) * (0.85 + 0.5 * mu * k3 / (vv * vv)))[:, None] * rho
        return pair(mu, Cpre, k2, k4, k22, 'covariance', remainder31=r31)[1]
    card.hooks['cov_pm'] = cov
    card.hooks['edge'] = lambda l, v, ctx: v * (1 + V3['edge'] + th.get('edge', 0.0))
    card.knobs['recenter'] = 1 + V3['rc'] + th.get('rc', 0.0)


def run_from(card, states, start, truth):
    with torch.no_grad():
        st = states[start]
        for l in range(start, 16): st = card.step(l, st)
    return st[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64) - truth[15]


def base_states(w, pl):
    card = cs.HookCard(torch, w, *pl); configure(card, {})
    keep = {}
    card.hooks['state_in'] = lambda l, st, ctx: (keep.__setitem__(l, st) if l in (1, 2, 6, 8, 10, 13) else None) or st
    with torch.no_grad():
        st = card.initial()
        for l in range(16): st = card.step(l, st)
    del card.hooks['state_in']
    return keep, st[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64)


if __name__ == '__main__':
    mode = sys.argv[1]; pl = cs.model.torch_payload(torch, np, False)
    if mode == 'fit':
        ranks = [int(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3])); out = sys.argv[4]
        E0, U = {}, {}
        for rk in ranks:
            t0 = time.time(); w = cs.weights(rk); truth = cs.truthpost(rk)
            states, pred = base_states(w, pl); E0[rk] = pred - truth[15]; cols = []
            for k in KNOBS:
                card = cs.HookCard(torch, w, *pl); configure(card, {k: STEP[k]})
                cols.append((run_from(card, states, FIRST[k], truth) - E0[rk]) / STEP[k])
            U[rk] = np.stack(cols, 1)
            print(json.dumps(dict(rank=rk, mse=float((E0[rk] ** 2).mean()), wall=round(time.time() - t0, 1))), flush=True)
            np.savez(out, knobs=np.array(KNOBS), ranks=np.array(list(E0)), steps=np.array([STEP[k] for k in KNOBS]),
                     **{f'e0_{r}': E0[r] for r in E0}, **{f'U_{r}': U[r] for r in U})
    else:
        arg, th, thr, out = sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]); torch.set_num_threads(thr)
        ranks = json.loads((cs.P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'] if arg == 'valid40' else [int(x) for x in arg.split(',')]
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk)
            states, pred = base_states(w, pl); e0 = pred - truth[15]
            card = cs.HookCard(torch, w, *pl); configure(card, th); e1 = run_from(card, states, 1, truth)
            rec = dict(rank=rk, card='S2+v3', theta=th, base=float((e0 ** 2).mean()), knobs=float((e1 ** 2).mean())); rec['ratio'] = rec['knobs'] / rec['base']
            print(json.dumps(rec), flush=True)
            with out.open('a') as f: f.write(json.dumps(rec) + '\n')
