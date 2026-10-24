#!/usr/bin/env python3
"""TAG fable-covhunter (CH-22): covariance-channel knobs ON THE S2 CARD (fired #331947: K31 row factor .85/.5 in the pair map, gain
b-fix gamma .1, every3 joins; covstep_s2.make_s2, bit-exact vs fable_analyst_v1/s2card.py). Knobs (global strengths):
  iso  (L2-14, covariance output only): kink rows (|mu/sd|<1.5) K21_ac += th * k3_a C_ac / v_a
  k21n (L2-14, covariance output only): non-kink rows K21 x (1+th)
  k31  (L2-14, covariance output only): T31 (with the row factor) x (1+th)
  k22  (L2-14, covariance output only): K22 off-diagonal x (1+th)
  edge (L8-14): message edge x (1+th);  rc: recentering x (1+th)
  bfix: gain b-fix gamma = .1 + th;  c0, c1: K31 row constants (.85 + th, .5 + th), all pair outputs
Modes: fit  -> gn3_knobs.py fit <ranks> <knobs> <threads> <out.npz>   (one finite-difference run per knob, quadratic model)
       eval -> gn3_knobs.py eval <ranks|valid40> <theta json> <threads> <out.jsonl>   (S2 vs S2 + knobs, per net)"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
import covstep_s2 as s2
n = 1024; eye = torch.eye(n, dtype=torch.bool)
STEP = dict(iso=0.05, k21n=0.05, k31=0.05, k22=0.05, edge=0.1, rc=0.1, bfix=0.02, c0=0.03, c1=0.1)
V3 = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212, 'edge': 0.164, 'rc': -0.080}


def configure(card, th):
    c0, c1 = 0.85 + th.get('c0', 0.0), 0.5 + th.get('c1', 0.0)
    s2.make_s2(card, rowf=(c0, c1), gamma=0.1 + th.get('bfix', 0.0))
    cov_knobs = {k: v for k, v in th.items() if k in ('iso', 'k21n', 'k31', 'k22') and v != 0}
    if cov_knobs:
        def cov(l, C, ctx):
            if l < 2 or l > 14: return C
            mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
            vv = torch.diagonal(Cpre); s = vv.sqrt(); al = mu / s
            kink = (al.abs() < 1.5).double(); km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
            nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
            k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
            k2, k22n, r31 = k21, k22, None
            for k, t in cov_knobs.items():
                if k == 'iso': k2 = k2 + t * km * (k3[:, None] * Cpre / vv[:, None])
                elif k == 'k21n': k2 = k2 + t * nm * k21
                elif k == 'k22': k22n = torch.where(eye, k22n, (1 + t) * k22n)
                elif k == 'k31': r31 = t * ((k4 / vv ** 2) * (c0 + c1 * mu * k3 / (vv * vv)))[:, None] * rho
            o = pair(mu, Cpre, k2, k4, k22n, 'covariance') if r31 is None else pair(mu, Cpre, k2, k4, k22n, 'covariance', remainder31=r31)
            return o[1]
        card.hooks['cov_pm'] = cov
    if th.get('edge', 0): card.hooks['edge'] = lambda l, v, ctx, t=th['edge']: v * (1 + t)
    if th.get('rc', 0): card.knobs['recenter'] = 1 + th['rc']


def final_err(w, truth, th, pl):
    p, a, c, e = pl
    card = cs.HookCard(torch, w, p, a, c, e); configure(card, th)
    with torch.no_grad(): outs, _ = card.run_states()
    return outs[15].clamp_min(0).to(torch.float32).numpy().astype(np.float64) - truth[15]


if __name__ == '__main__':
    mode = sys.argv[1]
    pl = cs.model.torch_payload(torch, np, False)
    if mode == 'eval':
        arg, th, thr, out = sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]); torch.set_num_threads(thr)
        ranks = json.loads((cs.P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'] if arg == 'valid40' else [int(x) for x in arg.split(',')]
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk); t0 = time.time()
            e0 = final_err(w, truth, {}, pl); e1 = final_err(w, truth, th, pl)
            m0, m1 = float((e0 ** 2).mean()), float((e1 ** 2).mean())
            rec = dict(rank=rk, card='S2', theta=th, s2=m0, knobs=m1, ratio=m1 / m0, wall=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            with out.open('a') as f: f.write(json.dumps(rec) + '\n')
    elif mode == 'fit':
        ranks = [int(x) for x in sys.argv[2].split(',')]; knobs = sys.argv[3].split(','); torch.set_num_threads(int(sys.argv[4])); out = sys.argv[5]
        E0, U = {}, {}
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk); t0 = time.time()
            E0[rk] = final_err(w, truth, {}, pl); cols = []
            for k in knobs:
                ek = final_err(w, truth, {k: STEP[k]}, pl); cols.append((ek - E0[rk]) / STEP[k])
                print(json.dumps(dict(rank=rk, knob=k, step=STEP[k], ratio_fd=float((ek ** 2).mean() / (E0[rk] ** 2).mean()))), flush=True)
            U[rk] = np.stack(cols, 1)
            print(f'# rank {rk} done {time.time() - t0:.0f}s', flush=True)
            np.savez(out, knobs=np.array(knobs), ranks=np.array(list(E0)), **{f'e0_{r}': E0[r] for r in E0}, **{f'U_{r}': U[r] for r in U})
