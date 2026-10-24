#!/usr/bin/env python3
"""TAG fable-covhunter (CH-32): GN RE-FIT of the five covariance knobs (deviations from v3) on the S6b physics (fired #332005), in torch:
S2 physics (K31 row .85/.5 L1-14, gain b-fix .1, every3 joins = join v2's predictions) + the v3 knobs + the L15 k3 row cut
(zc 3.5: bank and star parts of k3_15 only on the select(zc - |mu/sd|, 0) rows, rounded up to 32; physicist's term32) + the closure
refit literals (fable_analyst_v1/refit_s2_full_lr2e4_v2/params_epoch2.pt: p, a, c, e). Kernel-level changes of S5e/S6b (push5, f32
third, memory diet) are rounding-level and absent here.
Knobs (global deviations): iso, k21n, k31 (covariance output only, L2-14), edge (L8-14 message edge scale), rc (recentering scale).
usage: gn7_s6b.py fit <ranks> <threads> <out.npz> | gn7_s6b.py eval <ranks|valid40> <dev json> <threads> <out.jsonl> | gn7_s6b.py parity <ranks>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
import covstep_s2 as s2
n = 1024; eye = torch.eye(n, dtype=torch.bool)
V3 = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212, 'edge': 0.164, 'rc': -0.080}
KNOBS = ['iso', 'k21n', 'k31', 'edge', 'rc']
STEP = dict(iso=0.05, k21n=0.05, k31=0.05, edge=0.1, rc=0.1)
FIRST = dict(iso=2, k21n=2, k31=2, edge=8, rc=1)
REFIT = cs.P2 / 'fable_analyst_v1/refit_s2_full_lr2e4_v2/params_epoch2.pt'
ZC = 3.5


def payload():
    d = torch.load(REFIT, map_location='cpu')
    p = {k: v.detach().to(torch.float32).clone() for k, v in d['p'].items()}
    e = {k: v.detach().to(torch.float32).clone() for k, v in d['e'].items()}
    return p, d['a'].detach().to(torch.float32).clone(), d['c'].detach().to(torch.float32).clone(), e


def configure(card, dev, cut=True):
    s2.make_s2(card)
    iso, k21n, k31 = (V3[k] + dev.get(k, 0.0) for k in ('iso', 'k21n', 'k31'))
    def cov(l, C, ctx):
        if l < 2 or l > 14: return C
        mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
        vv = torch.diagonal(Cpre); s = vv.sqrt(); kink = ((mu / s).abs() < 1.5).double()
        km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
        k2 = k21 + iso * km * (k3[:, None] * Cpre / vv[:, None]) + k21n * nm * k21
        r31 = k31 * ((k4 / vv ** 2) * (0.85 + 0.5 * mu * k3 / (vv * vv)))[:, None] * rho
        return pair(mu, Cpre, k2, k4, k22, 'covariance', remainder31=r31)[1]
    card.hooks['cov_pm'] = cov
    card.hooks['edge'] = lambda l, v, ctx: v * (1 + V3['edge'] + dev.get('edge', 0.0))
    card.knobs['recenter'] = 1 + V3['rc'] + dev.get('rc', 0.0)
    if cut:
        def k3cut(l, k3, ctx):
            score = ZC - (ctx['mu'] / ctx['vv'].sqrt()).abs()
            count = min(1024, ((int((score >= 0).sum()) + 31) // 32) * 32)
            keep = torch.zeros_like(score); keep[torch.argsort(-score, stable=True)[:count]] = 1.0
            return keep * (ctx['bank'] + ctx['star']) + ctx['bterm']
        card.hooks['k3_15'] = k3cut


def base_states(w, pl, dev=None):
    card = cs.HookCard(torch, w, *pl); configure(card, dev or {})
    keep = {}
    card.hooks['state_in'] = lambda l, st, ctx: (keep.__setitem__(l, st) if l in (1, 2, 8) else None) or st
    with torch.no_grad():
        st = card.initial()
        for l in range(16): st = card.step(l, st)
    del card.hooks['state_in']
    return keep, st[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64)


def run_from(w, pl, dev, states, start, truth):
    card = cs.HookCard(torch, w, *pl); configure(card, dev)
    with torch.no_grad():
        st = states[start]
        for l in range(start, 16): st = card.step(l, st)
    return st[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64) - truth[15]


if __name__ == '__main__':
    mode = sys.argv[1]; pl = payload()
    if mode == 'parity':
        torch.set_num_threads(3)
        for rk in [int(x) for x in sys.argv[2].split(',')]:
            w = cs.weights(rk); truth = cs.truthpost(rk)
            _, pred = base_states(w, pl); print(json.dumps(dict(rank=rk, torch_S6b_mse=float(((pred - truth[15]) ** 2).mean()))), flush=True)
    elif mode == 'fit':
        ranks = [int(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3])); out = sys.argv[4]
        E0, U = {}, {}
        for rk in ranks:
            t0 = time.time(); w = cs.weights(rk); truth = cs.truthpost(rk)
            states, pred = base_states(w, pl); E0[rk] = pred - truth[15]
            U[rk] = np.stack([(run_from(w, pl, {k: STEP[k]}, states, FIRST[k], truth) - E0[rk]) / STEP[k] for k in KNOBS], 1)
            print(json.dumps(dict(rank=rk, mse=float((E0[rk] ** 2).mean()), wall=round(time.time() - t0, 1))), flush=True)
            np.savez(out, knobs=np.array(KNOBS), ranks=np.array(list(E0)), steps=np.array([STEP[k] for k in KNOBS]),
                     **{f'e0_{r}': E0[r] for r in E0}, **{f'U_{r}': U[r] for r in U})
    else:
        arg, dev, thr, out = sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]); torch.set_num_threads(thr)
        ranks = json.loads((cs.P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'] if arg == 'valid40' else [int(x) for x in arg.split(',')]
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk)
            states, pred = base_states(w, pl); e0 = pred - truth[15]; e1 = run_from(w, pl, dev, states, 1, truth)
            rec = dict(rank=rk, card='S6b', dev=dev, base=float((e0 ** 2).mean()), knobs=float((e1 ** 2).mean())); rec['ratio'] = rec['knobs'] / rec['base']
            print(json.dumps(rec), flush=True)
            with out.open('a') as f: f.write(json.dumps(rec) + '\n')
