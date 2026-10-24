#!/usr/bin/env python3
"""TAG fable-covhunter (CH-29): L14-15-ONLY zero-bill knobs on the S2 card + v3 covariance knobs (= fired S4 physics). The state entering
L14 is cached per net, so each knob evaluation reruns only steps 14 and 15 (the knobs cannot touch L0-13). Knobs (global strengths):
  k3b / k3s / k3t : final-layer k3_15 parts x (1+th): bank columns, star diagonal, b term 3 v b
  k4   : k4_15 (= radial v^2) x (1+th)          v15 : v_15 x (1+th), all units        v15k : v_15 x (1+th), kink units (|alpha|<1.5)
  dl   : final GNN delta x (1+th)
  iso14 / k31_14 / k21n14 : extra covariance-only knobs at L14 only (added to v3's), feeding v_15
Fit: one finite-difference run per knob (quadratic model, gn_select.py). Eval: S2+v3 vs S2+v3+final knobs.
usage: gn5_final.py fit <ranks> <knobs> <threads> <out.npz> | gn5_final.py eval <ranks|valid40> <theta json> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
import gn3_knobs as g3
n = 1024; eye = torch.eye(n, dtype=torch.bool)
STEP = dict(k3b=0.05, k3s=0.05, k3t=0.05, k4=0.05, v15=0.002, v15k=0.002, dl=0.1, iso14=0.05, k31_14=0.05, k21n14=0.05)
V3 = g3.V3


def make_card(w, pl):
    card = cs.HookCard(torch, w, *pl); g3.configure(card, V3)
    return card


def set_final(card, th):
    """install the L14-15 knobs on a configured card (on top of v3); th may be empty."""
    base_cov = card.hooks.get('cov_pm')
    extra14 = {k: th.get(k, 0.0) for k in ('iso14', 'k31_14', 'k21n14')}
    if any(extra14.values()):
        def cov(l, C, ctx):
            if l != 14: return base_cov(l, C, ctx)
            mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
            vv = torch.diagonal(Cpre); s = vv.sqrt(); kink = ((mu / s).abs() < 1.5).double()
            km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
            nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
            k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
            k2 = k21 + (V3['iso'] + extra14['iso14']) * km * (k3[:, None] * Cpre / vv[:, None]) + (V3['k21n'] + extra14['k21n14']) * nm * k21
            r31 = (V3['k31'] + extra14['k31_14']) * ((k4 / vv ** 2) * (0.85 + 0.5 * mu * k3 / (vv * vv)))[:, None] * rho
            return pair(mu, Cpre, k2, k4, k22, 'covariance', remainder31=r31)[1]
        card.hooks['cov_pm'] = cov
    if th.get('v15', 0) or th.get('v15k', 0):
        def vh(l, vv, ctx, a=th.get('v15', 0.0), b=th.get('v15k', 0.0)):
            return vv * (1 + a) if not b else vv * (1 + a + b * (((card._mu15 / vv.sqrt()).abs() < 1.5).double()))
        card.hooks['vv15'] = vh
        card.hooks['mu'] = lambda l, mu, ctx: (setattr(card, '_mu15', mu) or mu) if l == 15 else mu
    if th.get('k3b', 0) or th.get('k3s', 0) or th.get('k3t', 0):
        card.hooks['k3_15'] = lambda l, k3, ctx: ctx['bank'] * (1 + th.get('k3b', 0.0)) + ctx['star'] * (1 + th.get('k3s', 0.0)) + ctx['bterm'] * (1 + th.get('k3t', 0.0))
    if th.get('k4', 0): card.hooks['k4_15'] = lambda l, k4, ctx: k4 * (1 + th['k4'])
    if th.get('dl', 0): card.hooks['delta'] = lambda l, d, ctx: d * (1 + th['dl']) if l == 15 else d


def cached_state(card):
    keep = {}
    card.hooks['state_in'] = lambda l, st, ctx: (keep.__setitem__('st', st) if l == 14 else None) or st
    with torch.no_grad():
        st = card.initial()
        for l in range(14): st = card.step(l, st)
    del card.hooks['state_in']
    return st


def final_err(card, st14, truth, th):
    saved = dict(card.hooks); knobs = dict(card.knobs)
    set_final(card, th)
    with torch.no_grad():
        st = card.step(14, st14); st = card.step(15, st)
    card.hooks.clear(); card.hooks.update(saved); card.knobs.update(knobs)
    return st[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64) - truth[15]


if __name__ == '__main__':
    mode = sys.argv[1]; pl = cs.model.torch_payload(torch, np, False)
    if mode == 'fit':
        ranks = [int(x) for x in sys.argv[2].split(',')]; knobs = sys.argv[3].split(','); torch.set_num_threads(int(sys.argv[4])); out = sys.argv[5]
        E0, U = {}, {}
        for rk in ranks:
            t0 = time.time(); w = cs.weights(rk); truth = cs.truthpost(rk); card = make_card(w, pl); st14 = cached_state(card)
            E0[rk] = final_err(card, st14, truth, {}); cols = []
            for k in knobs:
                ek = final_err(card, st14, truth, {k: STEP[k]}); cols.append((ek - E0[rk]) / STEP[k])
            U[rk] = np.stack(cols, 1)
            print(json.dumps(dict(rank=rk, mse=float((E0[rk] ** 2).mean()), fd={k: round(float(((E0[rk] + STEP[k] * U[rk][:, i]) ** 2).mean() / (E0[rk] ** 2).mean()), 5) for i, k in enumerate(knobs)}, wall=round(time.time() - t0, 1))), flush=True)
            np.savez(out, knobs=np.array(knobs), ranks=np.array(list(E0)), **{f'e0_{r}': E0[r] for r in E0}, **{f'U_{r}': U[r] for r in U})
    else:
        arg, th, thr, out = sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]); torch.set_num_threads(thr)
        ranks = json.loads((cs.P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'] if arg == 'valid40' else [int(x) for x in arg.split(',')]
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk); card = make_card(w, pl); st14 = cached_state(card)
            e0 = final_err(card, st14, truth, {}); e1 = final_err(card, st14, truth, th)
            rec = dict(rank=rk, card='S2+v3', theta=th, base=float((e0 ** 2).mean()), knobs=float((e1 ** 2).mean())); rec['ratio'] = rec['knobs'] / rec['base']
            print(json.dumps(rec), flush=True)
            with out.open('a') as f: f.write(json.dumps(rec) + '\n')
