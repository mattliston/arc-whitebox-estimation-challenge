#!/usr/bin/env python3
"""TAG fable-analyst (AN-16): ZERO-COST row factors for the radial-rebuilt per-unit k4 and K22 plane, closed loop in the torch card
(learned parts ON, no refit), through the physicist's harness (fable_physicist_v1/model_bank_phys.PHYS_SUPPLY 'fn').
q_a = mu_a k3_a / v_a^2 (card's own mu, diag K21, v at the pre-activation). Terms (joined with '+'), each applied at layers L0..14:
  k4:<c0>:<c1>:<t31>@L0   per-unit k4 = radial v^2 (c0 + c1 q) in the pair map AND the final map (layer 15, via nonlinear_scalar);
                          t31 = hold (K31 kept at the card's t4*rho) | follow (K31 = k4'/v * C, the card's regression form) | phys
                          (K31 = the physicist's row factor 0.85 + 0.5 q)
  k22r:<c0>:<c1>@L0       K22 off-diagonal x sqrt(r_a r_c), r = c0 + c1 q (clamped >= 0.05)
  k22g:<g>@L0             K22 off-diagonal += g (radial/2)(mu_a K21_ca + mu_c K21_ac)   (gain-on-skewed-base term, card's K21)
  phys31@L0               K31 = physicist's per-row factor (0.85 + 0.5 q) alone (reference)
  suffix |a/b             channel split (physicist harness): keep only outputs a,b (cov, marg, target, radial) of the supplied pair-map call,
                          the rest from the unsupplied call (e.g. |cov = the supply acts only through the post covariance)
usage: rowfactor_closedloop.py <ranks> <specs> <threads> [--tag T]"""
import sys, json, time, gc
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_physicist_v1'))
import numpy as np, torch
import run_supply as rs
ranks = [int(x) for x in sys.argv[1].split(',')]; specs = sys.argv[2].split(',')
TAG = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--tag=')]; TAG = TAG[0] if TAG else 'rowfactor'
def parse(spec):
    cfg = dict(k4=None, t31='hold', k22r=None, k22g=None, phys31=None, split=None)
    if spec == 'base': return cfg
    if '|' in spec:
        spec, sp = spec.split('|'); cfg['split'] = sp.split('/')
    for term in spec.split('+'):
        body, L0 = term.split('@'); L0 = int(L0); f = body.split(':')
        if f[0] == 'k4': cfg['k4'] = (float(f[1]), float(f[2]), L0); cfg['t31'] = f[3]
        elif f[0] == 'k22r': cfg['k22r'] = (float(f[1]), float(f[2]), L0)
        elif f[0] == 'k22g': cfg['k22g'] = (float(f[1]), L0)
        elif f[0] == 'phys31': cfg['phys31'] = L0
        else: raise ValueError(term)
    return cfg
def make_fn(cfg):
    def fn(card, layer, mu, Cpre, vv, radial, k21, k4, k22, parts):
        t = card.t; n = vv.shape[0]; eye = torch.eye(n, dtype=torch.bool)
        k3 = torch.diagonal(k21); q = mu * k3 / (vv * vv); s = vv.sqrt()
        k4n = k4; t4pair = (k4 / (vv * vv)) if isinstance(k4, torch.Tensor) and k4.dim() == 1 else radial * torch.ones_like(vv)
        t4_card = t4pair.clone()
        if cfg['k4'] and layer >= cfg['k4'][2]:
            c0, c1, _ = cfg['k4']; r4 = (c0 + c1 * q).clamp_min(0.05)
            k4n = radial * vv * vv * r4; t4pair = radial * r4
        # desired per-row K31 scale (T31 = t4row[:,None] * rho); the pair map builds t4pair*rho from the k4 it receives
        t4row = t4pair if cfg['t31'] == 'follow' else t4_card
        if (cfg['k4'] and cfg['t31'] == 'phys' and layer >= cfg['k4'][2]) or (cfg['phys31'] is not None and layer >= cfg['phys31']):
            t4row = radial * (0.85 + 0.5 * q)
        r31 = None
        if not torch.equal(t4row, t4pair):
            rho = (Cpre / (s[:, None] * s[None, :])).masked_fill(eye, 0.0)
            r31 = ((t4row - t4pair)[:, None] * rho).to(torch.float32)
        k22n = k22
        if cfg['k22r'] and layer >= cfg['k22r'][2]:
            c0, c1, _ = cfg['k22r']; r = (c0 + c1 * q).clamp_min(0.05); sr = r.sqrt()
            k22n = torch.where(eye, k22n, k22n * sr[:, None] * sr[None, :])
        if cfg['k22g'] and layer >= cfg['k22g'][1]:
            g = cfg['k22g'][0]; G = (radial / 2) * (mu[:, None] * k21.T + mu[None, :] * k21)
            k22n = torch.where(eye, k22n, k22n + g * G)
        return k21, k4n, k22n, r31
    return fn
res = []
for rank in ranks:
    w = np.load(rs.STORE / Path(rs.man[rank]['path']).parent / 'weights.npy'); base = None
    has_truth = (rs.G19 / f'mps_truth_rank{rank:04d}_P32M.npz').exists() or (rs.G19 / f'mps_truth_rank{rank:04d}_P8M.npz').exists()
    for spec in specs:
        cfg = parse(spec)
        rs.model.PHYS_SUPPLY = None if spec == 'base' else {l: ({'fn': make_fn(cfg), 'split': cfg['split']} if cfg['split'] else {'fn': make_fn(cfg)}) for l in range(1, 15)}
        cap = {}; rs.model.PHYS_CAPTURE = cap if has_truth else None
        card = rs.model.Card(torch, w, rs.p, rs.a, rs.c, rs.e)
        if cfg['k4']:
            c0, c1, _ = cfg['k4']; orig_s = card.env['nonlinear_scalar']
            def scal(mu, v, k3, k4, orig_s=orig_s, c0=c0, c1=c1):
                return orig_s(mu, v, k3, k4 * (c0 + c1 * mu * k3 / (v * v)).clamp_min(0.05))
            card.env['nonlinear_scalar'] = scal
        t0 = time.time()
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
        fin = float(((pred[15] - TP[15]) ** 2).mean())
        if spec == 'base': base = fin
        rec = dict(rank=rank, spec=spec, final=fin, rel=(fin / base - 1) if base else None, wall=round(time.time() - t0, 1))
        if has_truth:
            g = rs.grade(rank, pred, cap); rec['v15'] = round(g['v_rel'][15], 5); rec['k3_15'] = round(g['k3_rel'][15], 4)
        print(json.dumps(rec), flush=True); res.append(rec)
        del card, cap; gc.collect()
with open(HERE / f'{TAG}.jsonl', 'a') as fh:
    for r in res: fh.write(json.dumps(r) + '\n')
