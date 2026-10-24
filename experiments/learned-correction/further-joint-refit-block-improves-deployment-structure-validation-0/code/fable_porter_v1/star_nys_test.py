#!/usr/bin/env python3
"""TAG fable-porter (PT-48, lead 23:5x "Nystrom range finder for the low-rank star"): torch, learned ON, no refit, VALID16 (inputs.json
valid[:16]) on the S23-structure card = fable_covhunter_v1/valid40_dials.py construction with STAR=s1zero JOIN=nystrom GL=1 PS=1
(s6bcard_joint2 + join_nystrom_torch + the physicist's lrstar_s1zero_hook + gn_linear_hook + pair_series), loopR1 params
(s6b_loopR1/params_epoch0.pt), round-1 dials (dials_r1_ep2.json), 3 torch threads. The star's range finder (native _lr_factor), 3 arms:
  ctrl  Q = qr(A Omega) (Omega = the fixed seeded 1024 x 48 Gaussian), one power step Q = qr(A qr(A^T Q)), B = Q^T A, rank-32 SVD truncation
        (= lrstar_s1zero_hook; this versioned copy is certified equal to that hook on the first net: arm 'lz' = the hook itself)
  nys   Q = qr(A[:, cols]), cols = the 48 columns of largest norm (float64 sums of squares, stable argsort, ascending order); no power step
  nysp  the same columns, then the power step as in ctrl
Each run also records the energy the kept rank-32 plane captures, ||Qr^T A||^2 / ||A||^2, per star call (mean and min over the net's calls).
usage: star_nys_test.py <arms e.g. ctrl,nys,nysp> <out.jsonl>   (resumable; the first net also runs 'lz' for the certificate)"""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; CV = P2 / 'fable_covhunter_v1'
sys.path.insert(0, str(CV)); sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_engineer_v1'))
sys.path.insert(2, str(P2 / 'fable_physicist_v1/refit6b'))
import torch
import s6bcard_joint2 as SJ, gn_linear_hook as gl, pair_series_torch as ps, join_nystrom_torch as jn, lrstar_s1zero_hook as lz
torch.set_num_threads(3); torch.set_grad_enabled(False)
K, QS, R, SEED = 48, 1, 32, 20260924
PARAMS = P2 / 'fable_physicist_v1/refit6b/s6b_loopR1/params_epoch0.pt'; DIALS = json.load(open(CV / 'dials_r1_ep2.json'))
assert lz.S1 == 'zero' and (lz.K, lz.Q_STEPS, lz.R, lz.SEED) == (K, QS, R, SEED)


def install_star(card, t, rf, stats):
    om_np = np.random.default_rng(SEED).standard_normal((1024, K), dtype=np.float32)
    om = {}

    def basis(A32):
        with t.no_grad():
            Ad = A32.detach()
            if rf == 'gauss':
                o = om.get(Ad.dtype)
                if o is None:
                    o = om[Ad.dtype] = t.from_numpy(om_np).to(Ad.dtype)
                Q = t.linalg.qr(Ad @ o)[0]
            else:
                cn = (Ad.to(t.float64) ** 2).sum(0)
                cols = t.sort(t.argsort(-cn, stable=True)[:K]).values
                Q = t.linalg.qr(Ad[:, cols])[0]
            for _ in range(QS if rf in ('gauss', 'nysp') else 0):
                Q = t.linalg.qr(Ad @ t.linalg.qr(Ad.T @ Q)[0])[0]
            U = t.linalg.svd(Q.T @ Ad, full_matrices=False)[0]
            Qr = Q @ U[:, :R]
            stats.append(float((Qr.T @ Ad).to(t.float64).pow(2).sum() / Ad.to(t.float64).pow(2).sum()))
            return Qr

    def star(w, A):   # = lrstar_s1zero_hook.star with S1 = 'zero'
        wt = w.T
        A32 = A if A.dtype == t.float32 else A.to(t.float32)
        Qr = basis(A32)
        Br = Qr.T @ A32
        P1 = wt @ Qr
        pw = (P1 @ Br) * wt
        sub = (((wt * wt) @ Br.T) @ P1.T) / 3
        diagonal = (pw * wt).sum(dim=1, dtype=t.float64)
        result = sub.to(t.float64).clone()
        result.diagonal().copy_(diagonal)
        return result

    def star_diagonal(w, A):
        wt = w.T
        A32 = A.to(t.float32)
        Qr = basis(A32)
        pushed = (wt @ Qr) @ (Qr.T @ A32)
        return ((pushed * wt) * wt).sum(dim=1, dtype=t.float64)

    card.env['_flms_star'] = star
    card.env['_flms_star_diagonal'] = star_diagonal
    return card


def literals(path):
    sv = torch.load(path); p, a, c, e = SJ.payload(torch)
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
    return p, a, c, e


def run(arm, lit, w, tp):
    SJ.DIALS = dict(DIALS); cd = SJ.card(torch, w, *lit)
    jn.install(cd, torch)
    stats = []
    if arm == 'lz': lz.install(cd, torch)
    else: install_star(cd, torch, {'ctrl': 'gauss', 'nys': 'nys', 'nysp': 'nysp'}[arm], stats)
    gl.install(cd, torch); ps.install(cd, torch)
    t0 = time.time(); mse = float(((cd.run(False)[15].to(torch.float64) - tp) ** 2).mean())
    return dict(mse=mse, e_mean=float(np.mean(stats)) if stats else None, e_min=float(np.min(stats)) if stats else None, calls=len(stats), wall=round(time.time() - t0, 1))


if __name__ == '__main__':
    arms = sys.argv[1].split(','); OUT = HERE / sys.argv[2]
    nets = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:16]
    done = {(r['net'], r['arm']) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    print(json.dumps(dict(TAG='fable-porter PT-48 star Nystrom range finder, torch VALID16, S23-structure card (joint2 + nystrom join + s1=0 star + '
                          'gn_linear_hook + pair_series), loopR1 params, round-1 dials, learned ON, no refit', arms=arms, nets=nets, resume=len(done))), flush=True)
    lit = literals(PARAMS)
    for i, r in enumerate(nets):
        w = SJ.weights(r); tp = torch.tensor(SJ.truthpost(r)[15]).to(torch.float64)
        for arm in (['lz'] if i == 0 else []) + arms:
            if (r, arm) in done: continue
            row = dict(net=r, arm=arm, **run(arm, lit, w, tp))
            with open(OUT, 'a') as f: f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
