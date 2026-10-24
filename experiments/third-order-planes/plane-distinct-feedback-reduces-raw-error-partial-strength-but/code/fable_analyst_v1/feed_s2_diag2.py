#!/usr/bin/env python3
"""TAG fable-analyst (AN-39c): why is the (0,1,0) feed's off-diagonal plane read ANTI-aligned with the S2 card's plane error at L4-14?
The S2 off-diagonal pre-plane is 27-44% too small (amplitude defect). Joint least squares on the off-diagonal entries per read layer L:
  dK = truth8M - card  ~  bP * P + bK * card        (P = feed plane read from the card's pre-state at L-1, open loop, base card)
plus corr(P, truth), corr(P, card). If bP turns positive once the amplitude defect (bK) is absorbed, the feed is real physics masked by it.
usage: feed_s2_diag2.py <ranks> <threads> <out.json>"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
sys.argv, _argv = [sys.argv[0], '35', 'base', sys.argv[2], '/dev/null'], sys.argv   # feed_s2 parses argv at import
src = (HERE / 'feed_s2.py').read_text().split('with OUT.open')[0]                    # its definitions only, not its run loop
mod = type(sys)('feed_s2_mod'); mod.__file__ = str(HERE / 'feed_s2.py'); exec(compile(src, str(HERE / 'feed_s2.py'), 'exec'), mod.__dict__)
sys.argv = _argv
ranks = [int(x) for x in sys.argv[1].split(',')]; torch.set_num_threads(int(sys.argv[2])); OUT = HERE / sys.argv[3]
G19 = P2 / 'public_g19_attribution_v1'
corr = lambda x, y: float(x @ y / np.sqrt((x @ x) * (y @ y)))
out = {}
for rank in ranks:
    w = np.load(mod.STORE / Path(mod.man[rank]['path']).parent / 'weights.npy')
    card = mod.s2card.card(torch, w, mod.p, mod.a, mod.c, mod.e); cap = {}
    orig = card.env['_nonlinear_pair_batch']
    def pair(mu, C, K21, *args, **kw):
        cap[card.call_layer] = (mu.detach().clone(), C.detach().clone(), K21.detach().clone()); return orig(mu, C, K21, *args, **kw)
    card.env['_nonlinear_pair_batch'] = pair
    with torch.no_grad(): card.run(False)
    pl = np.load(G19 / f'mps_planes_rank{rank:04d}.npz'); off = ~np.eye(1024, dtype=bool); rec = {}
    for L in range(2, 15):
        with torch.no_grad(): P, _ = mod.feed('plane', card.ws[L], *cap[L - 1])
        p, k, t = P.numpy()[off], cap[L][2].numpy()[off], pl[f'K21_{L}'].astype(np.float64)[off]
        r = t - k; X = np.stack([p, k], 1); coef = np.linalg.lstsq(X, r, rcond=None)[0]
        e0 = float(r @ r); eK = float(((r - (r @ k) / (k @ k) * k) ** 2).sum()); eJ = float(((r - X @ coef) ** 2).sum())
        rec[L] = dict(bP_joint=round(float(coef[0]), 3), bK_joint=round(float(coef[1]), 3), bP_alone=round(float(p @ r / (p @ p)), 3),
                      corr_P_truth=round(corr(p, t), 4), corr_P_card=round(corr(p, k), 4), corr_P_resid_after_K=round(corr(p, r - (r @ k) / (k @ k) * k), 4),
                      expl_K_only=round(1 - eK / e0, 4), expl_joint=round(1 - eJ / e0, 4), rms_P_over_truth=round(float(np.sqrt((p @ p) / (t @ t))), 4))
        print(json.dumps({'rank': rank, 'L': L, **rec[L]}), flush=True)
    out[str(rank)] = rec
    OUT.write_text(json.dumps(out, indent=1))
