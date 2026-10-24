#!/usr/bin/env python3
"""TAG fable-analyst (AN-39b): does the (0,1,0) feed point along the S2 card's OWN third-order error? Base S2 torch card (learned ON), open
loop: at each read layer L = 2..14, the feed from the card's pre-state at L-1 (feed_s2.feed, imported formulas) against
  k3:    dk3 = 32M truth pre k3 (public_g19 mps_truth_rank*_P32M mc_pre_k3[L]) - card k3 (diag of the card's pre-plane at L)
  plane: dK = 8M truth pre-plane (mps_planes_rank*.npz K21_L) - card pre-plane at L, off-diagonal entries.
Per layer: beta (least-squares slope of the error on the feed), corr, energy explained at strength 1 and 0.5, and the truth-noise share of the
k3 error energy (Gaussian estimate 6 v^3 / N, N = 32M). usage: feed_s2_diag.py <ranks> <threads> <out.json>"""
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
def st(x, y):
    xx, xy, yy = float(x @ x), float(x @ y), float(y @ y)
    return dict(beta=round(xy / xx, 4), corr=round(xy / np.sqrt(xx * yy), 4), expl_1=round(1 - float((y - x) @ (y - x)) / yy, 4),
                expl_05=round(1 - float((y - .5 * x) @ (y - .5 * x)) / yy, 4), rms_feed=float(np.sqrt(xx / x.size)), rms_err=float(np.sqrt(yy / y.size)))
out = {}
for rank in ranks:
    w = np.load(mod.STORE / Path(mod.man[rank]['path']).parent / 'weights.npy')
    card = mod.s2card.card(torch, w, mod.p, mod.a, mod.c, mod.e); cap = {}
    orig = card.env['_nonlinear_pair_batch']
    def pair(mu, C, K21, *args, **kw):
        cap[card.call_layer] = (mu.detach().clone(), C.detach().clone(), K21.detach().clone()); return orig(mu, C, K21, *args, **kw)
    card.env['_nonlinear_pair_batch'] = pair
    with torch.no_grad(): card.run(False)
    tr = np.load(G19 / f'mps_truth_rank{rank:04d}_P32M.npz'); pl = np.load(G19 / f'mps_planes_rank{rank:04d}.npz')
    off = ~np.eye(1024, dtype=bool); rec = {}
    for L in range(2, 15):
        mu, C, K = cap[L - 1]
        with torch.no_grad(): P, _ = mod.feed('plane', card.ws[L], mu, C, K)
        P = P.numpy(); Kc = cap[L][2].numpy()
        dk3 = tr['mc_pre_k3'][L].astype(np.float64) - np.diag(Kc)
        noise = float(np.mean(6 * tr['mc_pre_v'][L].astype(np.float64) ** 3 / 32e6) / np.mean(dk3 ** 2))
        dK = pl[f'K21_{L}'].astype(np.float64) - Kc
        rec[L] = dict(k3=st(np.diag(P), dk3), k3_truth_noise_share=round(noise, 3), plane_off=st(P[off], dK[off]))
        print(json.dumps({'rank': rank, 'L': L, 'k3': rec[L]['k3'], 'noise': rec[L]['k3_truth_noise_share'], 'plane': rec[L]['plane_off']}), flush=True)
    out[str(rank)] = rec
    OUT.write_text(json.dumps(out, indent=1))
