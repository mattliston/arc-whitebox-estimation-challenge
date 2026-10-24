"""TAG fable-covhunter (CH-34): the S6b physics as a torch research card with the SAME interface as fable_analyst_v1/s2card.py, for the
refit drivers: card(torch, w, p, a, c, e, fine_ckpt=False) -> a HookCard (covstep.py) with the S2 physics (K31 row .85/.5 L1-14, gain
b-fix .1, every3 joins), the v3 covariance knobs (covariance-only kink K21 += -.094 k3 C/v, non-kink K21 x.972, T31 x.788 at L2-14; edge
x1.164; recentering x.92) and the L15 k3 row cut (zc 3.5). The refit literals are NOT built in: pass them as p, a, c, e (e.g. the driver's
--resume params_epoch2.pt), exactly as refit_s2.py does. Differentiable end to end (the row selections are the card's own discrete
selects); fine_ckpt=True wraps the pair map, _bd_update, the star push, the tier push and the plane read in nested non-reentrant
checkpoints (memory only: identical values and gradients), as in s2card.py.
Parity (bank params + epoch-2 refit literals) vs the fired native S6b #332005: r35 -0.07%, r100 +0.20% (gn7_s6b.py parity).
Self-test: python s6bcard_ch.py <rank> <threads> [groups heads|full|fullP] -> forward MSE, fwd+bwd wall, grad norms, finiteness, peak RSS."""
import sys, time, resource
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import covstep as cs
import gn7_s6b as g7
import s2card
model = cs.model                       # model_bank_research: torch_payload / parameters identical to s2card.model's


def card(torch, w, p, a, c, e, fine_ckpt=False):
    cd = cs.HookCard(torch, w, p, a, c, e)
    g7.configure(cd, {})                # S2 physics + v3 knobs + L15 row cut
    if fine_ckpt:
        for k in ('_nonlinear_pair_batch', '_bd_update', '_flms_star'): cd.env[k] = s2card._ckpt(torch, cd.env[k])
        for k in ('push', 'read'): cd.shared[k] = s2card._ckpt(torch, cd.shared[k])
    return cd


if __name__ == '__main__':
    import numpy as np, torch
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35; torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    groups = sys.argv[3] if len(sys.argv) > 3 else 'full'
    p, a, c, e = g7.payload()           # bank + epoch-2 refit literals = S6b
    params = [p['head'], p['head_b'], e['output'], e['bias']]
    if groups in ('full', 'fullP'): params += [a, c] + [e[k] for k in e if k not in ('output', 'bias')]
    if groups == 'fullP': params += [p[k] for k in ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')]
    for x in params: x.requires_grad_(True)
    w = cs.weights(rank); truth = torch.tensor(cs.truthpost(rank)[15])
    with torch.no_grad(): mse0 = float(((card(torch, w, p, a, c, e).run(False)[15].to(torch.float64) - truth) ** 2).mean())
    t0 = time.time(); cd = card(torch, w, p, a, c, e, fine_ckpt=True)
    L = ((cd.run(True)[15].to(torch.float64) - truth) ** 2).mean(); t1 = time.time(); (L * 1e8).backward(); t2 = time.time()
    gn = [float(x.grad.norm()) if x.grad is not None else None for x in params]
    print(dict(rank=rank, groups=groups, mse_nograd=mse0, mse_grad=float(L), fwd_s=round(t1 - t0, 1), bwd_s=round(t2 - t1, 1),
               n_params=int(sum(x.numel() for x in params)), grads_finite=all(g is not None and np.isfinite(g) for g in gn),
               grad_norm_min=min(g for g in gn if g is not None), grad_norm_max=max(g for g in gn if g is not None),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)))
