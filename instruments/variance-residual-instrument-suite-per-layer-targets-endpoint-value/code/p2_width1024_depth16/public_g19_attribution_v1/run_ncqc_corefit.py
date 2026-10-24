#!/usr/bin/env python3
"""TAG ncqc-corefit-v1: fit a compact quadratic-chaos core (a_i, M_i)
on a unit sub-block's (C, K21) from banked engine planes; gates:
(1) held-out-entry K21 reproduction, (2) VALUE retention of the
core-reproduced plane vs the engine plane through the C1 adjoint
sub-block. Model: C_ij = a_i.a_j + 0.5 tr(M_i M_j);
K21_ij = tr(M_i^2 M_j) + a_i^T M_j a_i + 2 a_i^T M_i a_j.
Usage: run_ncqc_corefit.py <rank> <layer> [r=24] [ns=256]"""
import json
import sys
import numpy as np
import torch

rank = int(sys.argv[1]); L = int(sys.argv[2])
r = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ns = int(sys.argv[4]) if len(sys.argv) > 4 else 256
dev = torch.device("mps")
T = np.load(f"mps_cov2_rank{rank:04d}.npz")
tp = np.load(f"tensor_planes_rank{rank:04d}.npz")
z = np.load(f"host_cov_rank{rank:04d}_base.npz")
rng = np.random.default_rng(11)
sel = np.sort(rng.choice(1024, size=ns, replace=False))
C = torch.tensor(T["mc_pre_cov"][L][np.ix_(sel, sel)], dtype=torch.float32, device=dev)
K21 = torch.tensor(tp[f"pre_{L}"][np.ix_(sel, sel)], dtype=torch.float32, device=dev)
sC = float(C.pow(2).mean().sqrt()); sK = float(K21.pow(2).mean().sqrt())
# holdout mask on entries (20%)
mask = torch.tensor(rng.random((ns, ns)) < 0.8, device=dev)
a = torch.randn(ns, r, device=dev) * 0.05
Mh = torch.randn(ns, r, r, device=dev) * 0.02
a.requires_grad_(); Mh.requires_grad_()
opt = torch.optim.Adam([a, Mh], lr=3e-3)
for it in range(4000):
    M = 0.5 * (Mh + Mh.transpose(1, 2))
    Cf = a @ a.T + 0.5 * torch.einsum("iab,jab->ij", M, M)
    M2 = torch.einsum("iab,ibc->iac", M, M)
    t1 = torch.einsum("iab,jab->ij", M2, M)
    t2 = torch.einsum("ia,jab,ib->ij", a, M, a)
    t3 = 2.0 * torch.einsum("ia,iab,jb->ij", a, M, a)
    Kf = t1 + t2 + t3
    loss = ((Cf - C).pow(2) * mask).mean() / sC**2 \
        + ((Kf - K21).pow(2) * mask).mean() / sK**2
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 1000 == 0:
        print(f"it {it} loss {float(loss):.4e}")
with torch.no_grad():
    M = 0.5 * (Mh + Mh.transpose(1, 2))
    Cf = a @ a.T + 0.5 * torch.einsum("iab,jab->ij", M, M)
    M2 = torch.einsum("iab,ibc->iac", M, M)
    Kf = (torch.einsum("iab,jab->ij", M2, M)
          + torch.einsum("ia,jab,ib->ij", a, M, a)
          + 2.0 * torch.einsum("ia,iab,jb->ij", a, M, a))
    hm = ~mask
    def cc(X, Y, m):
        x = X[m]; y = Y[m]
        return float((x*y).sum() / (x.pow(2).sum().sqrt()*y.pow(2).sum().sqrt()+1e-30))
    print(json.dumps({
        "heldout_C_corr": round(cc(Cf, C, hm), 4),
        "heldout_K21_corr": round(cc(Kf, K21, hm), 4),
        "train_K21_corr": round(cc(Kf, K21, mask), 4)}))
    np.savez(f"ncqc_corefit_rank{rank:04d}_L{L}_r{r}.npz",
             sel=sel, a=a.cpu().numpy(), M=M.cpu().numpy(),
             K21_fit=Kf.cpu().numpy())
