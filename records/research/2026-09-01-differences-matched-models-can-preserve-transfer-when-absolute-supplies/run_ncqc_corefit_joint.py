#!/usr/bin/env python3
"""TAG ncqc-corefit-JOINT-v1 (adds certified K22/K31 losses, pair-batched)
Base: ncqc-corefit-v1: fit a compact quadratic-chaos core (a_i, M_i)
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
tp = np.load(f"mps_planes_rank{rank:04d}.npz")
z = np.load(f"host_cov_rank{rank:04d}_base.npz")
rng = np.random.default_rng(11)
sel = np.sort(rng.choice(1024, size=ns, replace=False))
C = torch.tensor(T["mc_pre_cov"][L][np.ix_(sel, sel)], dtype=torch.float32, device=dev)
K21 = torch.tensor(np.load(f"tensor_planes_rank{rank:04d}.npz")[f"pre_{L}"][np.ix_(sel, sel)], dtype=torch.float32, device=dev)
K22t = torch.tensor(tp[f"K22_{L}"][np.ix_(sel, sel)], dtype=torch.float32, device=dev)
K31t = torch.tensor(tp[f"K31_{L}"][np.ix_(sel, sel)], dtype=torch.float32, device=dev)
sK22 = float(K22t.pow(2).mean().sqrt()); sK31 = float(K31t.pow(2).mean().sqrt())
sC = float(C.pow(2).mean().sqrt()); sK = float(K21.pow(2).mean().sqrt())
# holdout mask on entries (20%)
mask = torch.tensor(rng.random((ns, ns)) < 0.8, device=dev)
# a-heavy init: top-r factor of C scaled to ~2/3 of diag mass
_wc, _Vc = np.linalg.eigh(C.cpu().numpy())
_w = torch.tensor(_wc, dtype=torch.float32, device=dev)
_V = torch.tensor(_Vc, dtype=torch.float32, device=dev)
_fa = (_V[:, -r:] * _w[-r:].clamp_min(1e-6).sqrt())
_fa = _fa * (2.0 / 3.0) ** 0.5
a = _fa.clone() + torch.randn(ns, r, device=dev) * 0.01
Mh = torch.randn(ns, r, r, device=dev) * 0.02
a.requires_grad_(); Mh.requires_grad_()
import os
lr = float(os.environ.get("LR", "3e-3"))
L4W = float(os.environ.get("L4W", "0.05"))
NIT = int(os.environ.get("NIT", "4000"))
opt = torch.optim.Adam([a, Mh], lr=lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NIT)
for it in range(NIT):
    M = 0.5 * (Mh + Mh.transpose(1, 2))
    Cf = a @ a.T + 0.5 * torch.einsum("iab,jab->ij", M, M)
    M2 = torch.einsum("iab,ibc->iac", M, M)
    t1 = torch.einsum("iab,jab->ij", M2, M)
    t2 = torch.einsum("ia,jab,ib->ij", a, M, a)
    t3 = 2.0 * torch.einsum("ia,iab,jb->ij", a, M, a)
    Kf = t1 + t2 + t3
    # pair-batched certified 4th-order losses
    if it == 0:
        thr22 = torch.quantile(
            K22t.abs().flatten().cpu(), 0.90).to(dev)
        thr31 = torch.quantile(
            K31t.abs().flatten().cpu(), 0.90).to(dev)
        good = ((K22t.abs() > thr22) | (K31t.abs() > thr31)).nonzero()
    ridx = torch.randint(0, good.shape[0], (2048,), device=dev)
    pi, pj = good[ridx, 0], good[ridx, 1]
    Mi = M[pi]; Mj = M[pj]; ai_ = a[pi]; aj_ = a[pj]
    M2i = M2[pi]; M2j = M2[pj]
    Jb = 0.5 * (torch.einsum("pab,pbc->pac", Mi, Mj)
                + torch.einsum("pab,pbc->pac", Mj, Mi))
    k22f = (torch.einsum("pab,pab->p", M2i, M2j)
            + 2 * torch.einsum("pab,pab->p", Jb, Jb)
            + 2 * torch.einsum("pa,pab,pb->p", ai_, M2j, ai_)
            + 2 * torch.einsum("pa,pab,pb->p", aj_, M2i, aj_)
            + 8 * torch.einsum("pa,pab,pb->p", ai_, Jb, aj_))
    k31f = (3 * torch.einsum("pab,pbc,pac->p", M2i, Mi, Mj)
            + 6 * torch.einsum("pa,pab,pb->p", ai_, Jb, ai_)
            + 6 * torch.einsum("pa,pab,pb->p", ai_, M2i, aj_))
    l4 = ((k22f - K22t[pi, pj]).pow(2).mean() / sK22**2
          + (k31f - K31t[pi, pj]).pow(2).mean() / sK31**2)
    l4w_now = L4W * min(1.0, max(0.0, (it - NIT // 8) / (NIT // 4)))
    loss = l4w_now * l4 + 3.0 * ((Cf - C).pow(2) * mask).mean() / sC**2 \
        + ((Kf - K21).pow(2) * mask).mean() / sK**2
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
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
    np.savez(f"ncqc_corefitJ_rank{rank:04d}_L{L}_r{r}.npz",
             sel=sel, a=a.cpu().numpy(), M=M.cpu().numpy(),
             K21_fit=Kf.cpu().numpy())
