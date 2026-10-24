#!/usr/bin/env python3
"""TAG g22-puffi-feats (T-137): correlation-graph features for the closure, after Team Puffi's drift corrector (phase-1 PDF, D9-D11) and
the organizers' "high-variance directions" hint. For each honest Gaussian-card oracle row (rows_meanleak_mech15_gausspsd), replicate the
oracle recursion in torch (diff_card: truth post means teacher-forced, consistent covariance update, PSD projection) and at every layer
compute per-unit features from the PRE correlation matrix R- (diagonal zeroed) and the POST covariance C+:
  neighbour sums over the 8 largest |R-_ij|:  S_i(x) = sum_t r_it x_jt,  U_i(x) = sum_t |r_it| x_jt  for x in (a, g, phi)      (6)
  sum_t r_it^2, u_i1 = max_j |R-_ij|                                                                                       (2)
  softmax-weighted neighbour means P_gamma,i(x) = sum_{j!=i} e^{gamma R_ij^2} x_j / sum e^{...}, gamma in (8, 32), x in (a, g, phi) (6)
  post: top-8 eigen-subspace loading pi_i = sum_{k<=8} V_ik^2 of C+, <R+_ij>_j, <(R+_ij)^2>_j, log(C+_ii / mean diag), (C+ a)_i/s, (C+ g)_i/s (6)
-> puffifeat.npy (16, 1024, 20) per row. Sanity: the torch one-step post mean vs the row's host.npy. Usage: puffi_feats.py <lo> <hi>"""
import sys, math, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); import diff_card as dc
HERE = Path(__file__).resolve().parent; ROWS = HERE / "rows_meanleak_mech15_gausspsd"
lo, hi = int(sys.argv[1]), int(sys.argv[2]); dev = "mps"; SQ2PI = math.sqrt(2 * math.pi)

def feats_for(rank):
    W = dc.load_weights(rank); truth = torch.as_tensor(np.load(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float32))
    n = 1024; m = torch.zeros(n); C = torch.eye(n); out = np.zeros((16, n, 20), np.float32); host_chk = []
    for l in range(16):
        Wl = torch.as_tensor(np.asarray(W[l], dtype=np.float32)); Wd = Wl.to(dev)
        mu = (Wl.T @ m); Sigma = (Wd.T @ C.to(dev) @ Wd).cpu()
        var = torch.clamp(torch.diagonal(Sigma), min=1e-30); s = torch.sqrt(var); a = mu / s
        phi = torch.exp(-0.5 * a * a) / SQ2PI; g = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        R = Sigma / (s[:, None] * s[None, :]); R.fill_diagonal_(0.0)
        # top-8 neighbours by |R|
        absR = R.abs(); top = torch.topk(absR, 8, dim=1); idx = top.indices; r = torch.gather(R, 1, idx)
        xs = torch.stack([a, g, phi], 1)                                         # (n, 3)
        S = torch.stack([(r * xs[idx, k]).sum(1) for k in range(3)], 1); U = torch.stack([(r.abs() * xs[idx, k]).sum(1) for k in range(3)], 1)
        r2 = (r * r).sum(1); u1 = top.values[:, 0]
        Pg = []
        for gamma in (8.0, 32.0):
            w = torch.exp(gamma * R * R); w.fill_diagonal_(0.0); w = w / (w.sum(1, keepdim=True) + 1e-30)
            Pg.append(w @ xs)                                                    # (n, 3)
        Pg = torch.cat(Pg, 1)
        # post state (Gaussian card) + oracle correction
        mp, Cp, aux = dc.relu_gauss_layer(mu.double(), Sigma.double()); mp = mp.float(); Cp = Cp.float()
        host_chk.append(mp.numpy().copy())
        d = truth[l] - mp; Cn = Cp - torch.outer(mp, d) - torch.outer(d, mp) - torch.outer(d, d)
        ev, V = torch.linalg.eigh(0.5 * (Cn + Cn.T).double()); Cn = (V * torch.clamp(ev, min=0.0)[None, :]) @ V.T; Cn = Cn.float()
        # T-137b LEAK FIX: the post features must come from the CARD's post covariance Cp (available before the correction), never from Cn
        # (Cn = Cp - m d^T - d m^T - d d^T contains the target d; using it gave 97.8% "removal" = leakage)
        evp, Vp = torch.linalg.eigh(0.5 * (Cp + Cp.T).double()); pi = (Vp[:, -8:].float() ** 2).sum(1)
        dp = torch.clamp(torch.diagonal(Cp), min=1e-30); sp = torch.sqrt(dp); Rp = Cp / (sp[:, None] * sp[None, :]); Rp.fill_diagonal_(0.0)
        post = torch.stack([pi, Rp.mean(1), (Rp * Rp).mean(1), torch.log(dp / dp.mean()), (Cp @ a) / sp, (Cp @ g) / sp], 1)
        out[l] = torch.cat([S, U, r2[:, None], u1[:, None], Pg, post], 1).numpy()
        m = truth[l].clone(); C = Cn
    return out, np.stack(host_chk)

for rank in range(lo, hi + 1):
    d = ROWS / f"row_{rank:04d}"
    if not (d / "host.npy").exists():
        continue
    t0 = time.time(); F, hc = feats_for(rank); h = np.load(d / "host.npy").astype(np.float32)
    rel = [float(np.sqrt(np.mean((hc[l] - h[l]) ** 2) / np.mean(h[l] ** 2))) for l in (1, 8, 15)]
    np.save(d / "puffifeat2.npy", F); print(f"rank {rank}: puffifeat {F.shape} | torch one-step vs row host rel L1 {rel[0]:.1e} L8 {rel[1]:.1e} L15 {rel[2]:.1e} | {time.time()-t0:.0f}s", flush=True)
