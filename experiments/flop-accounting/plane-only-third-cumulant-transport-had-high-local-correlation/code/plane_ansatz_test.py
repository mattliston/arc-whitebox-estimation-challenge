#!/usr/bin/env python3
"""TAG g22-plane-ansatz (T-125): does the K21 PLANE (two-equal-index third cumulants, 1024^2 per layer) carry the third-order content
that sets the next layer's per-unit pre-activation skew, or is the all-distinct part of the tensor essential?
MC on MPS: for net <rank>, layers l: post h_l -> plane P_ab = kappa3(h_a,h_a,h_b), diagonal T_aaa; next-layer pre z = h @ W_{l+1};
ansatz  kappa3(z_i) ~ sum_a W_ai^3 T_aaa + 3 sum_{a!=b} W_ai^2 W_bi P_ab ; direct kappa3(z_i) = E[(z_i-mu_i)^3] from the same samples.
  python plane_ansatz_test.py <rank> <N_samples> <layers,comma>"""
import sys, numpy as np, torch
sys.path.insert(0, '.'); import diff_card as dc
rank, N, layers = int(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')]
dev = "mps"; W = dc.load_weights(rank); Wt = [torch.as_tensor(np.asarray(W[l], dtype=np.float32), device=dev) for l in range(16)]
vl = np.load(f"vlabels/mps_truth_rank{rank:04d}_P1M.npz")
B = 4096; g = torch.Generator(device="cpu").manual_seed(rank)
for l in layers:
    n = 1024; c = dict(dtype=torch.float64)   # accumulators on the CPU in float64 (MPS has no float64)
    m = torch.zeros(n, **c); M11 = torch.zeros(n, n, **c); M21 = torch.zeros(n, n, **c); M20 = torch.zeros(n, **c); M30 = torch.zeros(n, **c)
    zm = torch.zeros(n, **c); z2 = torch.zeros(n, **c); z3 = torch.zeros(n, **c)
    nb = N // B
    for b in range(nb):
        x = torch.randn(B, n, generator=g).to(dev)
        h = x
        for k in range(l + 1):
            h = torch.relu(h @ Wt[k])
        z = h @ Wt[l + 1]
        m += h.sum(0).cpu().double(); M20 += (h * h).sum(0).cpu().double(); M30 += (h ** 3).sum(0).cpu().double()
        M11 += (h.T @ h).cpu().double(); M21 += ((h * h).T @ h).cpu().double()
        zm += z.sum(0).cpu().double(); z2 += (z * z).sum(0).cpu().double(); z3 += (z ** 3).sum(0).cpu().double()
    Ns = nb * B; m /= Ns; M20 /= Ns; M30 /= Ns; M11 /= Ns; M21 /= Ns; zm /= Ns; z2 /= Ns; z3 /= Ns
    # cumulants
    P = M21 - 2 * m[:, None] * M11 - m[None, :] * M20[:, None] + 2 * (m * m)[:, None] * m[None, :]     # kappa3(h_a, h_a, h_b)
    T = M30 - 3 * m * M20 + 2 * m ** 3                                                                  # kappa3(h_a,h_a,h_a)
    Poff = P.clone(); Poff.fill_diagonal_(0.0)
    Wn = Wt[l + 1].cpu().double()
    pred3 = (Wn ** 3).T @ T + 3 * torch.diagonal((Wn * Wn).T @ (Poff @ Wn))                     # first formula (treats P as symmetric)
    pred = (Wn ** 3).T @ T + 2 * torch.diagonal((Wn * Wn).T @ (Poff @ Wn)) + torch.diagonal((Wn * Wn).T @ (Poff.T @ Wn))   # exact index patterns: P_ab = k3(h_a,h_a,h_b) is not symmetric
    direct = z3 - 3 * zm * z2 + 2 * zm ** 3
    diag_only = (Wn ** 3).T @ T
    ref = torch.as_tensor(vl["mc_pre_k3"][l + 1], dtype=torch.float64)
    def rel(a, b): return float(torch.sqrt(torch.mean((a - b) ** 2) / torch.mean(b ** 2)))
    def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))
    print(f"L{l}->pre L{l+1} (N={Ns}): direct skew vs 1M labels rel {rel(direct, ref):.3f} | PLANE ANSATZ (exact patterns) vs direct: rel {rel(pred, direct):.3f} corr {corr(pred, direct):.4f} scale {float((pred*direct).sum()/(direct*direct).sum()):.3f} | symmetric-P formula: rel {rel(pred3, direct):.3f} corr {corr(pred3, direct):.4f} | diagonal-only: rel {rel(diag_only, direct):.3f} corr {corr(diag_only, direct):.4f} | rms skew {float(torch.sqrt(torch.mean(direct**2))):.3e}", flush=True)
