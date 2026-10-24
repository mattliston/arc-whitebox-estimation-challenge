#!/usr/bin/env python3
"""TAG g22-tucker-spectrum (T-126): is the TRUE third-cumulant tensor of the post-activations low-rank in the Tucker (mode) sense at width 1024?
From N centred samples h~ (N x 1024) of layer l's post, the mode-1 Gram of the sample third-moment tensor T_abc = (1/N) sum_n h~_na h~_nb h~_nc is
  G = T_(1) T_(1)^T = (1/N^2) H~^T K H~,  K_nn' = (h~_n . h~_n')^2   (N x N)
whose eigenvalues are the squared mode singular values. Sampling noise adds a nearly flat floor; a second independent half gives the floor
(cross-Gram of two independent halves has NO noise bias: G_x = T1_(1) T2_(1)^T is unbiased for the true Gram). We report the eigen-spectrum of
the symmetrised cross-Gram: energy captured by the top k modes, k = 16, 64, 128, 256, and compare with the plane's spectrum.
  python tucker_spectrum_test.py <rank> <N per half> <layers,comma>"""
import sys, numpy as np, torch
sys.path.insert(0, '.'); import diff_card as dc
rank, N, layers = int(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')]
dev = "mps"; W = dc.load_weights(rank); Wt = [torch.as_tensor(np.asarray(W[l], dtype=np.float32), device=dev) for l in range(16)]
g = torch.Generator(device="cpu").manual_seed(rank + 7)
def sample_post(l, n):
    x = torch.randn(n, 1024, generator=g).to(dev); h = x
    for k in range(l + 1):
        h = torch.relu(h @ Wt[k])
    return h
for l in layers:
    H1 = sample_post(l, N); H2 = sample_post(l, N)
    mu = 0.5 * (H1.mean(0) + H2.mean(0)); H1 = H1 - mu; H2 = H2 - mu
    # cross Gram: G_x = (1/N^2) H1^T K12 H2 with K12_nn' = (h1_n . h2_n')^2 ; symmetrise
    K12 = (H1 @ H2.T) ** 2                                   # (N, N)
    Gx = (H1.T @ K12 @ H2) / float(N * N)
    Gx = 0.5 * (Gx + Gx.T)
    ev = torch.linalg.eigvalsh(Gx.cpu().double()).flip(0).numpy()
    evp = np.clip(ev, 0, None); tot = evp.sum()
    # the plane's spectrum for comparison (P_ab = k3(h_a,h_a,h_b) via cumulants of the pooled sample)
    H = torch.cat([H1, H2], 0); m = H.mean(0); Hc = H - m
    P = ((Hc * Hc).T @ Hc) / float(2 * N)                    # third central moment plane (centred) = k3(a,a,b)
    sp = torch.linalg.svdvals(P.cpu().double()).numpy() ** 2
    cum = lambda s, k: float(s[:k].sum() / s.sum())
    print(f"L{l} post third-cumulant tensor (N={N} x2): mode-1 Gram top-k energy: k16 {cum(evp,16):.3f} k64 {cum(evp,64):.3f} k128 {cum(evp,128):.3f} k256 {cum(evp,256):.3f} k512 {cum(evp,512):.3f} | negative-eigen mass (noise sign) {float(-np.clip(ev,None,0).sum()/tot):.3f} | plane top-k energy: k16 {cum(sp,16):.3f} k64 {cum(sp,64):.3f} k128 {cum(sp,128):.3f}", flush=True)
