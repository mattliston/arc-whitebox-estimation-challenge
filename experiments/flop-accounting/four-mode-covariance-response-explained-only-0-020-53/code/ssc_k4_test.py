#!/usr/bin/env python3
"""TAG g22-ssc-k4 (T-134): can Team Puffi's 4-mode covariance-response form represent the next layer's per-unit pre-activation
FOURTH cumulant at width 1024?  k4(z_i) ~ 3 q_i^T Lambda q_i,  q_ai = w_i^T B_a w_i, modes from the TRUE post state (MC on MPS):
B1 = C, B2 = Q/vbar (Q = P_off[c0 vbar C + c1 C∘C + c2/2 (D_m K^T + K D_m) + c3 (t m^T + m t^T) + c4 v v^T + c5 vbar^2 11^T] with the
six c's fitted too), B3 = vbar I, B4 = 1/2 (g_i + g_j) C_ij.  We fit ridge on the 10 products of (q_1..q_4) plus the 6 Q-term products
expanded linearly (Q enters q_2 linearly in the c's, so q_2 = sum_k c_k q_2k; the products q_2 q_b are bilinear -> we regress on all
q_a q_b with q_2 replaced by its six components: 4+6 = 9 basic features -> 45 products). Target: MC k4 of z = h @ W_{l+1} from the same
samples. Reports R^2 (held-out half of the units) per layer, and the diagonal-only baseline (k4 from the diagonal k4 transport).
  python ssc_k4_test.py <rank> <N> <layers,comma>"""
import sys, numpy as np, torch
sys.path.insert(0, '.'); import diff_card as dc
rank, N, layers = int(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')]
dev = "mps"; W = dc.load_weights(rank); Wt = [torch.as_tensor(np.asarray(W[l], dtype=np.float32), device=dev) for l in range(16)]
g = torch.Generator(device="cpu").manual_seed(rank + 31)
def cum4(Z):
    Zc = Z - Z.mean(0, keepdim=True); m2 = (Zc**2).mean(0); m4 = (Zc**4).mean(0); return (m4 - 3 * m2 * m2).cpu().double()
for l in layers:
    x = torch.randn(N, 1024, generator=g).to(dev); h = x
    for k in range(l + 1):
        h = torch.relu(h @ Wt[k])
    m = h.mean(0); hc = h - m; C = (hc.T @ hc) / N; v = torch.diagonal(C); vbar = v.mean()
    t = (hc**3).mean(0)                                                   # k3 diag
    K = ((hc * hc).T @ hc) / N                                            # K21 plane P_ab = k3(h_a,h_a,h_b) (centred)
    gate = (h > 0).float().mean(0)                                        # ReLU gate probability
    Wn = Wt[l + 1]; z = hc @ Wn; target = cum4(z)
    # per-unit quadratic forms q_ai = w_i^T B w_i for each mode / Q-term (w_i = column i of W)
    def qf(B): return torch.einsum('ai,ab,bi->i', Wn, B, Wn).cpu().double()
    one = torch.ones(1024, device=dev); Dm = torch.diag(m); Kt = K.T
    def offd(M): M = M.clone(); M.fill_diagonal_(0.0); return M
    feats = {
        'B1=C': qf(C), 'B3=vbar I': qf(vbar * torch.eye(1024, device=dev)), 'B4=g-sym C': qf(0.5 * (gate[:, None] + gate[None, :]) * C),
        'Q0=vbar C off': qf(offd(vbar * C)), 'Q1=C∘C off': qf(offd(C * C)), 'Q2=(Dm K^T + K Dm)/2 off': qf(offd(0.5 * (Dm @ Kt + K @ Dm))),
        'Q3=(t m^T + m t^T) off': qf(offd(torch.outer(t, m) + torch.outer(m, t))), 'Q4=v v^T off': qf(offd(torch.outer(v, v))), 'Q5=vbar^2 11^T off': qf(offd(vbar * vbar * torch.outer(one, one))),
    }
    names = list(feats); Fm = torch.stack([feats[n] for n in names], 1).numpy()      # (1024, 9)
    # products (45) -> ridge, held-out half
    P = np.stack([Fm[:, a] * Fm[:, b] for a in range(9) for b in range(a, 9)], 1); y = target.numpy()
    idx = np.random.RandomState(0).permutation(1024); tr, te = idx[:512], idx[512:]
    def fit(Xtr, ytr, Xte, lam):
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-30; A = (Xtr - mu) / sd; ym = ytr.mean()
        w = np.linalg.solve(A.T @ A + lam * len(ytr) * np.eye(A.shape[1]), A.T @ (ytr - ym)); return ((Xte - mu) / sd) @ w + ym
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        pred = fit(P[tr], y[tr], P[te], lam); r2 = 1 - np.mean((pred - y[te])**2) / np.var(y[te]); best = max(best or -9, r2)
    # baselines: q1^2 only (Gaussian-like), and the 4 canonical modes' 10 products
    P4 = np.stack([Fm[:, a] * Fm[:, b] for a in range(3) for b in range(a, 3)], 1)
    r2_4 = max(1 - np.mean((fit(P4[tr], y[tr], P4[te], lam) - y[te])**2) / np.var(y[te]) for lam in (1e-4, 1e-3, 1e-2, 1e-1))
    diag_only = ((Wn**4).T @ cum4(h).float().to(dev)).cpu().double().numpy()
    r2_diag = 1 - np.mean((diag_only[te] * (np.dot(diag_only[te], y[te]) / np.dot(diag_only[te], diag_only[te])) - y[te])**2) / np.var(y[te])
    print(f"L{l}->pre L{l+1} (N={N}): true k4 rms {np.sqrt(np.mean(y**2)):.2e} | SSC 9-feature products ridge held-out R^2 {best:.3f} | 3 canonical modes (C, vbar I, gC) R^2 {r2_4:.3f} | diagonal-only transport R^2 {r2_diag:.3f}", flush=True)
