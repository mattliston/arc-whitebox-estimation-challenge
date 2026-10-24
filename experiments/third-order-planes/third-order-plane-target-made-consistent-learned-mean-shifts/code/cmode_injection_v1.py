"""dissect337259 agent: per-layer INJECTION of the covariance error and its common-mode anatomy (nets 35/100; dumps Cpre_l, mu_l; truths
cov_pre_l). In the pre-space of layer l+1: inherited = W^T [ (g g^T) o dC_l, diag g o diag(dC_l) ] W (first-order transfer of the layer-l
pre error through the ReLU pair map, exact push), born = dC_{l+1} - inherited. The mean direction splits as mu_{l+1} = W^T (Phi o mu_l) +
W^T (s o phi)_l = p2 + p1 (Gaussian part + kink part); u^T born u = (b11 + 2 b12 + b22)/|mu|^2 with b11 = p1^T born p1 (kink-kink:
the k22/radial sector), b12 (cross: the b-carrier / scale-mode third cumulant), b22 (Gaussian-Gaussian: the g g^T C transfer, exact at
first order). Noise on u^T N u ~ lam sqrt(2/P) ~ 0.003. usage: cmode_injection_v1.py"""
import numpy as np
from math import erf, sqrt, pi
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/'
STORE = P2 + 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/'
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
expired = {4: 4096, 5: 6080, 6: 8000, 7: 9792}   # columns of cohorts aged >= 3 before the join (2 x births of cohorts 0..l-3)
for rank in (35, 100):
    D = np.load('cpre_dump_rank%d_v1.npz' % rank); T = np.load(P2 + 'fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank)
    W = np.load(STORE + '%04d/weights.npy' % rank).astype(np.float64)
    print('=== rank %d   (all entries x1000; cm = u^T X u with u = mu/|mu|; noise ~ 3)' % rank)
    print('%3s %7s %7s %7s | %7s %7s %7s %7s | %7s %7s | %6s' % ('l+1', 'cm_tot', 'cm_inh', 'cm_born', 'b11', 'b12x2', 'b22', 'sum', 'born_F', 'inh_F', 'exp_k'))
    for l in range(1, 14):
        C = np.asarray(D['Cpre_%d' % l], float); mu = np.asarray(D['mu_%d' % l], float); Ct = np.asarray(T['cov_pre_%d' % l], float)
        C1 = np.asarray(D['Cpre_%d' % (l + 1)], float); mu1 = np.asarray(D['mu_%d' % (l + 1)], float); Ct1 = np.asarray(T['cov_pre_%d' % (l + 1)], float)
        dC = C - Ct; dC1 = C1 - Ct1
        v = np.maximum(np.diag(Ct), 1e-12); s = np.sqrt(v); a = mu / s; g = Phi(a); phi = np.exp(-a * a / 2) / sqrt(2 * pi)
        dPost = np.outer(g, g) * dC; np.fill_diagonal(dPost, g * np.diag(dC))
        Wn = W[l + 1]                                   # z_{l+1} = y_l @ W[l+1]  -> Cpre_{l+1} = W^T Cpost_l W
        inh = Wn.T @ dPost @ Wn; born = dC1 - inh
        u = mu1 / np.linalg.norm(mu1)
        p1 = Wn.T @ (s * phi); p2 = Wn.T @ (g * mu); n2 = mu1 @ mu1
        b11 = p1 @ born @ p1 / n2; b12 = p1 @ born @ p2 / n2; b22 = p2 @ born @ p2 / n2
        print('%3d %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f | %7.1f %7.1f | %6s' % (l + 1, 1e3 * (u @ dC1 @ u), 1e3 * (u @ inh @ u), 1e3 * (u @ born @ u), 1e3 * b11, 2e3 * b12, 1e3 * b22, 1e3 * (b11 + 2 * b12 + b22), 1e3 * np.linalg.norm(born), 1e3 * np.linalg.norm(inh), expired.get(l + 1, '')))
