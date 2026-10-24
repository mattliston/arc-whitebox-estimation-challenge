"""dissect337259 agent: along the common-mode direction, split the layer-l post-covariance transfer into the exact GAUSSIAN pair map
(Mehler/Hermite series to order 10 on (mu, Cpre): cov_G) and the NON-GAUSSIAN remainder NG = cov_post - cov_G, for the truth (true mu, C,
cov_post) and for the card (card mu, C; card cov_post read exactly through the next layer's pre: q^T cov_post q = u^T Cpre_{l+1} u with
q = W_{l+1} u, u = mu_{l+1}/|mu_{l+1}|). Injection along the mode = NG_card - NG_true; inherited (Gaussian transfer of the input error) =
G(card in) - G(true in). nets 35 / 100, l = 1..13. All numbers x1e-3."""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss, hermeval
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/'
STORE = P2 + 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/'
NORD = 10
zq, wq = hermegauss(80); wq = wq / wq.sum()                       # probabilists' Hermite quadrature, E over N(0,1)
HE = np.array([hermeval(zq, [0] * n + [1]) for n in range(NORD + 1)])   # He_n(z) at the nodes
from math import erf, sqrt, pi
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
def gauss_map(mu, C):
    v = np.maximum(np.diag(C), 1e-12); s = np.sqrt(v); a = mu / s
    rho = C / np.outer(s, s); np.fill_diagonal(rho, 0.0)
    relu = np.maximum(a[:, None] + zq[None, :], 0.0)              # relu(a + z) for each unit at the nodes (times s later)
    A = np.array([(relu * (HE[n] * wq)[None, :]).sum(1) for n in range(NORD + 1)])   # A_n(a) = E[relu(a+z) He_n(z)]
    cov = np.zeros_like(C); rn = np.ones_like(C); fact = 1.0
    for n in range(1, NORD + 1):
        rn = rn * rho; fact *= n; cov += np.outer(A[n], A[n]) * rn / fact
    cov *= np.outer(s, s)
    phi = np.exp(-a * a / 2) / sqrt(2 * pi); g = Phi(a); m = s * (a * g + phi); e2 = v * ((a * a + 1) * g + a * phi)
    np.fill_diagonal(cov, e2 - m * m)
    return cov
for rank in (35, 100):
    D = np.load('cpre_dump_rank%d_v1.npz' % rank); T = np.load(P2 + 'fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank)
    W = np.load(STORE + '%04d/weights.npy' % rank).astype(np.float64)
    print('=== rank %d   (x1e-3; along q = W u, u = card mu_{l+1}/|.|)' % rank)
    print('%3s %9s %9s %9s %9s | %8s %8s %8s | %8s' % ('l', 'R_true', 'R_card', 'G_true', 'G_card', 'NG_true', 'NG_card', 'inject', 'inherit'))
    for l in range(1, 14):
        muc = np.asarray(D['mu_%d' % l], float); Cc = np.asarray(D['Cpre_%d' % l], float)
        mut = np.asarray(T['mean_pre_%d' % l], float); Ct = np.asarray(T['cov_pre_%d' % l], float)
        mu1 = np.asarray(D['mu_%d' % (l + 1)], float); u = mu1 / np.linalg.norm(mu1); q = W[l + 1] @ u
        Rt = float(u @ np.asarray(T['cov_pre_%d' % (l + 1)], float) @ u); Rc = float(u @ np.asarray(D['Cpre_%d' % (l + 1)], float) @ u)
        Gt = float(q @ gauss_map(mut, Ct) @ q); Gc = float(q @ gauss_map(muc, Cc) @ q)
        NGt = Rt - Gt; NGc = Rc - Gc
        print('%3d %9.3f %9.3f %9.3f %9.3f | %8.2f %8.2f %8.2f | %8.2f' % (l, 1e3 * Rt, 1e3 * Rc, 1e3 * Gt, 1e3 * Gc, 1e3 * NGt, 1e3 * NGc, 1e3 * (NGc - NGt), 1e3 * (Gc - Gt)))
