"""TAG fable-lead 2026-10-20: Gram-Charlier covariance terms the live pair map may lack, tested against the measured next-layer
pre-variance error dv = v_card - v_true (k21dump card quantities at layer l + W_{l+1}; mcall truth). Mehler for mixed derivative orders:
E[f^(p)(z_i) g^(q)(z_j)] = sum_n C^n/n! h_{p+n}(i) h_{q+n}(j), h_0 = m (Gaussian mean), h_1 = Phi, h_n = (-1)^n He_{n-2}(a) phi / s^(n-1).
First order (n >= 1 so the mean product cancels): T30 = k3_i/6 sum_n C^n/n! h_{3+n,i} h_{n,j} + (i<->j); T40 = k4_i/24 (...h_{4+n}...) + (i<->j).
Second order: T33 = k3_i k3_j/36 sum_{n>=1} C^n/n! h_{3+n,i} h_{3+n,j}; T3_21 = k3_i K21_ij/12 sum_{n>=0} C^n/n! h_{5+n,i} h_{1+n,j} + (i<->j);
T21sq = K21_ij^2/8 sum_n C^n/n! h_{4+n,i} h_{2+n,j} + (i<->j); T2112 = K21_ij K21_ji/4 sum_{n>=0} C^n/n! h_{3+n,i} h_{3+n,j}.
Each term is pushed through W (dv_pred = diag(W T W^T)) and compared with -dv (the correction the card needs): corr, beta (1 = the term is
missing at its physical size), share of sum dv^2 removed at the best beta."""
import numpy as np, json
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermeval
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
STORE = P2 + '/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.load(open(STORE + '/manifest.json'))['roles']['observer_fit']['row_manifests']
def weights(rank):
    p = STORE + '/' + MAN[rank]['path']; p = p[:-len('/row_manifest.json')]
    return np.load(p + '/weights.npy')
def hfun(n, a, s, mu):
    if n == 0: return mu * norm.cdf(a) + s * norm.pdf(a)
    if n == 1: return norm.cdf(a)
    coef = np.zeros(n - 1); coef[-1] = 1.0
    return (-1) ** n * hermeval(a, coef) * norm.pdf(a) / s ** (n - 1)
NMAX = 4
for rank in (35, 100):
    T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank); Wl = weights(rank)
    for l in range(8, 14):
        d, dn = np.load('k21dump_rank%d_L%d.npz' % (rank, l)), np.load('k21dump_rank%d_L%d.npz' % (rank, l + 1))
        C = d['Cpre'].astype(np.float64); mu = d['mu']; v = d['v']; k3 = d['k3']; k4 = d['k4']; s = np.sqrt(v); a = mu / s
        K = d['K21_card'].astype(np.float64); np.fill_diagonal(K, 0.0)
        Co = C.copy(); np.fill_diagonal(Co, 0.0)
        H = {n: hfun(n, a, s, mu) for n in range(0, 10)}
        Cn = {0: np.ones_like(Co)}; f = 1.0
        for n in range(1, NMAX + 1): f *= n; Cn[n] = Co ** n / f
        def mixed(p, q, nmin):   # sum_n C^n/n! h_{p+n,i} h_{q+n,j}
            return sum(Cn[n] * np.outer(H[p + n], H[q + n]) for n in range(nmin, NMAX + 1))
        terms = {}
        t = (k3 / 6)[:, None] * mixed(3, 0, 1); terms['T30'] = t + t.T
        t = (k4 / 24)[:, None] * mixed(4, 0, 1); terms['T40'] = t + t.T
        terms['T33'] = np.outer(k3, k3) / 36 * mixed(3, 3, 1)
        t = (k3 / 12)[:, None] * K * mixed(5, 1, 0); terms['T3_21'] = t + t.T
        t = K ** 2 / 8 * mixed(4, 2, 0); terms['T21sq'] = t + t.T
        terms['T2112'] = K * K.T / 4 * mixed(3, 3, 0)
        W = np.asarray(Wl[l + 1], dtype=np.float64)
        mp = H[0]
        if abs(np.corrcoef(W.T @ mp, dn['mu'])[0, 1]) > abs(np.corrcoef(W @ mp, dn['mu'])[0, 1]): W = W.T
        vt = np.diag(T['cov_pre_%d' % (l + 1)]).astype(np.float64); dv = dn['v'] - vt
        out = []
        for name, M in terms.items():
            np.fill_diagonal(M, 0.0)
            p = np.einsum('ij,jk,ik->i', W, M, W)
            c = np.corrcoef(p, -dv)[0, 1]; beta = (p * -dv).sum() / (p ** 2).sum(); share = 1 - ((-dv - beta * p) ** 2).sum() / (dv ** 2).sum()
            out.append('%s rms %.1e corr %+.3f b%+.2f sh%.3f' % (name, p.std(), c, beta, share))
        print('rank %d L%2d->%2d dv rms %.2e | %s' % (rank, l, l + 1, dv.std(), ' | '.join(out)))
