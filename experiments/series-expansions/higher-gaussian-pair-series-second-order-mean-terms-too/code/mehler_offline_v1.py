"""TAG fable-lead 2026-10-20: size of the Gaussian pair map's Mehler truncation (the live `nonlinear` keeps n=1,2 of
Cov = sum_n C^n/n! h_n(i) h_n(j), h_1 = Phi(a), h_n = (-1)^n He_{n-2}(a) phi(a)/s^(n-1)) at the card's own inputs, and whether the
missing n>=3 terms, pushed through W_{l+1}, explain the measured next-layer pre-variance error dv = v_card - v_true (k21dump + mcall)."""
import numpy as np, json
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermeval
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
STORE = P2 + '/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.load(open(STORE + '/manifest.json'))['roles']['observer_fit']['row_manifests']
def weights(rank):
    p = STORE + '/' + MAN[rank]['path']; p = p[:-len('/row_manifest.json')]
    return np.load(p + '/weights.npy')
def h(n, a, s):
    if n == 1: return norm.cdf(a)
    coef = np.zeros(n - 1); coef[-1] = 1.0
    return (-1) ** n * hermeval(a, coef) * norm.pdf(a) / s ** (n - 1)
for rank in (35, 100):
    T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank); Wl = weights(rank)
    for l in range(8, 14):
        d, dn = np.load('k21dump_rank%d_L%d.npz' % (rank, l)), np.load('k21dump_rank%d_L%d.npz' % (rank, l + 1))
        C = d['Cpre'].astype(np.float64); mu = d['mu']; v = d['v']; s = np.sqrt(v); a = mu / s
        Co = C.copy(); np.fill_diagonal(Co, 0.0)
        rho = Co / np.outer(s, s)
        terms = {}
        Cn = Co.copy(); fact = 1.0
        for n in range(1, 9):
            if n > 1: Cn = Cn * Co; fact *= n
            hn = h(n, a, s); terms[n] = (Cn / fact) * np.outer(hn, hn)
        rest = sum(terms[n] for n in range(3, 9)); t3 = terms[3]; t4 = terms[4]
        W = np.asarray(Wl[l + 1], dtype=np.float64)
        mp = norm.cdf(a) * mu + s * norm.pdf(a)
        if abs(np.corrcoef(W.T @ mp, dn['mu'])[0, 1]) > abs(np.corrcoef(W @ mp, dn['mu'])[0, 1]): W = W.T
        vt = np.diag(T['cov_pre_%d' % (l + 1)]).astype(np.float64); dv = dn['v'] - vt
        def push(M): return np.einsum('ij,jk,ik->i', W, M, W)
        p_rest, p3, p4 = push(rest), push(t3), push(t4)
        def stat(p):
            c = np.corrcoef(p, -dv)[0, 1]; beta = (p * -dv).sum() / (p ** 2).sum(); share = 1 - ((-dv - beta * p) ** 2).sum() / (dv ** 2).sum()
            return '%+.3f b%.2f sh%.3f' % (c, beta, share)
        print('rank %d L%2d: |rho| mean %.4f max %.3f  |T1| %.2e |T2| %.2e |T3| %.2e |T4| %.2e |rest3-8| %.2e (Frob rms)  pushed rest rms %.2e vs dv rms %.2e | corr/beta/share: rest %s  T3 %s  T4 %s'
              % (rank, l, np.abs(rho).mean(), np.abs(rho).max(), *[np.sqrt((terms[n] ** 2).mean()) for n in (1, 2, 3, 4)], np.sqrt((rest ** 2).mean()), p_rest.std(), dv.std(), stat(p_rest), stat(p3), stat(p4)))
