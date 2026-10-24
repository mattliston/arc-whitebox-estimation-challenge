"""TAG fable-lead 2026-10-20: size of the SECOND-ORDER Edgeworth terms the card's per-unit mean omits (m = mg + k3 e/6 + k4 f/24 is
first order): dm2 = k3^2/72 h6 + k3 k4/144 h7 + k4^2/1152 h8, h_n = (-1)^n He_{n-2}(a) phi(a)/s^(n-1). Pushed through W_{l+1}
(dmu_{l+1} = W dm2) against the measured next-layer pre-mean error dmu = mu_card - mu_true (k21dump + mcall), and the per-layer
mean closure error itself: m_card(given card inputs) vs the truth post mean is NOT separable here (inputs differ), so we report the
pushed term's correlation with the measured error and the share it would remove at its best scale (beta ~ 1 = physical)."""
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
    coef = np.zeros(n - 1); coef[-1] = 1.0
    return (-1) ** n * hermeval(a, coef) * norm.pdf(a) / s ** (n - 1)
for rank in (35, 100):
    T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank); Wl = weights(rank)
    for l in range(8, 14):
        d, dn = np.load('k21dump_rank%d_L%d.npz' % (rank, l)), np.load('k21dump_rank%d_L%d.npz' % (rank, l + 1))
        mu, v, k3, k4 = d['mu'], d['v'], d['k3'], d['k4']; s = np.sqrt(v); a = mu / s
        # truth per-unit cumulants at this layer for the "physical" size (the card's k3 is 0.85x truth)
        mg = mu * norm.cdf(a) + s * norm.pdf(a)
        t1 = k3 * h(3, a, s) / 6 + k4 * h(4, a, s) / 24
        dm2 = k3 ** 2 / 72 * h(6, a, s) + k3 * k4 / 144 * h(7, a, s) + k4 ** 2 / 1152 * h(8, a, s)
        dm2_k3 = k3 ** 2 / 72 * h(6, a, s)
        W = np.asarray(Wl[l + 1], dtype=np.float64)
        if abs(np.corrcoef(W.T @ mg, dn['mu'])[0, 1]) > abs(np.corrcoef(W @ mg, dn['mu'])[0, 1]): W = W.T
        mt = T['mean_pre_%d' % (l + 1)].astype(np.float64); dmu = dn['mu'] - mt
        mpt = T['mean_post_%d' % l].astype(np.float64)
        p2 = W @ dm2; p3 = W @ dm2_k3
        def stat(p):
            c = np.corrcoef(p, -dmu)[0, 1]; beta = (p * -dmu).sum() / (p ** 2).sum(); share = 1 - ((-dmu - beta * p) ** 2).sum() / (dmu ** 2).sum()
            return '%+.3f b%+.2f sh%.3f' % (c, beta, share)
        print('rank %d L%2d: |mg| %.3e |t1| %.2e |dm2| %.2e (k3^2 part %.2e)  pushed |W dm2| %.2e vs |dmu_%d| %.2e | corr/beta/share: dm2 %s  k3^2 only %s'
              % (rank, l, np.sqrt((mg ** 2).mean()), np.sqrt((t1 ** 2).mean()), np.sqrt((dm2 ** 2).mean()), np.sqrt((dm2_k3 ** 2).mean()), p2.std(), l + 1, dmu.std(), stat(p2), stat(p3)))
