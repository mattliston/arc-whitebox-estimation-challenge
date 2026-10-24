"""TAG fable-lead 2026-10-20: does the MISSING (2,2)-plane covariance term explain the per-unit pre-variance error?
Gram-Charlier: Cov(relu_i, relu_j) += (K22_ij / 4) * phi_Sigma(0, 0)  (no mean term). Offline with the k21dump (K22_true, Cpre, mu, v
at layer l) and the truth: dCpost_l(K22) -> next-layer variance change dv_{l+1} = diag(W_{l+1} dCpost W_{l+1}^T), compared with the
measured error dv_{l+1} = v_card - v_true. Also the card's own scale-mode object (v v^T + 2 C^2)/3 as a proxy plane (radial by LS)."""
import numpy as np, json
from scipy.stats import norm
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
STORE = P2 + '/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.load(open(STORE + '/manifest.json'))['roles']['observer_fit']['row_manifests']
def weights(rank):
    p = STORE + '/' + MAN[rank]['path']; p = p[:-len('/row_manifest.json')]
    return np.load(p + '/weights.npy')
for rank in (35, 100):
    T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank); Wl = weights(rank)
    for l in range(8, 14):
        d, dn = np.load('k21dump_rank%d_L%d.npz' % (rank, l)), np.load('k21dump_rank%d_L%d.npz' % (rank, l + 1))
        C = d['Cpre'].astype(np.float64); mu = d['mu']; v = d['v']
        K22 = d['K22_true'].astype(np.float64)
        if K22.shape != (1024, 1024): print('no K22 at', rank, l); continue
        det = np.outer(v, v) - C ** 2
        q = (v[None, :] * mu[:, None] ** 2 - 2 * C * np.outer(mu, mu) + v[:, None] * mu[None, :] ** 2) / det
        p00 = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
        K22o = K22.copy(); np.fill_diagonal(K22o, 0.0)
        dCpost = 0.25 * K22o * p00
        # card proxy plane: scale mode (v v^T + 2 C^2)/3 with radial by least squares on the off-diagonal of the truth
        S = (np.outer(v, v) + 2 * C ** 2) / 3; np.fill_diagonal(S, 0.0)
        radial = (S * K22o).sum() / (S * S).sum(); corr_plane = (S * K22o).sum() / np.sqrt((S * S).sum() * (K22o * K22o).sum())
        dCpost_card = 0.25 * radial * S * p00
        W = np.asarray(Wl[l + 1], dtype=np.float64)
        mp = norm.cdf(mu / np.sqrt(v)) * mu + np.sqrt(v) * norm.pdf(mu / np.sqrt(v))
        if abs(np.corrcoef(W.T @ mp, dn['mu'])[0, 1]) > abs(np.corrcoef(W @ mp, dn['mu'])[0, 1]): W = W.T
        dv_pred = np.einsum('ij,jk,ik->i', W, dCpost, W); dv_pred_card = np.einsum('ij,jk,ik->i', W, dCpost_card, W)
        vt = np.diag(T['cov_pre_%d' % (l + 1)]).astype(np.float64); dv = dn['v'] - vt   # card - truth: the term should be ADDED to the card, so dv_pred ~ -dv
        c = np.corrcoef(dv_pred, -dv)[0, 1]; cc = np.corrcoef(dv_pred_card, -dv)[0, 1]
        beta = (dv_pred * -dv).sum() / (dv_pred ** 2).sum(); share = 1 - ((-dv - beta * dv_pred) ** 2).sum() / (dv ** 2).sum()
        betac = (dv_pred_card * -dv).sum() / (dv_pred_card ** 2).sum(); sharec = 1 - ((-dv - betac * dv_pred_card) ** 2).sum() / (dv ** 2).sum()
        # also the post-variance (diagonal of dCpost is zero by construction) and the magnitude of the term vs the post cov
        print('rank %d L%2d->%2d: K22 term rms %.2e (cov rms %.2e)  dv_pred rms %.2e vs dv rms %.2e  corr(truth K22 term, -dv) %+.3f beta %.2f share %.3f | card scale-mode plane: corr with K22 %.3f radial %.3g, corr(term,-dv) %+.3f beta %.2f share %.3f'
              % (rank, l, l + 1, np.sqrt((dCpost ** 2).mean()), np.sqrt((T['cov_post_%d' % l].astype(np.float64) ** 2).mean()), dv_pred.std(), dv.std(), c, beta, share, corr_plane, radial, cc, betac, sharec))
