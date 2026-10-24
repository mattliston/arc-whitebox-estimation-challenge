"""TAG fable-lead 2026-10-20: can a PAIR-GEOMETRY model of the K21 deficit D = K21_true - K21_card predict the next-layer per-unit
variance error? Effect path: dcov_ij = (d_i g_j D_ij + g_i d_j D_ji)/2 (the pair map's K21 kernel), dv_{l+1} = diag(W dcov W^T).
Basis B_k(ij) = s_i^2 s_j * P(rho_ij) * Q(a_i, a_j) (and K_card itself); theta fitted by ridge on net A (-dv target, pooled L9-13),
tested on net B. Also the ceiling: the exact deficit pushed through the same kernel (how much of dv^2 the K21 channel can explain)."""
import numpy as np, json, sys
from scipy.stats import norm
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
STORE = P2 + '/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.load(open(STORE + '/manifest.json'))['roles']['observer_fit']['row_manifests']
def weights(rank):
    p = STORE + '/' + MAN[rank]['path']; p = p[:-len('/row_manifest.json')]
    return np.load(p + '/weights.npy')
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '9,10,11,12,13').split(',')]
def basis(d):
    C = d['Cpre'].astype(np.float64); mu = d['mu']; v = d['v']; s = np.sqrt(v); a = mu / s
    rho = C / np.outer(s, s); np.fill_diagonal(rho, 0.0)
    phi, Phi = norm.pdf(a), norm.cdf(a)
    K = d['K21_card'].astype(np.float64); np.fill_diagonal(K, 0.0)
    D = d['K21_true'].astype(np.float64) - d['K21_card'].astype(np.float64); np.fill_diagonal(D, 0.0)
    S = np.outer(s ** 2, s)                      # the natural scale of kappa(i,i,j)
    B = {'K': K, 'rho': S * rho, 'rho2': S * rho ** 2, 'rho3': S * rho ** 3,
         'rho_phi_i': S * rho * phi[:, None], 'rho_phi_j': S * rho * phi[None, :], 'rho_a_i': S * rho * a[:, None], 'rho_a_j': S * rho * a[None, :],
         'rho2_a_i': S * rho ** 2 * a[:, None], 'rho2_phi_i': S * rho ** 2 * phi[:, None], 'rho_Phi_i': S * rho * Phi[:, None], 'rho_Phi_j': S * rho * Phi[None, :],
         'K_rho': K * rho, 'K_rho2': K * rho ** 2, 'K_a_i': K * a[:, None], 'K_phi_i': K * phi[:, None], 'K_a_j': K * a[None, :], 'K_phi_j': K * phi[None, :]}
    kern_d = phi / s; kern_g = Phi
    return B, D, kern_d, kern_g, s, a
def push(M, W, kd, kg):
    dc = 0.5 * (kd[:, None] * kg[None, :] * M + kg[:, None] * kd[None, :] * M.T)
    return np.einsum('im,ij,jm->m', W, dc, W)
data = {}
for rank in (35, 100):
    T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank); Wl = weights(rank)
    for l in LAYERS:
        d, dn = np.load('k21dump_rank%d_L%d.npz' % (rank, l)), np.load('k21dump_rank%d_L%d.npz' % (rank, l + 1))
        B, D, kd, kg, s, a = basis(d)
        W = np.asarray(Wl[l + 1], dtype=np.float64)
        mp = d['mu'] * norm.cdf(a) + s * norm.pdf(a)
        if abs(np.corrcoef(W.T @ mp, dn['mu'])[0, 1]) > abs(np.corrcoef(W @ mp, dn['mu'])[0, 1]): W = W.T
        vt = np.diag(T['cov_pre_%d' % (l + 1)]).astype(np.float64); y = -(dn['v'] - vt)      # the correction the card needs
        U = np.column_stack([push(M, W, kd, kg) for M in B.values()]); ut = push(D, W, kd, kg)
        data[(rank, l)] = (U, ut, y)
        c = np.corrcoef(ut, y)[0, 1]; beta = (ut * y).sum() / (ut ** 2).sum()
        print('rank %d L%2d: exact K21 deficit through the kernel -> next-layer dv: rms %.2e vs needed %.2e, corr %+.3f, beta %+.2f, share at beta %.3f, share at beta=1 %.3f'
              % (rank, l, ut.std(), y.std(), c, beta, 1 - ((y - beta * ut) ** 2).sum() / (y ** 2).sum(), 1 - ((y - ut) ** 2).sum() / (y ** 2).sum()), flush=True)
names = list(basis(np.load('k21dump_rank35_L10.npz'))[0].keys())
def fit(ranks, lam):
    X = np.concatenate([data[(r, l)][0] for r in ranks for l in LAYERS]); y = np.concatenate([data[(r, l)][2] for r in ranks for l in LAYERS])
    sc = np.sqrt((X ** 2).mean(0)).clip(1e-300); Xs = X / sc
    th = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(X.shape[1]), Xs.T @ y)
    return th / sc
def score(ranks, th):
    out = []
    for r in ranks:
        for l in LAYERS:
            U, ut, y = data[(r, l)]; p = U @ th; out.append((l, 1 - ((y - p) ** 2).sum() / (y ** 2).sum()))
    return out
for lam in (1e-3, 1e-2, 1e-1, 1.0):
    th = fit([35], lam); th2 = fit([100], lam)
    print('ridge %.0e: fit 35 -> share on 35 %s | on 100 %s || fit 100 -> share on 100 %s | on 35 %s' % (
        lam, ' '.join('%d:%+.3f' % x for x in score([35], th)), ' '.join('%d:%+.3f' % x for x in score([100], th)),
        ' '.join('%d:%+.3f' % x for x in score([100], th2)), ' '.join('%d:%+.3f' % x for x in score([35], th2))))
th = fit([35], 1e-2); print('theta (fit 35, 1e-2):', ' '.join('%s:%+.2e' % (n, t) for n, t in zip(names, th)))
