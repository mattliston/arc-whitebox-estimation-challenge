import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
pc.set_threads(3); rank = 35; l = 12; n = 1024; off = ~np.eye(n, dtype=bool)
D = np.load(HERE / f'e35_rank{rank:04d}_L{l}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
f = lambda k: MC[k].astype(np.float64); sym = lambda x: (x + x.T) / 2
C, mu, Cp, mp = f(f'cov_pre_{l}'), f(f'mean_pre_{l}'), f(f'cov_post_{l}'), f(f'mean_post_{l}')
cd = pc.build(rank, learned=True); pair = cd.env['_nonlinear_first_gaussian']; T = lambda x: torch.from_numpy(np.ascontiguousarray(x))
Z = torch.zeros((n, n), dtype=torch.float64); z1 = torch.zeros(n, dtype=torch.float64)
PM = lambda mu_, C_: pair(T(mu_), T(C_), Z, z1, None, 'covariance')[1].numpy().astype(np.float64)
P = sym(D['Mmix_0']); t = D['tz_0']
print('C diag mean', C.diagonal().mean(), 'C offdiag rms', np.sqrt((C[off] ** 2).mean()), 'P diag mean', P.diagonal().mean(), 'P offdiag rms', np.sqrt((P[off] ** 2).mean()), 't rms', np.sqrt((t ** 2).mean()), 'mu rms', np.sqrt((mu ** 2).mean()))
v = np.diag(C); sd = np.sqrt(v); al = mu / sd; pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd
hand = (g[:, None] * P) * g[None, :]
rowm = (d * t)[:, None] * C * g[None, :]; rowv = (-(d * al / (2 * sd)) * np.diag(P))[:, None] * C * g[None, :]
hand_full = hand + rowm + rowm.T + rowv + rowv.T
for eps in (2e-6, 2e-5, 2e-4):
    Poff = P - np.diag(np.diag(P))
    parts = {}
    for name, (dt, dP) in dict(all=(t, P), offdiag_only=(0 * t, Poff), diag_only=(0 * t, np.diag(np.diag(P))), mean_only=(t, 0 * P)).items():
        parts[name] = (PM(mu + eps * dt, C + eps * dP) - PM(mu - eps * dt, C - eps * dP)) / (2 * eps)
    c = lambda a, b: round(float(np.corrcoef(a[off], b[off])[0, 1]), 4); r = lambda a, b: round(float(np.sqrt((a[off] ** 2).mean() / (b[off] ** 2).mean())), 3)
    print(json.dumps(dict(eps=eps, offdiag_part_vs_gate=dict(corr=c(parts['offdiag_only'], hand), size=r(parts['offdiag_only'], hand)), diag_part_vs_rowv=dict(corr=c(parts['diag_only'], rowv + rowv.T), size=r(parts['diag_only'], rowv + rowv.T)),
                          mean_part_vs_rowm=dict(corr=c(parts['mean_only'], rowm + rowm.T), size=r(parts['mean_only'], rowm + rowm.T)), all_vs_hand=dict(corr=c(parts['all'], hand_full), size=r(parts['all'], hand_full)),
                          sizes_rms=dict(offdiag=float('%.3g' % np.sqrt((parts['offdiag_only'][off] ** 2).mean())), diag=float('%.3g' % np.sqrt((parts['diag_only'][off] ** 2).mean())), mean=float('%.3g' % np.sqrt((parts['mean_only'][off] ** 2).mean()))))))
