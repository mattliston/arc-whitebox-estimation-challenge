"""TAG fable-lead 2026-10-20: structure of the late-layer pre-covariance error dC = Cpre_card - Cpre_true (nets 35/100, L8-14,
card Cpre from the k21dump files, truth from mcall P32M). Questions: (1) is dC low-rank? (2) how much of diag(dC) (= the
per-unit pre-variance error, the -18/-26% oracle) lies along card-known directions g (W g for g in {post mean m, phi(a)s,
Phi(a), ones, top eigenvectors of the card's Cpre_{l-1}})? (3) a few universal coefficients -> fraction of diag(dC) explained,
fit on one net, checked on the other. Primary metric: the share of sum_i dv_i^2 explained (the oracle is diag-driven)."""
import numpy as np, sys
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
STORE = P2 + '/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
import json
man = json.load(open(STORE + '/manifest.json'))
MAN = man['roles']['observer_fit']['row_manifests']
def weights(rank):
    row = MAN[rank]
    p = STORE + "/" + row["path"]; p = p[:-len("/row_manifest.json")] if p.endswith("row_manifest.json") else p; return np.load(p + "/weights.npy")
from scipy.stats import norm
def feats(d, W):
    mu, v = d['mu'], d['v']; s = np.sqrt(v); a = mu / s
    phi, Phi = norm.pdf(a), norm.cdf(a)
    return mu, v, s, a, phi, Phi
out = {}
for rank in (35, 100):
    T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank)
    Wl = weights(rank)
    print('rank', rank, 'weights', None if Wl is None else type(Wl), file=sys.stderr)
    for l in range(8, 15):
        d = np.load('k21dump_rank%d_L%d.npz' % (rank, l))
        Cc = d['Cpre'].astype(np.float64); Ct = T['cov_pre_%d' % l].astype(np.float64)
        CtA, CtB = T['cov_pre_A_%d' % l].astype(np.float64), T['cov_pre_B_%d' % l].astype(np.float64)
        dC = Cc - Ct; dv = np.diag(dC); vt = np.diag(Ct)
        noise = np.diag(CtA - CtB) / 2.0   # MC noise of the truth diagonal (half-difference/2 ~ noise of the mean of halves... scaled)
        # spectrum of the full error (symmetric)
        ev = np.linalg.eigvalsh(dC); ev = ev[np.argsort(-np.abs(ev))]
        fro2 = (dC ** 2).sum(); diag2 = (dv ** 2).sum()
        top = np.cumsum(ev[:8] ** 2) / fro2
        # card eigvecs of its own Cpre_l (known at predict time)
        w, U = np.linalg.eigh(Cc); U = U[:, np.argsort(-w)]
        proj = []
        for r in (1, 2, 4, 8, 16, 32):
            Ur = U[:, :r]; R = Ur.T @ dC @ Ur   # error inside the card's top-r subspace
            # best diag-share explained by a rank-r symmetric correction inside the span (gives the diag of Ur R Ur^T)
            Dfit = np.einsum('ir,rs,is->i', Ur, R, Ur)
            proj.append((r, 1 - ((dv - Dfit) ** 2).sum() / diag2))
        print('rank %d L%2d |dC|_F/|Ct|_F %.3f  diag rms err/rms vt %.4f  rel dv mean %+.4f  |diag|^2/|dC|^2 %.3f  top8 eig share %s  '
              'diag share by card top-r subspace %s  noise/dv rms %.3f' % (
                  rank, l, np.sqrt(fro2 / (Ct ** 2).sum()), np.sqrt(diag2 / (vt ** 2).sum()), (dv / vt).mean(),
                  diag2 / fro2, ' '.join('%.2f' % x for x in top), ' '.join('%d:%.3f' % p for p in proj),
                  np.sqrt((noise ** 2).sum() / diag2)))
        out['%d_%d' % (rank, l)] = dict(dv=dv, vt=vt, mu=d['mu'], v=d['v'], k3=d['k3'], k4=d['k4'], gmean=d['gmean'], ev8=ev[:8])
np.savez('dcov_structure_v1.npz', **{k + '_' + kk: vv for k, v in out.items() for kk, vv in v.items()})
