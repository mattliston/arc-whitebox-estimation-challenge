"""dissect337259 agent: WHERE is the covariance error born? Per layer l: dC_l = Cpre_card - Cpre_true; split into the part INHERITED from
dC_{l-1} through the chain's first-order transfer (post: (g g^T) o dC with the true gates, diagonal via the pair map's diagonal response;
push: W^T . W) and the part BORN at layer l (the remainder). Also the common-mode share (along mu) and the diagonal share of each part.
usage: cpre_birth_v1.py <cpre dump npz> <mcall truth npz> <weights.npy> [rank]"""
import sys, numpy as np
D = np.load(sys.argv[1]); T = np.load(sys.argv[2]); W = np.load(sys.argv[3])
keys = list(D.keys()); print('dump keys (first 12):', keys[:12])
def get(prefix, l):
    for k in ('%s_%d' % (prefix, l), '%s%d' % (prefix, l), '%s_l%d' % (prefix, l)):
        if k in D: return np.asarray(D[k], dtype=np.float64)
    raise KeyError(prefix + str(l))
layers = sorted(int(k.split('_')[-1]) for k in keys if k.startswith('Cpre'))
print('layers', layers)
prev = None
print('%3s %8s %8s %8s %8s %8s %8s %8s' % ('l', '|dC|/|C|', 'diag', 'cm_mu', 'inherit', 'born', 'born_cm', 'born_dg'))
for l in layers:
    C = get('Cpre', l); mu = get('mu', l); Ct = np.asarray(T['cov_pre_%d' % l], dtype=np.float64)
    dC = C - Ct; nC = np.linalg.norm(Ct); nd = np.linalg.norm(dC)
    u = mu / np.linalg.norm(mu); cm = float(u @ dC @ u); cm_share = cm ** 2 / nd ** 2   # share of |dC|^2 along the unit vector mu (rank-one along mu)
    diag_share = float(np.sum(np.diag(dC) ** 2) / nd ** 2)
    inherit = born = born_cm = born_dg = float('nan')
    if prev is not None:
        dCp, Ctp, mup = prev
        vt = np.maximum(np.diag(Ctp), 1e-12); st = np.sqrt(vt); at = mup / st
        from math import erf, sqrt
        g = 0.5 * (1 + np.vectorize(erf)(at / sqrt(2)))
        # first-order transfer of a pre-covariance perturbation to the post covariance: off-diagonal g_i g_j dC_ij; diagonal: d var/d v = E[relu'^2] = g (plus mean shift terms, dropped)
        dPost = np.outer(g, g) * dCp; np.fill_diagonal(dPost, g * np.diag(dCp))
        Wl = np.asarray(W[l], dtype=np.float64)            # z_l = y_{l-1} @ W[l]
        inh = Wl.T @ dPost @ Wl
        bornM = dC - inh
        inherit = float(np.linalg.norm(inh) / nd); born = float(np.linalg.norm(bornM) / nd)
        born_cm = float((u @ bornM @ u) ** 2 / np.linalg.norm(bornM) ** 2); born_dg = float(np.sum(np.diag(bornM) ** 2) / np.linalg.norm(bornM) ** 2)
    print('%3d %8.4f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f' % (l, nd / nC, diag_share, cm_share, inherit, born, born_cm, born_dg))
    prev = (dC, Ct, mu)
