"""Full carried cubic contractions, offline invariant information only."""
import numpy as np
from scipy.linalg import eigh


def tensor_contractions(Q, pp, lam, W, star, bpre, Bpre):
    n, k = Q.shape
    ia, ib = np.triu_indices(k)
    p = pp.astype(np.float64)
    z = p.T@Q
    weights = lam.astype(np.float64)[:, None]*z[:, ia]*z[:, ib]
    values = p@weights
    del p, weights
    w = W.astype(np.float64)
    a = star.astype(np.float64)
    q = w@Q
    aq = a.T@q
    source = (a@(q[:, ia]*q[:, ib])+q[:, ia]*aq[:, ib]+q[:, ib]*aq[:, ia])/3
    values += w.T@source
    bq = bpre@Q
    mq = Bpre@Q
    gram = Q.T@mq
    values += bpre[:, None]*gram[ia, ib][None, :]+mq[:, ia]*bq[ib][None, :]+mq[:, ib]*bq[ia][None, :]
    full = np.zeros((n, k, k))
    full[:, ia, ib] = values
    full[:, ib, ia] = values
    return full


def quadratic_features(C, pp, lam, W, star, bpre, Bpre):
    n = len(C)
    values, U = eigh((C+C.T)/2, subset_by_index=(n-8, n-1), check_finite=False)
    assert values.min() > 0
    Q = U/np.sqrt(values)[None, :]
    sigma = np.sqrt(np.maximum(np.diag(C), 1e-10))
    B = tensor_contractions(Q, pp, lam, W, star, bpre, Bpre)/sigma[:, None, None]
    c = U*np.sqrt(values)[None, :]/sigma[:, None]
    B2 = B@B
    tr = np.trace(B, axis1=1, axis2=2)
    tr2 = np.sum(B*B, axis=(1, 2))
    tr3 = np.sum(B2*B, axis=(1, 2))
    tr4 = np.sum(B2*B2, axis=(1, 2))
    bc = (B@c[:, :, None])[:, :, 0]
    feat = np.stack((tr, tr2, tr*tr, tr3, tr4,
                     np.sum(c*bc, axis=1), np.sum(bc*bc, axis=1), np.sum(c*c, axis=1)), axis=1)
    assert np.isfinite(feat).all()
    return feat.astype(np.float32), float(values.sum()/np.trace(C))
