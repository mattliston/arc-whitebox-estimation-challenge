#!/usr/bin/env python3
"""TAG analyst A4: anatomy of the card's carried post-covariance error, network 35, layers 6, 10, 13.
E = C_card - C_truth. C_card = the carried post covariance after recentring (hook 'cov_out', analyst/a2_cov_rank0035.npz, the
current torch card). C_truth = fable_covhunter_v1/mcall_rank0035_P32M.npz cov_post_l (32M samples) with its two 16M halves
cov_post_A_l, cov_post_B_l; C_truth = (A + B) / 2 to 4e-7, so the truth's sampling-noise matrix is distributed as (A - B) / 2.
Reported per layer:
  1. relative Frobenius error of the off-diagonal part and of the diagonal, with the noise floor from the halves and the
     noise-free value from the cross product <C_card - A, C_card - B> (the two halves' noises are independent);
  2. the eigen-spectrum of E: share of squared Frobenius norm in the top 1, 4, 16, 64 eigen-directions (by |eigenvalue|), the same
     for the pure-noise matrix (A - B) / 2, and a cross-validated share (eigenvectors from C_card - A, energy measured on C_card - B);
  3. cosine of E's top eigenvector with the mean vector and with the top eigenvector of C_truth;
  4. consequence for the next layer's per-unit variance, dv = diag(W^T E W): relative rms error, and how much of it comes from the
     diagonal of E, from the off-diagonal, and from the top 1, 4, 16, 64 eigen-directions of E.
usage: a4_cov_anatomy.py [rank=35] -> a4_cov_anatomy_rank{rank}.json + text on stdout"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy').astype(np.float64)
cc = np.load(HERE / f'a2_cov_rank{rank:04d}.npz'); mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
layers = sorted(int(k.split('_')[-1]) for k in cc.files if k.startswith('cov_out_'))
off = lambda X: X - np.diag(np.diag(X)); fro = np.linalg.norm; KS = (1, 4, 16, 64)
def spectrum(E):
    lam, U = np.linalg.eigh(E); o = np.argsort(-np.abs(lam)); return lam[o], U[:, o]
out = []
for l in layers:
    C = cc[f'cov_out_{l}']; Ct = mc[f'cov_post_{l}']; Ca = mc[f'cov_post_A_{l}']; Cb = mc[f'cov_post_B_{l}']; m = mc[f'mean_post_{l}']
    E = C - Ct; Nz = (Ca - Cb) / 2; Ea, Eb = C - Ca, C - Cb
    r = dict(layer=l, asym=float(np.abs(C - C.T).max()))
    for name, f in (('off', off), ('diag', lambda X: np.diag(X))):
        den = fro(f(Ct))
        r[name] = dict(rel=float(fro(f(E)) / den), floor=float(fro(f(Nz)) / den),
                       real=float(np.sqrt(max((f(Ea) * f(Eb)).sum(), 0.0)) / den))
    r['total'] = dict(rel=float(fro(E) / fro(Ct)), floor=float(fro(Nz) / fro(Ct)), real=float(np.sqrt(max((Ea * Eb).sum(), 0.0)) / fro(Ct)))
    r['diag_share_of_E2'] = float((np.diag(E) ** 2).sum() / (E ** 2).sum())
    r['diag_mean_rel'] = float(np.diag(E).sum() / np.diag(Ct).sum())           # relative error of the trace
    lam, U = spectrum(E); lamn, _ = spectrum(Nz); lama, Ua = spectrum(Ea)
    r['share'] = {k: float((lam[:k] ** 2).sum() / (lam ** 2).sum()) for k in KS}
    r['share_noise_matrix'] = {k: float((lamn[:k] ** 2).sum() / (lamn ** 2).sum()) for k in KS}
    qb = np.einsum('ik,ij,jk->k', Ua, Eb, Ua)                                  # u_i^T (C - B) u_i, independent noise
    r['share_crossval'] = {k: float((lama[:k] * qb[:k]).sum() / (Ea * Eb).sum()) for k in KS}
    r['participation'] = float((lam ** 2).sum() ** 2 / (lam ** 4).sum())
    r['top_eigs'] = [float(x) for x in lam[:8]]
    lt, Ut = np.linalg.eigh(Ct); u1t = Ut[:, -1]
    r['cos_top_mean'] = float(abs(U[:, 0] @ m) / fro(m)); r['cos_top_covtop'] = float(abs(U[:, 0] @ u1t))
    r['cos_covtop_mean'] = float(abs(u1t @ m) / fro(m)); r['cov_top_share_of_trace'] = float(lt[-1] / lt.sum())
    r['top_quad_mean'] = float(m @ E @ m / (m @ Ct @ m))                       # relative error of the variance along the mean direction
    r['top_quad_covtop'] = float(u1t @ E @ u1t / lt[-1])
    # consequence for the next layer's per-unit pre-activation variance
    Wn = W[l + 1]; v = np.einsum('ia,ij,ja->a', Wn, Ct, Wn)
    dv = np.einsum('ia,ij,ja->a', Wn, E, Wn); dvn = np.einsum('ia,ij,ja->a', Wn, Nz, Wn)
    dva = np.einsum('ia,ij,ja->a', Wn, Ea, Wn); dvb = np.einsum('ia,ij,ja->a', Wn, Eb, Wn)
    dvd = (Wn ** 2).T @ np.diag(E); dvo = dv - dvd
    rms = lambda x: float(np.sqrt((x * x).mean()))
    P = Wn.T @ U                                                              # (unit a, direction i)
    r['next_v'] = dict(rel_rms=rms(dv / v), floor=rms(dvn / v), real=float(np.sqrt(max((dva * dvb / v ** 2).mean(), 0.0))),
                       mean_rel=float((dv / v).mean()), from_diag=rms(dvd / v), from_off=rms(dvo / v), corr_diag_off=float(np.corrcoef(dvd, dvo)[0, 1]),
                       unexplained_after_top={k: float((((dv - (P[:, :k] ** 2) @ lam[:k]) / v) ** 2).mean() / ((dv / v) ** 2).mean()) for k in KS})
    out.append(r)
    print(f"layer {l}: relative Frobenius error  off-diagonal {r['off']['rel'] * 100:.3f}% (floor {r['off']['floor'] * 100:.3f}%, noise-free {r['off']['real'] * 100:.3f}%)   "
          f"diagonal {r['diag']['rel'] * 100:.3f}% (floor {r['diag']['floor'] * 100:.3f}%, noise-free {r['diag']['real'] * 100:.3f}%)   trace error {r['diag_mean_rel'] * 100:+.3f}%   diagonal share of |E|^2 {r['diag_share_of_E2'] * 100:.1f}%")
    print(f"   eigen-spectrum of E, share of |E|^2 in top 1/4/16/64: " + ' / '.join(f"{r['share'][k] * 100:.1f}%" for k in KS)
          + "   cross-validated: " + ' / '.join(f"{r['share_crossval'][k] * 100:.1f}%" for k in KS)
          + "   pure-noise matrix: " + ' / '.join(f"{r['share_noise_matrix'][k] * 100:.1f}%" for k in KS) + f"   participation ratio {r['participation']:.1f}")
    print(f"   top eigenvalues of E: " + ' '.join(f'{x:+.2e}' for x in r['top_eigs']))
    print(f"   cos(top eigvec of E, mean) {r['cos_top_mean']:.3f}   cos(top eigvec of E, top eigvec of C) {r['cos_top_covtop']:.3f}   cos(top eigvec of C, mean) {r['cos_covtop_mean']:.3f}   "
          f"C top eigenvalue share of trace {r['cov_top_share_of_trace'] * 100:.1f}%   relative error of variance along mean {r['top_quad_mean'] * 100:+.3f}%, along C top {r['top_quad_covtop'] * 100:+.3f}%")
    nv = r['next_v']
    print(f"   next layer per-unit variance: relative rms error {nv['rel_rms'] * 100:.3f}% (floor {nv['floor'] * 100:.3f}%, noise-free {nv['real'] * 100:.3f}%), mean {nv['mean_rel'] * 100:+.3f}%; "
          f"from diag(E) alone {nv['from_diag'] * 100:.3f}%, from off-diag alone {nv['from_off'] * 100:.3f}% (corr {nv['corr_diag_off']:+.2f}); "
          f"share of squared error LEFT after removing the top 1/4/16/64 eigen-directions: " + ' / '.join(f"{nv['unexplained_after_top'][k] * 100:.0f}%" for k in KS))
json.dump(dict(TAG='analyst A4 covariance error anatomy', rank=rank, layers=out), open(HERE / f'a4_cov_anatomy_rank{rank:04d}.json', 'w'), indent=1)
