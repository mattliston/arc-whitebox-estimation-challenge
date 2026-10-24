#!/usr/bin/env python3
"""TAG opus-reexam E32 analysis (network 35): the norm-coupling matrix Ntilde(l) = E[z~ z~' (S - tr C)] - 2 C^2 of layers 9, 12, 13.
(a) how much of it is isotropic (tr C x C and C^2), how much mean forms (mu mu', mu (C mu)' + sym), what is left (network-specific part);
(b) spectrum of the network-specific part, cross-half (eigenvectors from half 0, energy read on half 1): diagonal share, top-k shares;
(c) transport law between layers 12 and 13: Ntilde(13) against W' G Ntilde(12) G W (cross-half) with the pushed Gaussian and diagonal sources.
usage: e32_analyze.py <rank>"""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import pcard as pc
rank = int(sys.argv[1]); n = 1024
D = np.load(HERE / f'e32_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
LAYERS = [int(x) for x in D['layers']]; print('samples per half', int(D['N']))
off = ~np.eye(n, dtype=bool)
def parts(l):
    C = MC[f'cov_pre_{l}'].astype(np.float64); mu = MC[f'mean_pre_{l}'].astype(np.float64); C2 = C @ C; Cmu = C @ mu
    N = [D[f'M_{l}_{h}'].astype(np.float64) - 2 * C2 for h in (0, 1)]; N = [(x + x.T) / 2 for x in N]
    forms = dict(trC_C=np.trace(C) * C, C2=C2, mumu=np.outer(mu, mu), muCmu=np.outer(mu, Cmu) + np.outer(Cmu, mu), eye=np.eye(n))
    return C, mu, N, forms
def project(N, forms, names, mask):
    X = np.stack([forms[k][mask] for k in names], 1); co = [np.linalg.lstsq(X, x[mask], rcond=None)[0] for x in N]
    res = []
    for x, c in zip(N, co):
        r = x.copy(); r[mask] = x[mask] - X @ c; res.append(r)
    return co, res
RES = {}
for l in LAYERS:
    C, mu, N, forms = parts(l)
    tot = float((N[0][off] * N[1][off]).sum())
    co_iso, r_iso = project(N, forms, ['trC_C', 'C2'], off); e_iso = float((r_iso[0][off] * r_iso[1][off]).sum())
    co_all, r_all = project(N, forms, ['trC_C', 'C2', 'mumu', 'muCmu'], off); e_all = float((r_all[0][off] * r_all[1][off]).sum())
    noise = float(((N[0] - N[1])[off] ** 2).sum()) / 2      # noise energy of one half
    # diagonal: residual after the same forms fitted on the diagonal + identity
    dg = np.eye(n, dtype=bool); co_d, r_d = project(N, forms, ['trC_C', 'C2', 'mumu', 'eye'], dg); e_d = float((r_d[0][dg] * r_d[1][dg]).sum()); tot_d = float((N[0][dg] * N[1][dg]).sum())
    RES[l] = (C, mu, N, forms, r_all)
    print(json.dumps(dict(layer=l, offdiag=dict(cross_energy=float('%.4g' % tot), noise_energy_one_half=float('%.4g' % noise), isotropic_share=round(1 - e_iso / tot, 4),
                                                with_mean_forms_share=round(1 - e_all / tot, 4), coefficients_iso=[float('%.4g' % x) for x in (co_iso[0] + co_iso[1]) / 2],
                                                radial_over_3_from_trC_C=float('%.4g' % ((co_iso[0][0] + co_iso[1][0]) / 2))),
                          diagonal=dict(cross_energy=float('%.4g' % tot_d), residual_share_after_forms=round(e_d / tot_d, 4)))))
    # spectrum of the network-specific part (off-diagonal residual, diagonal set to its own residual)
    R0, R1 = r_all[0].copy(), r_all[1].copy()
    np.fill_diagonal(R0, r_d[0][dg]); np.fill_diagonal(R1, r_d[1][dg])
    cross = float((R0 * R1).sum()); lam, U = np.linalg.eigh(R0); order = np.argsort(-np.abs(lam)); lam, U = lam[order], U[:, order]
    read = np.einsum('ik,ij,jk->k', U, R1, U); cum = np.cumsum(lam * read) / cross
    print(json.dumps(dict(layer=l, network_specific=dict(cross_energy=float('%.4g' % cross), diagonal_share=round(float((np.diag(R0) * np.diag(R1)).sum() / cross), 4),
                                                      top_k_share={k: round(float(cum[k - 1]), 3) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256)},
                                                      top_eigenvalues_half0=[float('%.3g' % x) for x in lam[:6]], top_eigenvalues_read_on_half1=[float('%.3g' % x) for x in read[:6]]))))
if 12 in RES and 13 in RES:
    PU = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz')
    W = pc.weights(rank)[13].astype(np.float64)                       # z13 = y12 @ W
    C12, mu12, N12, f12, r12 = RES[12]; C13, mu13, N13, f13, r13 = RES[13]
    v = np.diag(C12); sd = np.sqrt(v); al = mu12 / sd; pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd
    m = MC['mean_post_12'].astype(np.float64); gam = g - m * d; gbar = float(gam.mean())
    r2 = v * ((1 + al * al) * g + al * pdf); r3 = v * sd * ((al ** 3 + 3 * al) * g + (al * al + 2) * pdf); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * pdf)
    S2 = 2 * (g - g * g - m * d); S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2
    GC = g[:, None] * C12
    src_gauss = W.T @ ((GC * S2[None, :]) @ GC.T) @ W; src_diag = W.T @ (S4[:, None] * W)
    print(json.dumps(dict(gamma_bar=round(gbar, 4), two_gamma_bar=round(2 * gbar, 4))))
    for h, hp in ((0, 1), (1, 0)):
        T = W.T @ ((g[:, None] * N12[hp]) * g[None, :]) @ W                # transported matrix, from the OTHER half of layer 12
        Tw = W.T @ ((gam[:, None] * 0 + g[:, None]) * N12[hp] * g[None, :]) @ W
        y = N13[h]; names = ['T', 'trC_C', 'C2', 'mumu', 'muCmu', 'gauss', 'diagsrc']
        F = dict(T=T, gauss=src_gauss, diagsrc=src_diag, **{k: f13[k] for k in ('trC_C', 'C2', 'mumu', 'muCmu')})
        X = np.stack([F[k][off] for k in names], 1); co = np.linalg.lstsq(X, y[off], rcond=None)[0]
        # network-specific parts: remove the isotropic and mean forms from target and from T, then correlate
        Xi = np.stack([f13[k][off] for k in ('trC_C', 'C2', 'mumu', 'muCmu')], 1); pr = lambda a: a - Xi @ np.linalg.lstsq(Xi, a, rcond=None)[0]
        ty, tT, tg, td = pr(y[off]), pr(T[off]), pr(src_gauss[off]), pr(src_diag[off]); ty2 = pr(N13[hp][off])
        Xn = np.stack([tT, tg, td], 1); cn = np.linalg.lstsq(Xn, ty, rcond=None)[0]; fit = Xn @ cn
        print(json.dumps(dict(target_half=h, full_fit_coefficients=dict(zip(names, [float('%.4g' % x) for x in co])),
                              network_specific=dict(corr_with_transported=round(float(np.corrcoef(ty, tT)[0, 1]), 3), slope_on_transported=float('%.4g' % ((ty @ tT) / (tT @ tT))),
                                                    coefficients_T_gauss_diag=[float('%.4g' % x) for x in cn],
                                                    explained_share_cross_half=round(float((2 * fit @ ty2 - fit @ fit) / (ty @ ty2)), 4),
                                                    reliability_of_target=round(float(np.corrcoef(ty, ty2)[0, 1]), 3)))))
