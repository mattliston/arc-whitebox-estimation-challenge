#!/usr/bin/env python3
"""TAG fable-physicist (CR-22 v2: wider ridge grid to 1e3; second, sign-coherent control lam4_c = |lam_c|^(4/3)): ceiling of the one cheap FULL-DEPTH fourth-order family for per-unit k4: fourth-order weights on the
card's existing third-order bank columns (independent-source picture: K3 = sum_c lam_c p_c^3 -> K4 = sum_c lam4_c p_c^4; transport is
free and the per-unit read sum_c lam4_c p_ic^4 is an elementwise power of columns the card already pushes). ORACLE: one FREE lam4 per
column (~21k), fitted by ridge to the 32M truth per-unit k4 at L12-14 on half of the units, scored on the other half (2 folds, several
random splits), with the isotropic v^2 term fitted alongside. Bank: r35/r100 dumps (fable_oldtier_economics_v1/bank_r*/layer_LL.npz,
S5e/S6b-era card, pre-activation columns pp, weights lam). Sanity: the bank's own k3 read sum lam p^3 vs truth k3. Positive control: a
planted k4 = iso + sum_c lam4_c p^4 with lam4_c = s_c |lam_c|^(4/3) (random signs s_c), scaled to the truth's residual size, must be
recovered on held-out units. Report: held-out relative rms k4 error vs |truth k4| (isotropic-only, + columns), best ridge per layer.
usage: col_k4_unit_fit.py <rank> [<rank> ...]"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]
rng = np.random.default_rng(22)
def ridge_heldout(F, iso, y, alphas, splits=4):
    """held-out error of y ~ a*iso + b + F @ w (ridge on w only, dual form), averaged over random 2-fold unit splits"""
    n = len(y); res = {a: [] for a in alphas}; res_iso = []
    for s in range(splits):
        perm = rng.permutation(n); folds = (perm[: n // 2], perm[n // 2:])
        for tr, te in (folds, folds[::-1]):
            X = np.column_stack([np.ones(len(tr)), iso[tr]]); c, *_ = np.linalg.lstsq(X, y[tr], rcond=None)
            r_tr = y[tr] - X @ c; pred_iso_te = np.column_stack([np.ones(len(te)), iso[te]]) @ c
            res_iso.append(((y[te] - pred_iso_te) ** 2).sum())
            Ftr = F[tr]; mu = Ftr.mean(0); Ftr = Ftr - mu; Fte = F[te] - mu
            K = Ftr @ Ftr.T; Kte = Fte @ Ftr.T; scale = np.trace(K) / len(tr)
            for a in alphas:
                beta = np.linalg.solve(K + a * scale * np.eye(len(tr)), r_tr - r_tr.mean())
                pred = pred_iso_te + r_tr.mean() + Kte @ beta
                res[a].append(((y[te] - pred) ** 2).sum())
    tot = splits * (y ** 2).sum()
    return np.sqrt(np.sum(res_iso) / tot), {a: np.sqrt(np.sum(v) / tot) for a, v in res.items()}
alphas = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
out = {}
for rk in map(int, sys.argv[1:]):
    T = np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rk:04d}_indep32M.npz')
    D = P2 / 'fable_oldtier_economics_v1' / f'bank_r{rk}'
    print(f'net {rk}: held-out k4 error (share of |truth k4|, test units only) | isotropic only | + free lam4 on the bank columns (best ridge) | control recovered')
    for L in (12, 13, 14):
        z = np.load(D / f'layer_{L:02d}.npz'); pp = z['pp'].astype(np.float64); lam = z['lam'].astype(np.float64)
        v, k3t, k4t = T['mc_pre_v'][L], T['mc_pre_k3'][L], T['mc_pre_k4'][L]
        k3bank = (pp ** 3) @ lam
        sane = np.linalg.norm(k3bank - k3t) / np.linalg.norm(k3t); cos3 = k3bank @ k3t / np.linalg.norm(k3bank) / np.linalg.norm(k3t)
        F = pp ** 4; F /= np.sqrt((F ** 2).mean(0, keepdims=True)) + 1e-300           # per-column scale for the ridge
        iso = v ** 2
        e_iso, e_col = ridge_heldout(F, iso, k4t, alphas)
        best = min(e_col, key=e_col.get)
        # positive control: planted column-supported residual of the same size as the truth's residual
        X = np.column_stack([np.ones(1024), iso]); c, *_ = np.linalg.lstsq(X, k4t, rcond=None); rtrue = k4t - X @ c
        g = (pp ** 4) @ (rng.choice([-1, 1], size=len(lam)) * np.abs(lam) ** (4 / 3)); g -= g.mean()
        ysyn = X @ c + g * np.linalg.norm(rtrue) / np.linalg.norm(g)
        c_iso, c_col = ridge_heldout(F, iso, ysyn, alphas)
        cbest = min(c_col, key=c_col.get)
        g2 = (pp ** 4) @ (np.abs(lam) ** (4 / 3)); g2 -= g2.mean(); ysyn2 = X @ c + g2 * np.linalg.norm(rtrue) / np.linalg.norm(g2)
        d_iso, d_col = ridge_heldout(F, iso, ysyn2, alphas); dbest = min(d_col, key=d_col.get)
        out[f'{rk}_{L}'] = dict(k3_bank_err=sane, k3_bank_cos=cos3, iso=e_iso, col=e_col[best], alpha=best, ctrl_iso=c_iso, ctrl_col=c_col[cbest], ctrl2_iso=d_iso, ctrl2_col=d_col[dbest], ctrl_alpha=cbest, ctrl2_alpha=dbest, all_col={str(a): float(e) for a, e in e_col.items()},
                                ncols=int(pp.shape[1]))
        print(f'  L{L}: columns {pp.shape[1]} (bank k3 read vs truth: err {sane:.3f}, cos {cos3:+.3f}) | {e_iso:.3f} | {e_col[best]:.3f} (ridge {best:g}) | '
              f'control: {c_iso:.3f} -> {c_col[cbest]:.3f} ({cbest:g}) | coherent control: {d_iso:.3f} -> {d_col[dbest]:.3f} ({dbest:g})')
Path(HERE / 'col_k4_unit_fit_v2.json').write_text(json.dumps(out, indent=1))
