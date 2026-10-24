"""[opus-pass-release] Cholesky-sandwich PD census, analysis step (plain numpy, offline): for every captured sandwich call, the live
block C_live = C[R[:m], R[:m]] (dcv path) or C (full path): min/max eigenvalue, Cholesky success in float32 and float64, and the
float32 Cholesky route's relative error vs the direct float32 sandwich W_a^T (C_live W_a) (both against a float64 reference).
usage: pd_census_v1.py <cov.npz>"""
import sys
import numpy as np
z = np.load(sys.argv[1]); meta = z['meta']
ok32 = ok64 = 0
for i, (kind, nr, m) in enumerate(meta):
    C = z[f'C{i}'].astype(np.float64); W = z[f'W{i}'].astype(np.float64); R = z[f'R{i}']
    live = R[:m] if (kind == 0 and m > 0) else np.arange(C.shape[0])
    Cl = C[np.ix_(live, live)]; Wa = W[live]
    Cl = 0.5 * (Cl + Cl.T)
    ev = np.linalg.eigvalsh(Cl); emin, emax = ev[0], ev[-1]
    ref = Wa.T @ Cl @ Wa
    try:
        L32 = np.linalg.cholesky(Cl.astype(np.float32)); c32 = True; ok32 += 1
        Y = (L32.T.astype(np.float32) @ Wa.astype(np.float32)); new = (Y.T @ Y).astype(np.float64)
        err_new = np.linalg.norm(new - ref) / np.linalg.norm(ref)
    except np.linalg.LinAlgError:
        c32 = False; err_new = float('nan')
    try:
        np.linalg.cholesky(Cl); c64 = True; ok64 += 1
    except np.linalg.LinAlgError:
        c64 = False
    direct = (Wa.astype(np.float32).T @ (Cl.astype(np.float32) @ Wa.astype(np.float32))).astype(np.float64)
    err_dir = np.linalg.norm(direct - ref) / np.linalg.norm(ref)
    print(f'call {i:2d} kind {"dc" if kind == 0 else "flms"} m {len(live):4d} eig min {emin:+.3e} max {emax:.3e} ratio {emin/emax:+.2e} | chol f32 {c32} f64 {c64} | rel err chol-route {err_new:.2e} direct {err_dir:.2e}')
print(f'Cholesky OK f32 {ok32}/{len(meta)}, f64 {ok64}/{len(meta)}')
