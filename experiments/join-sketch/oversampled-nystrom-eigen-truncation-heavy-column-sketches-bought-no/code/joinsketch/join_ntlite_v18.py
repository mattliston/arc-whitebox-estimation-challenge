"""[opus-lead] NT on the SUBSAMPLED Gram: Nystrom with rank+p rows (energy order) on the heaviest kc columns, QR, Nystrom core
C = Q[rows]^-1 R^T in the (r+p)-frame, then (full) top-r eigenvectors of C, or (block q) keep the leading r-q frame columns and rotate
the last q+p by the top-q eigenvectors of M = CTT^2 + 2 CHT^T CHT (the team's _nt_truncate). f32 pipeline with emulated kernel noise,
dumped exact core for the old tier. usage: join_ntlite_v18.py <join_L.npz> kc"""
import sys, numpy as np
d = np.load(sys.argv[1]); kc = int(sys.argv[2]); r = 384
J = d['J'].astype(np.float32); lam = d['lam'].astype(np.float32); w = lam * lam; m, K = J.shape; old = int(d['old_columns'])
Qp = d['Qp'].astype(np.float32) if old else None; core = d['core'].astype(np.float32) if old else None; F = d['F'].astype(np.float32) if old else None
lam_old = d['lam_old'].astype(np.float32); w32 = d['w32'].astype(np.float64); prev = d['previous']
X = np.hstack([(Qp @ F) if old else np.zeros((m, 0), np.float32), J]).astype(np.float64); L = np.concatenate([lam_old, lam]).astype(np.float64); WJ = w32[prev, :].T
def plane(Y): return (Y * Y * L[None, :]) @ Y.T
P0 = plane(X); PP0 = plane(WJ @ X); n0 = np.linalg.norm(P0); np0 = np.linalg.norm(PP0)
def err(Q):
    Q = Q.astype(np.float64); R = Q @ (Q.T @ X); return np.linalg.norm(plane(R) - P0) / n0, np.linalg.norm(plane(WJ @ R) - PP0) / np0
rng = np.random.default_rng(0)
def mm(A, B, eps):
    C = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
    if eps: C = C + (eps * np.abs(C).max() * rng.standard_normal(C.shape)).astype(np.float32)
    return C
dg = (J * J) @ w; Qc = Qp @ core if old else None
if old: dg = dg + (Qc * Qp).sum(1)
energy = np.argsort(-dg.astype(np.float64), kind='stable')
cn = ((J * J).sum(0) * w).astype(np.float64); c1 = np.sort(np.argsort(-cn, kind='stable')[:kc]); J1 = J[:, c1]
def sketch(rows, Jx, wx, eps):
    Y = mm(Jx, wx[:, None] * Jx[rows].T, eps)
    if old: Y = Y + mm(Qc, Qp[rows].T, eps)
    return Y
def nt(rows, Y, p, q):
    Q, R = np.linalg.qr(Y); k = Q.shape[1]
    C = np.linalg.solve(Q[rows].astype(np.float32), R.astype(np.float32).T).astype(np.float32); C = (C + C.T) / 2
    if q == 0:
        ev, V = np.linalg.eigh(C.astype(np.float64)); return Q @ V[:, k - r:][:, ::-1].astype(np.float32)
    h = r - q; CT = C[:, h:]; CTT = CT[h:, :]; CHT = CT[:h, :]
    M = CTT @ CTT + 2.0 * CHT.T @ CHT; M = (M + M.T) / 2
    ev, V = np.linalg.eigh(M.astype(np.float64)); Xv = V[:, (k - h) - q:][:, ::-1].astype(np.float32)
    return np.concatenate([Q[:, :h], Q[:, h:] @ Xv], axis=1)
print('layer', int(d['layer']), 'kc', kc)
for eps in (0.0, 3e-7, 1e-6):
    rows = np.sort(energy[:r])
    line = 'eps %.0e | today(full) %.4f/%.4f' % (eps, *err(np.linalg.qr(sketch(rows, J, w, eps))[0]))
    line += ' | lite %.4f/%.4f' % err(np.linalg.qr(sketch(rows, J1, w[c1], eps))[0])
    for p, q in ((32, 128), (64, 192), (64, 0), (128, 192)):
        rows_p = energy[:r + p]
        try:
            line += ' | lite+NT p%d q%d %.4f/%.4f' % (p, q, *err(nt(rows_p, sketch(rows_p, J1, w[c1], eps), p, q)))
        except np.linalg.LinAlgError as e:
            line += ' | lite+NT p%d q%d FAIL' % (p, q)
    rows_p = energy[:r + 64]
    line += ' | full+NT p64 q192 %.4f/%.4f' % err(nt(rows_p, sketch(rows_p, J, w, eps), 64, 192))
    print(line, flush=True)
