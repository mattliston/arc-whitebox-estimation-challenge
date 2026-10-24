"""[opus-pass] offline: INTERPOLATIVE old tier (columns J ~ A J_S with A = Y K^-1, J_S = the pivot rows of J: no QR, no Fj = Q'J product)
vs today's ORTHOGONAL PROJECTION (Q = orth(Y), Fj = Q'J) on the captured join Grams. The W-Gram of the interpolation residual
J - A J_S is exactly G - G_nys (Nystrom error); the projection residual's is (I-P) G (I-P). Reports lost energy tr(.)/tr(G) and
Frobenius |.|/|G| for both, and the eigen optimum. Also the conditioning of K = G[S,S] (float32 solve risk)."""
import glob
import numpy as np
from pathlib import Path
C = Path(__file__).resolve().parent.parent/'opus_sketchgb_v1/cap'
for f in sorted(glob.glob(str(C/'join_*.npz'))):
    z = np.load(f); J, w, Qp, core = z['J'], z['w'], z['Qp'], z['core']
    if Qp.size == 0: Qp = np.zeros((J.shape[0], 0))
    G = (J * w[None, :]) @ J.T + (Qp @ core @ Qp.T if Qp.shape[1] else 0); G = 0.5 * (G + G.T)
    r = 384
    S = np.sort(np.argsort(-np.diag(G), kind='stable')[:r])
    Y = G[:, S]; K = G[np.ix_(S, S)]
    Gn = Y @ np.linalg.solve(K, Y.T); Gn = 0.5 * (Gn + Gn.T)
    Q, _ = np.linalg.qr(Y); PG = Q @ (Q.T @ G); PGP = PG @ Q @ Q.T
    ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    tG, nG = np.trace(G), np.linalg.norm(G)
    tr_int, fro_int = np.trace(G - Gn) / tG, np.linalg.norm(G - Gn) / nG
    tr_prj, fro_prj = (tG - np.trace(PG)) / tG, np.linalg.norm(G - PGP) / nG
    tr_eig = ev[r:].sum() / tG
    print('%s n%d | lost energy: interp %.3e proj %.3e (x%.2f) eig %.3e | Frobenius: interp %.3e proj %.3e (x%.2f) | cond(K) %.1e' % (
        Path(f).stem, G.shape[0], tr_int, tr_prj, tr_int / tr_prj, tr_eig, fro_int, fro_prj, fro_int / fro_prj, np.linalg.cond(K)))
