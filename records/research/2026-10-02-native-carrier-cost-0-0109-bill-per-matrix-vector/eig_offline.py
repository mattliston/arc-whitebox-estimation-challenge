"""TAG opus_engineer a36 (2026-10-02): how many power sweeps does a native carrier need for the top-K eigenvectors of the
pre-activation covariance? (eigh is billed 9 n^3 = 3.85% of the bill per call; a sweep is one covariance product with the block.)
Offline float64 numpy on the state stored by dump_state.py (no native run; set the BLAS threads to 1).
Scheme (the one in NCAR_SPEC.md): layer 1 starts from a seeded Gaussian block with `first` sweeps; layer l > 1 starts from the
previous layer's block pushed through the gates and the weights, U0 = orth(W_l' (g_{l-1} o U_{l-1})), then `s` sweeps
U <- orth(C_l U), then a Rayleigh-Ritz rotation inside the block (a K x K eigenproblem) so that the block's columns are ordered.
The block has K + p columns (p extra columns speed up the trailing ones); the first K Ritz vectors are compared with the exact
top-K eigenvectors: share of the exact K-dimensional subspace held, and the same for each exact eigenvector separately (min).
usage: eig_offline.py <state.npz> <rank> [<out.json (new)>]"""
from pathlib import Path
import json
import sys
import numpy as np

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
state, rank = np.load(sys.argv[1]), int(sys.argv[2])
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False).astype(np.float64)
C = {l: state['C_%02d' % l].astype(np.float64) for l in range(1, 15)}
C = {l: (M + M.T) / 2 for l, M in C.items()}
gate = {l: state['aux_%02d' % l][0] for l in range(1, 15)}
exact = {}
for l in range(1, 15):
    lam, V = np.linalg.eigh(C[l])
    exact[l] = (lam[::-1].copy(), V[:, ::-1].copy())
print('exact spectrum, eigenvalues 1, 2, 8, 9, 16, 17, 32 relative to the trace:')
for l in (1, 2, 5, 9, 12, 14):
    lam = exact[l][0]
    print('  layer %2d: %s' % (l, ' '.join('%.4f' % (lam[i - 1] / lam.sum()) for i in (1, 2, 8, 9, 16, 17, 32))))


def ritz(Cl, U):
    T = U.T @ (Cl @ U)
    w, R = np.linalg.eigh((T + T.T) / 2)
    return U @ R[:, ::-1]


def run(K, p, s, first):
    rng = np.random.default_rng(7)
    out = []
    U = None
    for l in range(1, 15):
        if U is None:
            U = np.linalg.qr(rng.standard_normal((1024, K + p)))[0]; sweeps = first
        else:
            U = np.linalg.qr(weights[l].T @ (gate[l - 1][:, None] * U))[0]; sweeps = s
        for _ in range(sweeps):
            U = np.linalg.qr(C[l] @ U)[0]
        U = ritz(C[l], U)
        E = exact[l][1][:, :K]
        ov = E.T @ U[:, :K]
        out.append((float((ov ** 2).sum() / K), float((ov ** 2).sum(1).min()), float(((E.T @ U) ** 2).sum() / K)))
    return out


rows = []
for K, p in ((8, 0), (8, 8), (16, 0), (16, 8), (16, 16)):
    for s in (0, 1, 2, 3):
        r = run(K, p, s, first=4)
        rows.append(dict(K=K, extra=p, sweeps=s, share=[x[0] for x in r], worst_vector=[x[1] for x in r], share_in_whole_block=[x[2] for x in r]))
        pick = (1, 2, 5, 9, 12, 14)
        print('K %2d + %2d extra, %d sweeps: share of the exact top-K subspace held at layers 1 2 5 9 12 14: %s | worst single eigenvector: %s | block products a layer: %d' % (
            K, p, s, ' '.join('%.3f' % r[l - 1][0] for l in pick), ' '.join('%.3f' % r[l - 1][1] for l in pick), (K + p) * (s + 2)), flush=True)
if len(sys.argv) > 3:
    out = Path(sys.argv[3]); assert not out.exists()
    out.write_text(json.dumps(dict(state=sys.argv[1], rank=rank, rows=rows), indent=1))
