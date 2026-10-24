"""TAG opus_engineer a33 (2026-10-02): which cheap form of the transported star s2 = (W o W)' A' W is closest to the exact one?
Offline float64 numpy on the planes stored by dump_planes.py (no native run; set the BLAS threads to 1 before calling).
Forms, each with the row-sum part m (W' A 1)' exact:
  sums      that part alone (lever SS0)
  rowdir    one direction, no QR: a = A 1 / |A 1| (the row-sum direction), b = A' a;  s2 ~ ((W o W)' b) (W' a)'
  rowdir2   the same after one more round trip: b^ = b / |b|, a2 = A b^;  s2 ~ ((W o W)' b^) (W' a2)' + m (W' (A 1 - a2 (b^' 1)))'
  k<r>      the parent's factorization (r largest columns, one power step) at rank r, plus the missed row sums (lever SS<r>)
  plain32   the parent as fired (rank 32, no row-sum column)
Printed per layer: relative error of the transported plane off the diagonal, and of its diagonal (the per-unit third cumulant).
usage: forms_offline.py <planes.npz> <rank> [<out.json (new)>]"""
from pathlib import Path
import json
import sys
import numpy as np

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
planes, rank = np.load(sys.argv[1]), int(sys.argv[2])
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
RANKS = (1, 2, 4, 8, 16, 32)


def parent_factors(S, K):
    cn = (S * S).sum(0)
    cols = np.sort(np.argsort(-cn, kind='stable')[:K])
    Q = np.linalg.qr(S[:, cols])[0]
    Q = np.linalg.qr(S @ np.linalg.qr(S.T @ Q)[0])[0]
    return Q, Q.T @ S


rows = []
for l in range(1, 15):
    S = planes['star_into_layer_%02d' % l].astype(np.float64); W = weights[l].astype(np.float64); n = len(S)
    wt2 = W.T * W.T
    m = wt2.mean(1); one = np.ones(n)
    exact = wt2 @ S.T @ W
    off = ~np.eye(n, dtype=bool)
    rel = lambda X: (float(np.sqrt(((X - exact)[off] ** 2).mean() / (exact[off] ** 2).mean())), float(np.linalg.norm(np.diag(X - exact)) / np.linalg.norm(np.diag(exact))))
    forms = {}
    r1 = S @ one
    forms['sums'] = np.outer(m, r1 @ W)
    a = r1 / np.linalg.norm(r1); b = S.T @ a
    forms['rowdir'] = np.outer(wt2 @ b, a @ W)
    bh = b / np.linalg.norm(b); a2 = S @ bh
    forms['rowdir2'] = np.outer(wt2 @ bh, a2 @ W) + np.outer(m, (r1 - a2 * bh.sum()) @ W)
    for K in RANKS:
        Q, B = parent_factors(S, K)
        low = wt2 @ B.T @ (Q.T @ W)
        if K == 32:
            forms['plain32'] = low
        forms['k%d' % K] = low + np.outer(m, (r1 - Q @ B.sum(1)) @ W)
    sv = np.linalg.svd(S, compute_uv=False)
    row = dict(layer=l, rest=rel(forms['sums'])[0], top_singular_share=[float((sv[:k] ** 2).sum() / (sv ** 2).sum()) for k in (1, 2, 4, 8, 32)], forms={k: rel(v) for k, v in forms.items()})
    rows.append(row)
    print('layer %2d | off-diagonal error: %s | top 1/2/4/8/32 singular share %s' % (l, ' '.join('%s %.4f' % (k, v[0]) for k, v in row['forms'].items()), ' '.join('%.3f' % x for x in row['top_singular_share'])), flush=True)
print()
for row in rows:
    print('layer %2d | diagonal error: %s' % (row['layer'], ' '.join('%s %.4f' % (k, v[1]) for k, v in row['forms'].items())))
if len(sys.argv) > 3:
    out = Path(sys.argv[3]); assert not out.exists()
    out.write_text(json.dumps(dict(planes=sys.argv[1], rank=rank, rows=rows), indent=1))
